#!/usr/bin/env python3
"""t4b_train.py: 8xH200 DDP trainer for Qwen/Qwen3.5-4B (4.21B pure causal Qwen3_5ForCausalLM).

- LoRA r=64 / alpha=128 on all 12 LM linear target types across all 32 layers (24 GDN + 8 full attn).
- 255-way fp32 readout head initialized from lm_head rows of 255 boundary-stable single-token codes.
- Sandwich causal layout (Questions -> Context -> Selected Question + Options -> 'Answer: (').
- Pure soft-label KL/CE to karpo-decision-31B probabilities (weighted by row weight w).
- Automatic length-sorted right-padded micro-batching (verified against batch=1 in --smoke-parity,
  with automatic fallback to micro-batch=1 if parity exceeds 0.05).
- Periodic quick unseen-dev accuracy monitor (512 rows from dev.req.jsonl.gz, sharded across 8 GPUs).
- Full spot durability: periodic S3 checkpoints + SIGTERM/SIGINT/IMDSv2/07:30Z/exception saves + resume.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t4b_layout as LYT

STOP = {"why": None}


def log(path, rec):
    rec = {"t": dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%SZ"), **rec}
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def spot_watch(interval=5):
    import urllib.request

    while STOP["why"] is None:
        try:
            tok = urllib.request.urlopen(
                urllib.request.Request(
                    "http://169.254.169.254/latest/api/token",
                    method="PUT",
                    headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
                ),
                timeout=2,
            ).read().decode()
            r = urllib.request.urlopen(
                urllib.request.Request(
                    "http://169.254.169.254/latest/meta-data/spot/instance-action",
                    headers={"X-aws-ec2-metadata-token": tok},
                ),
                timeout=2,
            )
            if r.status == 200:
                STOP["why"] = "spot_notice"
                return
        except Exception:
            pass
        time.sleep(interval)


def inner_backbone(model):
    m = model
    if type(m).__name__.startswith("Peft"):
        m = m.base_model.model
    return m.model


class IndexedDataset:
    """Memory-lean line-offset indexed JSONL reader (only offsets in RAM, works cleanly across 8 ranks)."""

    def __init__(self, path: str, seed: int, rows_per_step: int, world: int, rank: int):
        self.path = path
        self.offsets = []
        with open(path, "rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(pos)
        self.fh = open(path, "rb")
        self.order = list(range(len(self.offsets)))
        random.Random(seed).shuffle(self.order)
        self.R, self.world, self.rank = rows_per_step, world, rank

    def get(self, idx: int) -> dict:
        self.fh.seek(self.offsets[idx])
        return json.loads(self.fh.readline().decode("utf-8"))

    def step_rows(self, step: int) -> list[dict]:
        n = len(self.order)
        idxs = [self.order[(step * self.R + j) % n] for j in range(self.R)]
        return [self.get(i) for i in idxs[self.rank :: self.world]]


def prepare_example(
    tok,
    codes: list[str],
    row: dict,
    max_tokens: int,
    shuffle_choice: bool = True,
    layout: str = LYT.LAYOUT_V1,
    read_twice_tokens: int = 0,
):
    opts = row["options"]
    p = row["p"]
    k = len(opts)
    if shuffle_choice and row.get("kind", "choice") == "choice":
        order = LYT.permute_order(k, LYT.seed_of("t4b-train", row["id"]))
        opts = [opts[o] for o in order]
        p = [p[o] for o in order]
    ids, _ = LYT.build_prompt_ids(
        tok,
        row["state"],
        row["preview"],
        row["instructions"],
        opts,
        codes,
        layout=layout,
        read_twice_tokens=read_twice_tokens,
    )
    if not ids or len(ids) > max_tokens:
        return None, len(ids)
    return (ids, p, float(row.get("w", 1.0))), len(ids)


def lr_at(step, total, base, warmup, floor):
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return base * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))


def forward_microbatch(torch, backbone, head, batch_items, pad_id: int, dev, max_batch_tokens: int, bidir: bool = False):
    """Run one right-padded micro-batch and return list of (logits[:k], target_tensor, weight)."""
    bsz = len(batch_items)
    lens = [len(it[0]) for it in batch_items]
    maxlen = max(lens)
    if bsz == 1:
        ids = torch.tensor([batch_items[0][0]], dtype=torch.long, device=dev)
        out = backbone(input_ids=ids, use_cache=False)
        hs = out.last_hidden_state[:, -1, :].float()
    else:
        ids = torch.full((bsz, maxlen), pad_id, dtype=torch.long, device=dev)
        mask = torch.zeros((bsz, maxlen), dtype=torch.long, device=dev)
        for b, it in enumerate(batch_items):
            L = len(it[0])
            ids[b, :L] = torch.tensor(it[0], dtype=torch.long, device=dev)
            mask[b, :L] = 1
        out = backbone(input_ids=ids, attention_mask=mask, use_cache=False)
        last_idx = torch.tensor([L - 1 for L in lens], dtype=torch.long, device=dev)
        hs = out.last_hidden_state[torch.arange(bsz, device=dev), last_idx].float()
    zs = head(hs)
    res = []
    for b, it in enumerate(batch_items):
        p = torch.tensor(it[1], dtype=torch.float32, device=dev)
        p = p / p.sum()
        res.append((zs[b, : len(it[1])], p, it[2]))
    return res


def eval_quick_dev(
    torch,
    dist,
    model,
    head,
    tok,
    codes,
    dev_rows,
    dev,
    rank,
    world,
    max_tokens,
    layout: str = LYT.LAYOUT_V1,
    read_twice_tokens: int = 0,
):
    model.eval()
    backbone = inner_backbone(model)
    mine = dev_rows[rank::world]
    correct, total, loss_sum = 0.0, 0.0, 0.0
    with torch.inference_mode():
        for r in mine:
            req = r["request"]
            qid, q = next(iter(req["questions"].items()))
            keys, opt_texts = LYT.options_of(q)
            if len(keys) > len(codes):
                continue
            ids, _ = LYT.build_prompt_ids(
                tok,
                LYT.describe(req["state"]),
                LYT.question_preview(req["questions"]),
                LYT.instructions_of(q),
                opt_texts,
                codes,
                layout=layout,
                read_twice_tokens=read_twice_tokens,
            )
            if not ids or len(ids) > max_tokens:
                continue
            t_ids = torch.tensor([ids], dtype=torch.long, device=dev)
            h = backbone(input_ids=t_ids, use_cache=False).last_hidden_state[0, -1].float()
            z = head(h)[: len(keys)]
            pred_k = keys[int(z.argmax().item())]
            g = r["gold"]
            gold_k = max(keys, key=lambda kk: (g.get(kk, 0.0), -keys.index(kk)))
            gold_t = torch.tensor([float(g.get(kk, 0.0)) for kk in keys], dtype=torch.float32, device=dev)
            if gold_t.sum() > 0:
                gold_t = gold_t / gold_t.sum()
                loss_sum += float(-(gold_t * torch.log_softmax(z, -1)).sum().item())
            correct += float(pred_k == gold_k)
            total += 1.0
    stats = torch.tensor([correct, total, loss_sum], device=dev, dtype=torch.float64)
    dist.all_reduce(stats)
    model.train()
    n = max(1.0, float(stats[1]))
    return {"acc": round(100.0 * float(stats[0]) / n, 3), "loss": round(float(stats[2]) / n, 4), "n": int(stats[1])}


def make_microbatches(items, max_bsz: int, max_batch_tokens: int):
    if max_bsz <= 1 or len(items) <= 1:
        return [[it] for it in items]
    # Length-sort so right-padding overhead stays minimal
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
    batches, cur = [], []
    for i in order:
        it = items[i]
        L = len(it[0])
        if cur and (len(cur) >= max_bsz or L * (len(cur) + 1) > max_batch_tokens):
            batches.append(cur)
            cur = [it]
        else:
            cur.append(it)
    if cur:
        batches.append(cur)
    return batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--s3", default=None)
    ap.add_argument("--dev-req", default=None, help="dev.req.jsonl.gz for quick unseen-family dev monitoring")
    ap.add_argument("--dev-n", type=int, default=512)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--steps", type=int, default=3516)
    ap.add_argument("--rows-per-step", type=int, default=128)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--max-batch-tokens", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--floor", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--layout", default=LYT.LAYOUT_V1)
    ap.add_argument("--bidir", action="store_true")
    ap.add_argument("--read-twice-tokens", type=int, default=0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--smoke-parity", action="store_true")
    a = ap.parse_args()

    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file, save_file
    from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lrank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(lrank)
    dev = torch.device("cuda", lrank)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tlog = str(out / "train_log.jsonl") if rank == 0 else None

    tok = AutoTokenizer.from_pretrained(a.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    codes, code_ids = LYT.build_codes(tok, LYT.N_CODES)

    cfg = AutoConfig.from_pretrained(a.model).get_text_config()
    model, info = Qwen3_5ForCausalLM.from_pretrained(
        a.model,
        config=cfg,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    missing = [k for k in (info.get("missing_keys") or []) if k != "lm_head.weight"]
    if missing:
        raise SystemExit(f"Missing keys on Qwen3_5ForCausalLM load: {missing[:5]}")
    model = model.to(dev)
    if a.bidir:
        model.set_attn_implementation(LYT.register_bidir())
    for p in model.parameters():
        p.requires_grad_(False)

    # 255-way fp32 readout initialized from lm_head[code_ids]
    lm = model.get_output_embeddings()
    W0 = lm.weight.detach()[code_ids].float().clone()
    head = torch.nn.Linear(W0.shape[1], len(code_ids), bias=True).to(dev, torch.float32)
    with torch.no_grad():
        head.weight.copy_(W0)
        head.bias.zero_()

    # Discover all LM linear projections inside model.layers
    names = sorted({n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "layers." in n and "lm_head" not in n})
    assert len(names) >= 10, f"Too few LM linear target names discovered: {names}"
    target_re = r".*layers\.\d+\..*\.(" + "|".join(names) + r")$"
    lcfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, target_modules=target_re, bias="none")

    ds = IndexedDataset(a.rows, a.seed, a.rows_per_step, world, rank)

    if a.smoke_parity:
        if rank == 0:
            model.eval()
            bb = inner_backbone(model)
            head_diffs = []
            items = []
            bidir_calls_per_fwd = 0
            for r in ds.step_rows(0)[:12]:
                it, n = prepare_example(
                    tok, codes, r, a.max_tokens, shuffle_choice=True, layout=a.layout, read_twice_tokens=a.read_twice_tokens
                )
                if it is None or n > 4096:
                    continue
                items.append(it)
                t_ids = torch.tensor([it[0]], dtype=torch.long, device=dev)
                c0 = LYT.BIDIR_STATE["calls"]
                with torch.inference_mode():
                    z_head = head(bb(input_ids=t_ids, use_cache=False).last_hidden_state[0, -1].float())
                    c1 = LYT.BIDIR_STATE["calls"]
                    z_lm = model(input_ids=t_ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()[code_ids]
                bidir_calls_per_fwd = c1 - c0
                head_diffs.append(float((z_head - z_lm).abs().max()))
                if len(items) >= 4:
                    break
            # Check right-padded micro-batch=4 vs batch=1 parity
            batch_max_diff = 0.0
            if len(items) >= 4:
                b4 = items[:4]
                with torch.inference_mode():
                    z_single = [forward_microbatch(torch, bb, head, [x], pad_id, dev, a.max_batch_tokens, bidir=a.bidir)[0][0] for x in b4]
                    z_batch = [x[0] for x in forward_microbatch(torch, bb, head, b4, pad_id, dev, a.max_batch_tokens, bidir=a.bidir)]
                batch_max_diff = max(float((a_ - b_).abs().max()) for a_, b_ in zip(z_single, z_batch))
            res = {
                "event": "smoke_parity",
                "n_codes": len(codes),
                "layout": a.layout,
                "bidir": bool(a.bidir),
                "bidir_calls_per_fwd": bidir_calls_per_fwd,
                "read_twice_tokens": int(a.read_twice_tokens),
                "lora_target_names": names,
                "head_max_abs_diff": round(max(head_diffs) if head_diffs else 999.0, 6),
                "rightpad_batch4_max_abs_diff": round(batch_max_diff, 6),
                "batch_safe": bool(batch_max_diff < 0.05),
            }
            (out / "smoke_parity.json").write_text(json.dumps(res, indent=2))
            log(tlog, res)
            assert res["head_max_abs_diff"] < 0.1, f"Head parity failed: {res}"
            if a.bidir:
                assert bidir_calls_per_fwd == 8, f"Expected 8 bidir full-attn calls per fwd, got {bidir_calls_per_fwd}"
        dist.barrier()
        dist.destroy_process_group()
        return

    # Auto-check micro-batch safety on rank 0
    micro_bsz = a.micro_batch
    if micro_bsz > 1:
        b_ok = torch.tensor([1], device=dev, dtype=torch.int32)
        if rank == 0:
            model.eval()
            bb = inner_backbone(model)
            sample_its = []
            for r in ds.step_rows(0)[:12]:
                it, n = prepare_example(
                    tok, codes, r, min(4096, a.max_tokens), shuffle_choice=True, layout=a.layout, read_twice_tokens=a.read_twice_tokens
                )
                if it is not None:
                    sample_its.append(it)
                if len(sample_its) == 4:
                    break
            if len(sample_its) == 4:
                with torch.inference_mode():
                    zs1 = [forward_microbatch(torch, bb, head, [x], pad_id, dev, a.max_batch_tokens, bidir=a.bidir)[0][0] for x in sample_its]
                    zsb = [x[0] for x in forward_microbatch(torch, bb, head, sample_its, pad_id, dev, a.max_batch_tokens, bidir=a.bidir)]
                diff = max(float((x - y).abs().max()) for x, y in zip(zs1, zsb))
                if diff >= 0.05:
                    b_ok[0] = 0
                    log(tlog, {"event": "microbatch_fallback", "rightpad_diff": round(diff, 6), "using_micro_batch": 1})
                else:
                    log(tlog, {"event": "microbatch_verified", "rightpad_diff": round(diff, 6), "using_micro_batch": micro_bsz})
        dist.broadcast(b_ok, src=0)
        if int(b_ok[0]) == 0:
            micro_bsz = 1

    torch.manual_seed(a.seed)
    model = get_peft_model(model, lcfg)
    torch.manual_seed(a.seed + rank)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    lora_params = [p for _, p in model.named_parameters() if p.requires_grad]
    for p in lora_params:
        if p.dtype != torch.float32:
            p.data = p.data.float()
    head_params = list(head.parameters())
    for p in lora_params + head_params:
        dist.broadcast(p.data, src=0)

    opt = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": a.lr, "base": a.lr},
            {"params": head_params, "lr": a.head_lr, "base": a.head_lr},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=a.wd,
    )

    step0 = 0
    if a.resume:
        rp = Path(a.resume)
        set_peft_model_state_dict(model, load_file(str(rp / "adapter.safetensors"), device=str(dev)))
        head.load_state_dict(torch.load(rp / "readout.pt", map_location=dev))
        opt.load_state_dict(torch.load(rp / "optim.pt", map_location=dev))
        step0 = int(json.loads((rp / "state.json").read_text())["step"])

    # Optional quick unseen-family dev subset
    dev_rows = []
    if a.dev_req and os.path.exists(a.dev_req):
        all_dev = [json.loads(l) for l in gzip.open(a.dev_req, "rt", encoding="utf-8") if l.strip()]
        rng_d = random.Random(20261119)
        rng_d.shuffle(all_dev)
        dev_rows = all_dev[: a.dev_n]

    n_lora_mods = sum(1 for k in get_peft_model_state_dict(model) if k.endswith(".lora_A.weight"))
    log(
        tlog,
        {
            "event": "start",
            "world": world,
            "examples": len(ds.offsets),
            "lora_params": sum(p.numel() for p in lora_params),
            "lora_modules": n_lora_mods,
            "lora_targets": names,
            "head_params": sum(p.numel() for p in head_params),
            "micro_batch": micro_bsz,
            "step0": step0,
            "steps": a.steps,
            "n_codes": len(codes),
            "dev_n": len(dev_rows),
            "args": vars(a),
        },
    )

    def handler(sig, frame):
        STOP["why"] = STOP["why"] or f"signal_{sig}"

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    if rank == 0:
        threading.Thread(target=spot_watch, daemon=True).start()
    mirror = {"p": None}

    def save(step, why, barrier=True):
        if rank == 0:
            d = out / f"ckpt_{step:05d}"
            d.mkdir(parents=True, exist_ok=True)
            save_file({k: v.detach().to("cpu").contiguous() for k, v in get_peft_model_state_dict(model).items()}, str(d / "adapter.safetensors"))
            torch.save(head.state_dict(), d / "readout.pt")
            torch.save(opt.state_dict(), d / "optim.pt")
            (d / "state.json").write_text(
                json.dumps(
                    {
                        "step": step,
                        "why": why,
                        "lora_r": a.lora_r,
                        "lora_alpha": a.lora_alpha,
                        "codes": codes,
                        "code_ids": code_ids,
                        "layout": a.layout,
                        "bidir": bool(a.bidir),
                        "causal": not bool(a.bidir),
                        "read_twice_tokens": int(a.read_twice_tokens),
                        "batch_safe": bool(micro_bsz > 1),
                        "rows": a.rows,
                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                )
            )
            lcfg.save_pretrained(str(d))
            if (out / "train_log.jsonl").exists():
                shutil.copy2(out / "train_log.jsonl", d / "train_log.jsonl")
            log(tlog, {"event": "save", "step": step, "why": why, "dir": str(d)})
            if a.s3:
                if mirror["p"] is not None and mirror["p"].poll() is None:
                    mirror["p"].wait()
                mirror["p"] = subprocess.Popen(["aws", "s3", "sync", "--quiet", str(d), f"{a.s3}/{d.name}/"])
        if barrier:
            dist.barrier()

    # Step-0 zero-shot unseen-dev baseline
    if step0 == 0 and dev_rows:
        dv0 = eval_quick_dev(
            torch, dist, model, head, tok, codes, dev_rows, dev, rank, world, a.max_tokens, layout=a.layout, read_twice_tokens=a.read_twice_tokens
        )
        log(tlog, {"event": "dev_eval", "step": 0, **dv0})

    bb = inner_backbone(model)
    forced_day = None
    model.train()
    step = step0
    try:
        while step < a.steps:
            t0 = time.time()
            sc = lr_at(step, a.steps, 1.0, a.warmup, a.floor)
            for g in opt.param_groups:
                g["lr"] = g["base"] * sc

            raw_rows = ds.step_rows(step)
            items, skipped, toks = [], 0, 0
            for r in raw_rows:
                it, n = prepare_example(
                    tok, codes, r, a.max_tokens, shuffle_choice=True, layout=a.layout, read_twice_tokens=a.read_twice_tokens
                )
                if it is None:
                    skipped += 1
                    continue
                items.append(it)
                toks += n

            # Compute global weight sum for exact weighted mean across all 8 ranks
            w_local = sum(it[2] for it in items)
            w_global = torch.tensor([w_local], device=dev, dtype=torch.float64)
            dist.all_reduce(w_global)
            denom = max(1e-6, float(w_global[0])) / world  # divide each rank's grad by denom, then average over world -> exact global mean!

            loss_sum, correct, used = 0.0, 0.0, 0
            for mb in make_microbatches(items, micro_bsz, a.max_batch_tokens):
                outs = forward_microbatch(torch, bb, head, mb, pad_id, dev, a.max_batch_tokens, bidir=a.bidir)
                mb_loss = None
                for z, t_p, w_i in outs:
                    li = -(t_p * torch.log_softmax(z, -1)).sum()
                    wli = li * (w_i / denom)
                    mb_loss = wli if mb_loss is None else (mb_loss + wli)
                    loss_sum += float(li.item()) * w_i
                    correct += float(int(z.argmax().item()) == int(t_p.argmax().item()))
                    used += 1
                if mb_loss is not None:
                    mb_loss.backward()

            stats = torch.tensor([loss_sum, correct, used, toks, skipped, w_local], device=dev, dtype=torch.float64)
            dist.all_reduce(stats)
            params = lora_params + head_params
            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad)
                p.grad.div_(world)
            gn = torch.nn.utils.clip_grad_norm_(params, a.clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            u = max(1.0, float(stats[2]))
            wt = max(1e-6, float(stats[5]))
            log(
                tlog,
                {
                    "event": "step",
                    "step": step,
                    "loss": round(float(stats[0]) / wt, 4),
                    "acc": round(float(stats[1]) / u, 4),
                    "rows": int(stats[2]),
                    "skipped": int(stats[4]),
                    "tokens": int(stats[3]),
                    "gnorm": round(float(gn), 3),
                    "lr": round(opt.param_groups[0]["lr"], 7),
                    "s": round(time.time() - t0, 2),
                },
            )

            if dev_rows and (step % a.eval_every == 0 or step == a.steps):
                dv = eval_quick_dev(
                    torch, dist, model, head, tok, codes, dev_rows, dev, rank, world, a.max_tokens, layout=a.layout, read_twice_tokens=a.read_twice_tokens
                )
                log(tlog, {"event": "dev_eval", "step": step, **dv})

            now = dt.datetime.now(dt.timezone.utc)
            flags = torch.tensor(
                [1 if STOP["why"] else 0, 1 if (now.hour == 7 and 30 <= now.minute < 40 and forced_day != now.date()) else 0],
                device=dev,
            )
            dist.all_reduce(flags, op=dist.ReduceOp.MAX)
            if flags[1]:
                forced_day = now.date()
                save(step, "forced_0730")
            elif step % a.save_every == 0 or step == a.steps:
                save(step, "periodic" if step < a.steps else "final")
            if flags[0]:
                save(step, STOP["why"] or "stop_flag")
                break
    except Exception:
        log(tlog, {"event": "exception", "trace": traceback.format_exc()[-2000:]})
        try:
            save(step, "exception", barrier=False)
        finally:
            raise

    if rank == 0 and mirror["p"] is not None:
        mirror["p"].wait()
    log(tlog, {"event": "done", "step": step, "why": STOP["why"]})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
