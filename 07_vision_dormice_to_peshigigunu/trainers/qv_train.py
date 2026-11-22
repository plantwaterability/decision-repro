"""Qwen3.8-27B vision decision trainer: LoRA on the language model + a 26-way last-token head, soft-label CE,
frozen vision tower, non-causal full-attention layers. torchrun, one full bf16 replica per rank, micro-batch 1,
gradient accumulation to --rows-per-step, manual all-reduce of the trainable grads.

Durability (standing rule): saves every --save-every steps, on SIGTERM/SIGINT, on an IMDSv2 spot notice, on any
exception, and once in the 07:30-07:40Z window before the daily 08:05Z event; every save is mirrored to --s3 in the
background. --resume picks up adapter, head, optimizer, step and data position.
"""
import argparse
import datetime as dt
import json
import math
import os
import random
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qv_common as C  # noqa: E402
import qv_ocr as O  # noqa: E402

STOP = {"why": None}


def log(path, rec):
    rec = {"t": dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%SZ"), **rec}
    line = json.dumps(rec)
    print(line, flush=True)
    if path:
        with open(path, "a") as f:
            f.write(line + "\n")


def spot_watch(interval=5):
    import urllib.request

    while STOP["why"] is None:
        try:
            tok = urllib.request.urlopen(urllib.request.Request("http://169.254.169.254/latest/api/token", method="PUT",
                                                                headers={"X-aws-ec2-metadata-token-ttl-seconds": "300"}), timeout=2).read().decode()
            r = urllib.request.urlopen(urllib.request.Request("http://169.254.169.254/latest/meta-data/spot/instance-action",
                                                              headers={"X-aws-ec2-metadata-token": tok}), timeout=2)
            if r.status == 200:
                STOP["why"] = "spot_notice"
                return
        except Exception:
            pass
        time.sleep(interval)


class Rows:
    """Deterministic order: shuffled once by seed; step s uses global rows [s*R, (s+1)*R); rank r takes r::world."""

    def __init__(self, manifest, seed, rows_per_step, world, rank):
        self.rows = [json.loads(l) for l in open(manifest) if l.strip()]
        self.order = list(range(len(self.rows)))
        random.Random(seed).shuffle(self.order)
        self.R, self.world, self.rank = rows_per_step, world, rank

    def step_rows(self, step):
        n = len(self.order)
        idx = [self.order[(step * self.R + j) % n] for j in range(self.R)]
        return [self.rows[i] for i in idx[self.rank::self.world]]


def encode(proc, row, img_root, max_tokens, ocr=None):
    pils = [C.load_image(os.path.join(img_root, p)) for p in row["images"]]
    pr = row.get("prompt") or {}
    question = O.ocr_prefix(ocr, row["images"]) + row["question"]  # OCR block before the question (--ocr-train), else unchanged
    text = C.render(proc, question, row["options"], len(pils), pr.get("options_in_image", False), pr.get("readout"))
    enc = proc(text=[text], images=pils if pils else None, return_tensors="pt")
    n = int(enc["input_ids"].shape[1])
    return (enc, n) if n <= max_tokens else (None, n)


def target(row, k, eps):
    import torch

    t = torch.full((k,), 0.0)
    soft = row.get("soft")
    if soft:
        for i, l in enumerate(C.LETTERS[:k]):
            t[i] = float(soft.get(l, 0.0))
        t = t / t.sum()
    else:
        t[C.LETTERS.index(row["gold"])] = 1.0
    if eps > 0:
        t = (1 - eps) * t + eps / k
    return t


def lr_at(step, total, base, warmup, floor):
    if step < warmup:
        return base * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


def kit_eval(model, head, proc, rows, root, rank, world, dev, out_path, ocr=None):
    """Score kit vision rows with the training model (same render + head + kit image cap as qv_engine). With an OCR cache
    the OCR block goes before the question exactly as in training. Saves the argmax and the option probabilities (T = 1)."""
    import torch
    import torch.distributed as dist
    from decision_index.vision import images as kimg
    from decision_index.vision.score import score

    model.eval()
    mine, minep = {}, {}
    for r in rows[rank::world]:
        q = r["questions"]["answer"]
        keys = list(q["criteria"])
        pr = r.get("prompt") or {}
        pils = [kimg.pil(p, root, C.MAX_PIXELS).convert("RGB") for p in r["images"]]
        question = O.ocr_prefix(ocr, r["images"]) + q["instructions"]
        text = C.render(proc, question, [q["criteria"][k] for k in keys], len(pils), pr.get("options_in_image", False), pr.get("readout"))
        enc = proc(text=[text], images=pils if pils else None, return_tensors="pt")
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            z = head(C.last_hidden(model, enc).float())[: len(keys)]
        mine[r["id"]] = keys[int(z.argmax())]
        minep[r["id"]] = [round(float(x), 6) for x in torch.softmax(z, -1)]
    model.train()
    allp = [None] * world
    dist.all_gather_object(allp, (mine, minep))
    if rank != 0:
        return None
    preds = {k: v for d, _ in allp for k, v in d.items()}
    probs = {k: v for _, d in allp for k, v in d.items()}
    res = score(rows, preds)
    Path(out_path).write_text(json.dumps({"index": res["public_vision_index"], "benchmarks": res["benchmarks"], "preds": preds,
                                          "probs": probs, "ocr": ocr is not None}))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--s3", default=None)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--rows-per-step", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--floor", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--init-from", default=None, help="checkpoint dir: load adapter + head only (fresh optimizer, step 0)")
    ap.add_argument("--merge-from", default=None, help="comma list of ckpt dirs: merge their MEAN LoRA deltas into the base and start the head "
                                                         "from their mean head (a soup), then train a fresh LoRA on top")
    ap.add_argument("--expect-rows-sha256", default=None)
    ap.add_argument("--smoke-head-parity", type=int, default=0, help="rows to check head==LM-head letter logits at step 0, then exit")
    ap.add_argument("--no-bidir", action="store_true", help="stock causal attention (for the parity test only)")
    ap.add_argument("--kit", default=None, help="kit_vis dir (decision_index.vision.score / images) for in-trainer reads")
    ap.add_argument("--gate-file", default=None, help="kit vision rows (the 1,500-row gate set)")
    ap.add_argument("--gate-sha256", default=None)
    ap.add_argument("--gate-root", default=None)
    ap.add_argument("--gate-steps", default="300,600")
    ap.add_argument("--gate-continue", type=float, default=74.0)
    ap.add_argument("--gate-stop", type=float, default=70.0)
    ap.add_argument("--public-file", default=None, help="kit vision rows (full public noWG) read once at --public-step")
    ap.add_argument("--public-root", default=None)
    ap.add_argument("--public-step", type=int, default=1000)
    ap.add_argument("--ocr-train", default=None, help="OCR cache jsonl for the manifest's images: OCR block before the question")
    ap.add_argument("--ocr-gate", default=None, help="OCR cache jsonl for the gate rows' images")
    ap.add_argument("--ocr-public", default=None, help="OCR cache jsonl for the public rows' images")
    ap.add_argument("--gate-baseline", action="store_true", help="read the gate set at step 0 (the init model) and anchor the step gates to it")
    ap.add_argument("--gate-continue-rel", type=float, default=0.0, help="with --gate-baseline: continue if read >= baseline + this")
    ap.add_argument("--gate-stop-rel", type=float, default=-1.73, help="with --gate-baseline: stop if read < baseline + this")
    a = ap.parse_args()

    import hashlib
    import torch
    import torch.distributed as dist
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, get_peft_model_state_dict

    if a.expect_rows_sha256:
        got = hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest()
        if got != a.expect_rows_sha256:
            raise SystemExit(f"manifest sha256 {got} != expected {a.expect_rows_sha256}")

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lrank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(lrank)
    dev = torch.device("cuda", lrank)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tlog = str(out / "train_log.jsonl") if rank == 0 else None
    torch.manual_seed(a.seed + rank)

    proc = AutoProcessor.from_pretrained(a.model)
    C.set_pixel_cap(proc)
    tok = proc.tokenizer
    lids = C.letter_ids(tok)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
    if not a.no_bidir:
        model.set_attn_implementation({"text_config": C.register_bidir()})
    for p in model.parameters():
        p.requires_grad_(False)
    lm = C.lm_head_of(model)
    W0 = lm.weight.detach()[lids].float().clone()
    head = torch.nn.Linear(W0.shape[1], len(lids), bias=True).to(dev, torch.float32)
    with torch.no_grad():
        head.weight.copy_(W0)
        head.bias.zero_()
        if getattr(lm, "bias", None) is not None:
            head.bias.copy_(lm.bias.detach()[lids].float())

    merge_list = [x for x in (a.merge_from or "").split(",") if x]
    if merge_list:  # soup start: W_base += mean_m[(alpha/r) B_m A_m], module by module; head = mean of member heads
        from safetensors.torch import load_file

        sds, scales = [], []
        for ck in merge_list:
            st = json.loads((Path(ck) / "state.json").read_text())
            scales.append(st["lora_alpha"] / st["lora_r"])
            sds.append(load_file(str(Path(ck) / "adapter.safetensors"), device=str(dev)))
        with torch.no_grad():
            for k in sds[0]:
                if not k.endswith(".lora_A.weight"):
                    continue
                mod = model.get_submodule(k[len("base_model.model."):-len(".lora_A.weight")])
                w = mod.weight.float()
                kb = k.replace(".lora_A.weight", ".lora_B.weight")
                for sd, sc in zip(sds, scales):
                    w.add_(sd[kb].float() @ sd[k].float(), alpha=sc / len(sds))
                mod.weight.copy_(w.to(mod.weight.dtype))
                del w
            hs = [torch.load(Path(ck) / "head.pt", map_location=dev) for ck in merge_list]
            head.weight.copy_(sum(h["weight"].float() for h in hs) / len(hs))
            head.bias.copy_(sum(h["bias"].float() for h in hs) / len(hs))
        del sds
        torch.cuda.empty_cache()
    # LoRA on every linear layer of the language model only (vision tower + merger stay frozen and untouched)
    names = set()
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and ".language_model." in f".{n}." and "lm_head" not in n:
            names.add(n.split(".")[-1])
    target_re = r".*language_model.*\.(" + "|".join(sorted(names)) + r")$"
    lcfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, target_modules=target_re, bias="none")

    if a.smoke_head_parity:
        rows = Rows(a.manifest, a.seed, a.rows_per_step, world, rank)
        diffs = []
        model.eval()
        for row in rows.step_rows(0)[: a.smoke_head_parity]:
            enc, n = encode(proc, row, a.img_root, a.max_tokens)
            if enc is None:
                continue
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.inference_mode():
                h = C.last_hidden(model, enc).float()
                z_head = head(h)
                z_lm = model(**enc, use_cache=False, logits_to_keep=1).logits[0, -1].float()[lids]
            diffs.append(float((z_head - z_lm).abs().max()))
        log(tlog, {"event": "head_parity", "rank": rank, "rows": len(diffs), "max_abs_diff": max(diffs) if diffs else None,
                   "bidir_calls": C.BIDIR_CALLS["n"], "bidir": not a.no_bidir})
        dist.barrier()
        dist.destroy_process_group()
        return

    torch.manual_seed(a.seed)  # identical LoRA init on every rank (then broadcast from rank 0 as a belt-and-braces check)
    model = get_peft_model(model, lcfg)
    torch.manual_seed(a.seed + rank)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    for p in lora_params:
        if p.dtype != torch.float32:
            p.data = p.data.float()
    head_params = list(head.parameters())
    for p in lora_params + head_params:
        dist.broadcast(p.data, src=0)
    opt = torch.optim.AdamW([{"params": lora_params, "lr": a.lr, "base": a.lr},
                             {"params": head_params, "lr": a.head_lr, "base": a.head_lr}], betas=(0.9, 0.999), eps=1e-8, weight_decay=a.wd)
    n_lora = sum(p.numel() for p in lora_params)
    step0 = 0
    if a.init_from and not a.resume:
        from safetensors.torch import load_file

        ip = Path(a.init_from)
        set_peft_model_state_dict(model, load_file(str(ip / "adapter.safetensors"), device=str(dev)))
        head.load_state_dict(torch.load(ip / "head.pt", map_location=dev))
        log(tlog, {"event": "init_from", "dir": str(ip), "init_state": json.loads((ip / "state.json").read_text())})
    if a.resume:
        rp = Path(a.resume)
        from safetensors.torch import load_file

        set_peft_model_state_dict(model, load_file(str(rp / "adapter.safetensors"), device=str(dev)))
        head.load_state_dict(torch.load(rp / "head.pt", map_location=dev))
        opt.load_state_dict(torch.load(rp / "optim.pt", map_location=dev))
        step0 = json.loads((rp / "state.json").read_text())["step"]
    rows = Rows(a.manifest, a.seed, a.rows_per_step, world, rank)
    gate_rows = pub_rows = None
    if a.kit:
        sys.path.insert(0, a.kit)
    if a.gate_file:
        gb = Path(a.gate_file).read_bytes()
        if a.gate_sha256 and hashlib.sha256(gb).hexdigest() != a.gate_sha256:
            raise SystemExit("gate rows sha256 mismatch: fail closed")
        gate_rows = [json.loads(l) for l in gb.decode().splitlines() if l.strip()]
    if a.public_file:
        pub_rows = [json.loads(l) for l in open(a.public_file) if l.strip()]
    gate_steps = [int(x) for x in a.gate_steps.split(",") if x.strip().isdigit()]
    gate_done = {"decided": False}
    ocr_train = O.load_ocr(a.ocr_train)[0] if a.ocr_train else None
    ocr_gate = O.load_ocr(a.ocr_gate)[0] if a.ocr_gate else None
    ocr_public = O.load_ocr(a.ocr_public)[0] if a.ocr_public else None
    for nm, cache, rr in (("train", ocr_train, rows.rows), ("gate", ocr_gate, gate_rows), ("public", ocr_public, pub_rows)):
        miss = sum(p not in cache for r in rr or [] for p in r["images"]) if cache is not None else 0
        if miss:  # fail at start, not at the read hours later
            raise SystemExit(f"OCR cache {nm}: {miss} image refs missing")
    log(tlog, {"event": "start", "world": world, "rows": len(rows.rows), "lora_params": n_lora, "lora_targets": sorted(names),
               "step0": step0, "steps": a.steps, "bidir": not a.no_bidir, "letter_ids": lids[:4],
               "ocr": {k: (len(v), sum(1 for x in v.values() if x)) for k, v in
                       (("train", ocr_train), ("gate", ocr_gate), ("public", ocr_public)) if v is not None}})

    def handler(sig, frame):
        STOP["why"] = STOP["why"] or f"signal_{sig}"

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    if rank == 0:
        threading.Thread(target=spot_watch, daemon=True).start()
    mirror = {"p": None}

    def save(step, why, barrier=True):
        from safetensors.torch import save_file

        if rank == 0:
            d = out / f"ckpt_{step:05d}"
            d.mkdir(parents=True, exist_ok=True)
            sd = {k: v.detach().to("cpu").contiguous() for k, v in get_peft_model_state_dict(model).items()}
            save_file(sd, str(d / "adapter.safetensors"))
            torch.save(head.state_dict(), d / "head.pt")
            torch.save(opt.state_dict(), d / "optim.pt")
            (d / "state.json").write_text(json.dumps({"step": step, "why": why, "lora_r": a.lora_r, "lora_alpha": a.lora_alpha, "merge_from": merge_list,
                                                      "target_re": target_re, "letter_ids": lids, "model": a.model,
                                                      "manifest": a.manifest, "seed": a.seed, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
            lcfg.save_pretrained(str(d))
            log(tlog, {"event": "save", "step": step, "why": why, "dir": str(d)})
            import shutil  # the run log (gate decisions) and gate/public reads travel with every checkpoint for fresh-node resumes
            for extra in [out / "train_log.jsonl", *sorted(out.glob("gate_*.json")), *sorted(out.glob("public_*.json"))]:
                if extra.exists():
                    shutil.copy2(extra, d / extra.name)
            if a.s3:
                if mirror["p"] is not None and mirror["p"].poll() is None:
                    mirror["p"].wait()
                mirror["p"] = subprocess.Popen(["aws", "s3", "sync", "--quiet", str(d), f"{a.s3}/{d.name}/"])
        if barrier:
            dist.barrier()

    forced_day = None
    model.train()
    k_acc = a.rows_per_step // world
    step = step0
    if gate_rows and a.gate_baseline and step0 == 0:  # the init model on the SAME in-trainer path anchors the step gates
        res0 = kit_eval(model, head, proc, gate_rows, a.gate_root, rank, world, dev, out / "gate_00000.json", ocr_gate)
        base = torch.tensor([res0["public_vision_index"] if rank == 0 else 0.0], device=dev, dtype=torch.float64)
        dist.broadcast(base, src=0)
        a.gate_continue = float(base[0]) + a.gate_continue_rel
        a.gate_stop = float(base[0]) + a.gate_stop_rel
        if rank == 0:
            log(tlog, {"event": "gate_baseline", "step": 0, "index": float(base[0]), "continue_at": a.gate_continue, "stop_below": a.gate_stop,
                       "skill": {b: round(x["skill_clipped"] * 100, 2) for b, x in res0["benchmarks"].items()}})
    try:
        while step < a.steps:
            t0 = time.time()
            lr_scale = lr_at(step, a.steps, 1.0, a.warmup, a.floor)
            for g in opt.param_groups:
                g["lr"] = g["base"] * lr_scale
            loss_sum, correct, used, toks, skipped = 0.0, 0, 0, 0, 0
            for row in rows.step_rows(step):
                enc, n = encode(proc, row, a.img_root, a.max_tokens, ocr_train)
                if enc is None:
                    skipped += 1
                    continue
                enc = {k: v.to(dev) for k, v in enc.items()}
                k = len(row["options"])
                h = C.last_hidden(model, enc).float()
                z = head(h)[:k]
                t = target(row, k, a.label_smoothing).to(dev)
                loss = -(t * torch.log_softmax(z, -1)).sum()
                (loss / k_acc).backward()
                loss_sum += float(loss)
                correct += int(int(z.argmax()) == int(t.argmax()))
                used += 1
                toks += n
            stats = torch.tensor([loss_sum, correct, used, toks, skipped], device=dev, dtype=torch.float64)
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
            log(tlog, {"event": "step", "step": step, "loss": float(stats[0]) / u, "acc": float(stats[1]) / u, "rows": int(stats[2]),
                       "skipped": int(stats[4]), "tokens": int(stats[3]), "gnorm": float(gn), "lr": opt.param_groups[0]["lr"],
                       "s": round(time.time() - t0, 2)})
            now = dt.datetime.now(dt.timezone.utc)
            flags = torch.tensor([1 if STOP["why"] else 0, 1 if (now.hour == 7 and 30 <= now.minute < 40 and forced_day != now.date()) else 0],
                                 device=dev)
            dist.all_reduce(flags, op=dist.ReduceOp.MAX)
            if flags[1]:
                forced_day = now.date()
                save(step, "forced_0730")
            elif step % a.save_every == 0 or step == a.steps:
                save(step, "periodic" if step < a.steps else "final")
            if flags[0]:
                save(step, STOP["why"] or "stop_flag")
                break
            if gate_rows and step in gate_steps and not gate_done["decided"]:
                res = kit_eval(model, head, proc, gate_rows, a.gate_root, rank, world, dev, out / f"gate_{step:05d}.json", ocr_gate)
                verdict = torch.tensor([0], device=dev)
                if rank == 0:
                    g = res["public_vision_index"]
                    v = "CONTINUE" if g >= a.gate_continue else ("STOP" if g < a.gate_stop else "READ_AGAIN")
                    if v == "READ_AGAIN" and step == gate_steps[-1]:
                        v = "CONTINUE_TO_PUBLIC"
                    verdict[0] = {"CONTINUE": 1, "STOP": 2, "READ_AGAIN": 0, "CONTINUE_TO_PUBLIC": 1}[v]
                    log(tlog, {"event": "gate", "step": step, "index": g, "verdict": v,
                               "skill": {b: round(x["skill_clipped"] * 100, 2) for b, x in res["benchmarks"].items()}})
                dist.broadcast(verdict, src=0)
                if int(verdict[0]) == 1:
                    gate_done["decided"] = True
                elif int(verdict[0]) == 2:
                    save(step, "gate_stop")
                    STOP["why"] = "gate_stop"
                    break
            if pub_rows and step == a.public_step:
                res = kit_eval(model, head, proc, pub_rows, a.public_root, rank, world, dev, out / f"public_{step:05d}.json", ocr_public)
                if rank == 0:
                    log(tlog, {"event": "public_read", "step": step, "index": res["public_vision_index"],
                               "skill": {b: round(x["skill_clipped"] * 100, 2) for b, x in res["benchmarks"].items()}})
    except Exception:
        log(tlog, {"event": "exception", "rank": rank, "trace": traceback.format_exc()[-2000:]})
        try:
            save(step, "exception", barrier=False)
        finally:
            raise
    finally:
        if rank == 0 and mirror["p"] is not None:
            mirror["p"].wait()
        log(tlog, {"event": "end", "step": step, "why": STOP["why"] or "done"})
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
