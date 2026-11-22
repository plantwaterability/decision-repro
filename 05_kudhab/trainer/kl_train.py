#!/usr/bin/env python3
"""kl_train.py (KL agent, 2026-11-15): a trainer for karpo's prompt layout (karpo-blocks-v1) on one 8-GPU node.

The forward pass, layout, attention rule and readout are karpo's own modules, imported unmodified from machinepulse-ai/karpo-decision
@ a6fa38bf (Apache-2.0; its LICENSE is kept beside this file as LICENSE.karpo-decision; nothing of it is copied into this file):
families.gemma.Packed (packed block layouts, global layers bidirectional, sliding layers causal, last layer at the markers), encoding.Encoder
/ blocks.BlockLayout (the layout as token ids), model.packed.pack, checkpoint.lm_head_rows (readout = the init's LM-head rows of the 255
codes, fp32, frozen), the soft cap 30 tanh(l / 30). Trained parameters = families.gemma.TRAINED (the decoder's linear projections).
The sharding (FSDP2 per decoder layer + root, bf16 compute over an fp32 master), the rank-local weight load, resume shards, S3 mirror,
spot / SIGTERM / scheduled saves and the HF export are our fullft_train.py / fullft_ckpt.py (pinned 4048f97c), imported unmodified.

Loss per decision: w [(1 - a) CE(gold, p) + a KL(q || p)], a = --alpha where a teacher target q exists (else 0), p = softmax of the capped
code logits over the decision's own options at the readout position; summed over the step's decisions / the step's weight.

Modes: train (default) | predict (--predict-in requests.jsonl -> --predict-out, the same sharded bf16 forward, no grad; the parity gate)
| smoke (--smoke-steps N real steps on the LONGEST rows first: the memory test).
  torchrun --nproc-per-node 8 kl_train.py --init-from DIR --manifest DATA_PREFIX --val DEV_PREFIX --out RUN ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import types

import numpy as np
import torch

VERSION = "kl_train v1 (2026-11-15)"      # v1.1 22:50Z: --epochs > 1 = one shuffle per pass under one schedule (K13); nothing else changed, v1 runs resume as before


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("train", "predict", "smoke"), default="train")
    ap.add_argument("--init-from", required=True, help="HF-layout weights dir (model.safetensors.index.json, config.json, tokenizer)")
    ap.add_argument("--manifest", default="", help="kl_data.py output prefix (.ids.npy / .meta.npz / .info.json)")
    ap.add_argument("--val", default="", help="dev data prefix (same format)")
    ap.add_argument("--val-max", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pinned-code", required=True, help="dir of the pinned fullft_train.py")
    ap.add_argument("--lr", type=float, default=2.5e-6)
    ap.add_argument("--lr-floor", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--decisions-per-step", type=int, default=64)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--sched-steps", type=int, default=0, help="length of the cosine schedule (0 = the run's own step count)")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-row-tokens", type=int, default=8192)
    ap.add_argument("--pack-tokens", type=int, default=8192)
    ap.add_argument("--eval-tokens", type=int, default=16384)
    ap.add_argument("--checkpoint-every", type=int, default=1, help="karpo PackedLayers: recompute every n-th layer in backward (1 = all)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-at-start", action="store_true")
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--keep-resume", type=int, default=1)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--resume-s3", default="")
    ap.add_argument("--keep-resume-s3", type=int, default=2)
    ap.add_argument("--resume-kind", default="full")
    ap.add_argument("--stop-tiers", default="full")
    ap.add_argument("--spot-watch", choices=("on", "off"), default="on")
    ap.add_argument("--force-save-utc", default="")
    ap.add_argument("--async-save", choices=("on", "off"), default="on")
    ap.add_argument("--snapshot-pinned", choices=("on", "off"), default="off")
    ap.add_argument("--save-on-exception", choices=("on", "off"), default="on")
    ap.add_argument("--allow-code-change-on-resume", action="store_true")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--smoke-steps", type=int, default=2)
    ap.add_argument("--predict-in", default="")
    ap.add_argument("--predict-out", default="")
    ap.add_argument("--predict-max-length", type=int, default=32768)
    ap.add_argument("--licence-note", default="")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    a.tiny = False
    return a


a = parse()
sys.path.insert(0, a.pinned_code)
import fullft_ckpt as fc  # noqa: E402
import fullft_train as ft  # noqa: E402
from karpo_decision import checkpoint as kck  # noqa: E402
from karpo_decision.codes import answer_codes  # noqa: E402
from karpo_decision.encoding import Encoded, Encoder, LengthError  # noqa: E402
from karpo_decision.families import gemma  # noqa: E402
from karpo_decision.model.packed import pack as kpack  # noqa: E402
from karpo_decision.protocol import RequestError  # noqa: E402
from karpo_decision.templates import template as ktemplate  # noqa: E402

N_CODES = 255


class KL(torch.nn.Module):
    """root module: the text stack under the name fullft_train's weight I/O expects; forward = karpo's packed forward (the FSDP root
    hook gathers the embedding / final norm for it)."""

    def __init__(self, text):
        super().__init__()
        self.text = text

    def forward(self, p):
        return self.__dict__["_packed"](p)


def text_prefix(mdir):
    idx = ft.weight_map(mdir)
    for p in ("model.language_model.", "language_model."):
        if any(k.startswith(p + "layers.0.") for k in idx):
            return p
    raise SystemExit(f"{mdir}: no text stack under model.language_model. / language_model.")


def build(a, dist):
    from transformers import AutoConfig
    mdir = a.init_from
    cfg = AutoConfig.from_pretrained(mdir).get_text_config()
    cfg._attn_implementation = "sdpa"
    ft.TEXT_PREFIX = text_prefix(mdir)
    with torch.device("meta"):
        text = ft.text_classes(cfg)[0](cfg)
        model = KL(text)
    for n, p in text.named_parameters():             # karpo's recipe: only the decoder's linear projections train
        p.requires_grad_(n.startswith("layers.") and n.rsplit(".", 2)[-2] in gemma.TRAINED)
    if dist.on:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
        mesh = init_device_mesh("cuda", (dist.world,))
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for layer in text.layers:
            fully_shard(layer, mesh=mesh, mp_policy=mp)
        fully_shard(model, mesh=mesh, mp_policy=mp)
        dist.loss_scale = float(dist.world)          # FSDP averages gradients over ranks; the loss is divided by the GLOBAL step weight
    else:
        dist.loss_scale = 1.0
    model.to_empty(device=dist.dev)
    model.float()
    n, scal = ft.load_pretrained(model, mdir, cfg)
    ft.rebuild_buffers(text, cfg, dist.dev, scal)
    if not dist.on:                                  # one GPU (predict only): the engine's own bf16 weights
        model.bfloat16()
    packed = gemma.Packed(types.SimpleNamespace(language_model=text), "bidirectional", checkpoint_every=a.checkpoint_every,
                          recompute="layer")
    model.__dict__["_packed"] = packed
    if dist.on:
        from torch.distributed.fsdp import register_fsdp_forward_method
        for layer in text.layers:                    # a sharded layer gathers its weights for karpo's per-layer method
            register_fsdp_forward_method(layer, "packed_forward")
    proc = gemma.load_processor(mdir, 0)
    codes, token_ids = answer_codes(proc.tokenizer)
    enc = Encoder(proc, codes, ktemplate("karpo-blocks-v1"), gemma.MARKER)
    W = kck.lm_head_rows(mdir, token_ids).float().to(dist.dev)       # readout rows: frozen, fp32 (karpo's Readout.from_rows)
    cap = float(cfg.final_logit_softcapping)
    if dist.rank == 0:
        tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        al = sum(p.numel() for p in model.parameters())
        print(f"BUILD {VERSION}: loaded {n} text tensors from {mdir} (prefix {ft.TEXT_PREFIX}); trained {tr:,} of {al:,} parameters; "
              f"codes {len(codes)} cap {cap} layers {len(text.layers)} window {cfg.sliding_window}", flush=True)
    return model, text, cfg, enc, W, cap


def readout(hidden, W, cap):
    logits = hidden.float() @ W.t()
    return cap * torch.tanh(logits / cap)


class Data:
    """kl_data.py's arrays: token ids (karpo-blocks-v1, one question per row), option counts, gold / teacher distributions, weights."""

    def __init__(self, prefix):
        self.info = json.load(open(prefix + ".info.json"))
        self.ids = np.load(prefix + ".ids.npy", mmap_mode="r")
        m = np.load(prefix + ".meta.npz")
        self.off, self.K, self.w, self.ooff = m["off"], m["K"].astype(np.int64), m["w"].astype(np.float64), m["ooff"]
        self.gold, self.q, self.hasq = m["gold"], m["q"], m["hasq"]
        self.len = np.diff(self.off)
        self.n = len(self.K)

    def encoded(self, i):
        ids = torch.from_numpy(np.asarray(self.ids[self.off[i]:self.off[i + 1]], dtype=np.int64))
        n = int(ids.numel())
        return Encoded(keys=("q",), counts=(int(self.K[i]),), input_ids=ids, markers=torch.tensor([n - 1]), positions=torch.arange(n),
                       segment_ids=torch.zeros(n, dtype=torch.int16))

    def targets(self, items, dev):
        T = torch.zeros(len(items), N_CODES, dtype=torch.float32)
        Q = torch.zeros(len(items), N_CODES, dtype=torch.float32)
        for j, i in enumerate(items):
            k = int(self.K[i])
            T[j, :k] = torch.from_numpy(np.asarray(self.gold[self.ooff[i]:self.ooff[i] + k]))
            if self.hasq[i]:
                Q[j, :k] = torch.from_numpy(np.asarray(self.q[self.ooff[i]:self.ooff[i] + k]))
        K = torch.as_tensor(self.K[items])
        return T.to(dev), Q.to(dev), K.to(dev), torch.as_tensor(self.hasq[items]).to(dev)


def losses(logits, T, Q, K, hasq, alpha):
    """per decision: CE to gold, KL(q || p) (0 without a teacher), the mixed loss, correct (argmax p == argmax gold)"""
    mask = torch.arange(N_CODES, device=logits.device)[None] < K[:, None]
    lp = torch.log_softmax(logits.double().masked_fill(~mask, float("-inf")), -1).masked_fill(~mask, 0.0)
    T, Q = T.double(), Q.double()
    ce = -(T * lp).sum(-1)
    kl = (Q * (torch.log(Q.clamp_min(1e-30)) - lp)).sum(-1)
    aq = alpha * hasq.double()
    ok = (lp.masked_fill(~mask, float("-inf")).argmax(-1) == T.argmax(-1)).double()
    return ce, kl, (1 - aq) * ce + aq * kl, ok


def run_pack(a, model, W, cap, data, items, wscale, dist, W_step, train):
    """one micro-batch on this rank -> (loss tensor or None, [sum w*loss, sum w*ce, sum w*kl_on_teacher_rows, sum w, n, correct, tokens])"""
    p = kpack([data.encoded(i) for i in items], None, dist.dev)
    T, Q, K, hasq = data.targets(items, dist.dev)
    w = torch.as_tensor(data.w[items], dtype=torch.float64, device=dist.dev) * wscale
    logits = readout(model(p), W, cap)
    ce, kl, mix, ok = losses(logits, T, Q, K, hasq, a.alpha if train else 0.0)
    loss = (w * mix).sum() / W_step * dist.loss_scale if train else None
    st = torch.stack([(w * mix).sum(), (w * ce).sum(), (w * kl).sum(), w.sum(), torch.tensor(float(len(items)) * wscale, device=w.device,
                      dtype=torch.float64), (ok * (w > 0)).sum(), torch.tensor(float(p.tokens), device=w.device, dtype=torch.float64)]).detach()
    return loss, st


def global_packs(lengths, idx, budget, world):
    """idx sorted longest first into packs of <= budget tokens; pack j runs on rank j % world; every rank the same number of packs
    (None = repeat a dummy), as FSDP's collectives require. -> per rank list of packs"""
    order = sorted(idx, key=lambda i: (-int(lengths[i]), i))
    packs, cur, used = [], [], 0
    for i in order:
        n = int(lengths[i])
        if cur and used + n > budget:
            packs.append(cur)
            cur, used = [], 0
        cur.append(i)
        used += n
    if cur:
        packs.append(cur)
    per = [packs[r::world] for r in range(world)]
    m = max(len(x) for x in per)
    return [x + [None] * (m - len(x)) for x in per]


def evaluate(a, model, W, cap, data, idx, dist):
    """collective: dev NLL (CE to gold, weight 1 per row) and accuracy over idx"""
    mine = global_packs(data.len, idx, a.eval_tokens, dist.world)[dist.rank]
    tot = torch.zeros(7, dtype=torch.float64, device=dist.dev)
    dummy = [min(idx, key=lambda i: int(data.len[i]))]
    with torch.no_grad():
        for pk in mine:
            _, st = run_pack(a, model, W, cap, data, pk or dummy, 1.0 if pk else 0.0, dist, 1.0, False)
            tot += st
    dist.sum_([tot])
    ft.reshard_all(model)
    w = float(tot[3])
    return {"n": int(round(float(tot[4]))), "nll": float(tot[1]) / max(w, 1e-9), "acc": float(tot[5]) / max(float(tot[4]), 1e-9)}


def predict(a, model, W, cap, enc, dist):
    """the parity path: requests (jsonl: id, state, questions) through the SAME sharded forward, several sequences per pack"""
    reqs = [json.loads(l) for l in open(a.predict_in)]
    parts, bad = [], {}
    for ri, r in enumerate(reqs):
        try:
            for pi, e in enumerate(enc.plan({"state": r["state"], "questions": r["questions"]}, a.predict_max_length)):
                parts.append((ri, pi, e))
        except (RequestError, LengthError) as e:
            bad[ri] = f"{type(e).__name__}: {str(e)[:120]}"
    lens = [e.length for _, _, e in parts]
    mine = global_packs(lens, list(range(len(parts))), a.eval_tokens, dist.world)[dist.rank]
    dummy = [min(range(len(parts)), key=lambda i: lens[i])]
    out = {}
    t0 = time.time()
    with torch.no_grad():
        for n, pk in enumerate(mine):
            seqs = [parts[i][2] for i in (pk or dummy)]
            logits = readout(model(kpack(seqs, None, dist.dev)), W, cap).cpu()
            if pk:
                cur = 0
                for i in pk:
                    ri, pi, e = parts[i]
                    for key, count in zip(e.keys, e.counts):
                        out[(ri, key)] = logits[cur, :count].double().softmax(-1).tolist()
                        cur += 1
            if dist.rank == 0 and n % 20 == 0:
                print(f"PREDICT pack {n}/{len(mine)} {time.time() - t0:.0f} s", flush=True)
    ft.reshard_all(model)
    allout = dist.gather_obj(out)
    if dist.rank == 0:
        merged = {}
        for o in allout:
            merged.update(o)
        ntok = {}
        for ri, pi, e in parts:
            ntok[ri] = ntok.get(ri, 0) + e.length
        with open(a.predict_out, "w") as f:
            for ri, r in enumerate(reqs):
                if ri in bad:
                    f.write(json.dumps({"id": r["id"], "unsupported": bad[ri]}) + "\n")
                else:
                    f.write(json.dumps({"id": r["id"], "input_tokens": ntok[ri], "probs": {k: merged[(ri, k)] for k in r["questions"]}}) + "\n")
        print(f"PREDICT_DONE {len(reqs)} requests ({len(bad)} unsupported), {len(parts)} sequences, {sum(lens)} tokens, "
              f"{time.time() - t0:.0f} s", flush=True)


def main():
    dist = ft.setup_dist(a)
    torch.manual_seed(a.seed + dist.rank)
    os.makedirs(a.out, exist_ok=True)
    t_build = time.time()
    model, text, cfg, enc, W, cap = build(a, dist)
    cuda = dist.dev.type == "cuda"
    if dist.rank == 0:
        print(f"BUILT in {time.time() - t_build:.0f} s; args {json.dumps(vars(a), default=str)}", flush=True)
    if a.mode == "predict":
        predict(a, model, W, cap, enc, dist)
        return 0
    data = Data(a.manifest)
    a.data_rows_sha256 = data.info["data_sha256"]
    dev = Data(a.val) if a.val else None
    dev_idx = list(range(min(dev.n, a.val_max))) if dev else []
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=a.weight_decay, fused=cuda)
    ok_rows = np.nonzero(data.len <= a.max_row_tokens)[0]
    rng = np.random.default_rng(a.seed)
    if a.mode == "smoke":                             # the longest rows first: the memory test
        order = ok_rows[np.argsort(-data.len[ok_rows], kind="stable")][: a.smoke_steps * a.decisions_per_step]
    else:
        # one permutation per pass (--epochs 2: the second pass is a fresh shuffle of the same rows); with --epochs <= 1 this is the single permutation of v1, value for value
        order = np.concatenate([ok_rows[rng.permutation(len(ok_rows))] for _ in range(max(1, int(math.ceil(a.epochs))))])
    dps = a.decisions_per_step
    steps_all = int(math.ceil(len(order) * a.epochs / dps)) if a.mode == "smoke" else int(math.ceil(len(ok_rows) * a.epochs / dps))      # = v1's value at --epochs 1 (len(order) == len(ok_rows))
    sched = a.sched_steps or steps_all
    steps = min(steps_all, a.max_steps) if a.max_steps else steps_all
    order_fp = hashlib.sha256(np.asarray(order, dtype=np.int64).tobytes()).hexdigest()[:16]
    W_total = float(data.w[ok_rows].sum())
    if dist.rank == 0:
        print(f"DATA rows {data.n} used {len(ok_rows)} (dropped {data.n - len(ok_rows)} over {a.max_row_tokens} tokens) weight {W_total:.2f} "
              f"teacher rows {int(data.hasq[ok_rows].sum())} tokens {int(data.len[ok_rows].sum())} longest {int(data.len[ok_rows].max())} "
              f"steps {steps} of schedule {sched} order_fp {order_fp} data {a.data_rows_sha256[:16]}", flush=True)
    start = 0 if a.mode == "smoke" else ft.load_resume(a, model, opt, None, dist, None, order_fp)
    intr = fc.Interrupts()
    old_sig = fc.install_signal_handlers(intr)
    s3 = fc.S3(a.resume_s3) if (a.resume_s3 and a.mode == "train") else None
    writer = fc.ResumeWriter(a.out, dist.rank, dist.world, s3, a.keep_resume, a.keep_resume_s3, single_node=True, async_ok=a.async_save == "on")
    writer.marker_extra = {"ident": ft.run_ident(a), "data_rows_sha256": a.data_rows_sha256, "code": ft.code_identity(a.pinned_code),
                           "order_fp": order_fp, "kl_train": VERSION}
    watcher = None
    if dist.rank == 0 and a.mode == "train" and (a.spot_watch == "on" or a.force_save_utc):
        def on_notice(kind, body):
            if kind == "instance-action":
                intr.stop("spot instance-action", 75, body)
            elif kind == "rebalance":
                intr.rebalance = body or kind
            else:
                intr.scheduled = body
            print(f"NOTICE {kind} {body}", flush=True)
        watcher = fc.SpotWatcher(on_notice, daily_utc=[t.strip() for t in a.force_save_utc.split(",") if t.strip()], imds=a.spot_watch == "on")
        watcher.start()

    def blob(bstep, reuse=False):
        return ft.resume_blob(a, model, opt, None, bstep, dist, None, "full", order_fp, reuse=reuse)

    def stop_save(bstep, code, why):
        t0 = time.time()
        if dist.rank == 0:
            print(f"STOP_REQUEST {why}: saving resume_step{bstep}, then exit {code}", flush=True)
        if bstep > start:
            writer.wait()
            writer.save(bstep, blob(bstep), "full", sync=True, why=f"stop: {why}")
            if dist.on:
                torch.distributed.barrier()
            if dist.rank == 0:
                writer.ensure_published(bstep, "full")
                print(f"STOP_SAVED resume_step{bstep} +{time.time() - t0:.1f} s", flush=True)
        if watcher is not None:
            watcher.stop()
        if dist.rank == 0:
            print(f"STOP_EXIT {code} ({why})", flush=True)
        if dist.on:
            torch.distributed.destroy_process_group()
        raise SystemExit(code)

    def async_save(bstep, why):
        t0 = time.time()
        writer.wait()
        b = blob(bstep, reuse=True)
        writer.save(bstep, b, "full", sync=not writer.async_ok, why=why)
        del b
        if not writer.async_ok:
            if dist.on:
                torch.distributed.barrier()
            if dist.rank == 0:
                writer.ensure_published(bstep, "full")
        if dist.rank == 0:
            print(f"SAVE_DONE_LOCAL resume_step{bstep} ({why}) {time.time() - t0:.1f} s", flush=True)

    def do_eval(step):
        if not dev_idx:
            return
        t0 = time.time()
        m = evaluate(a, model, W, cap, dev, dev_idx, dist)
        m.update({"step": step, "seconds": round(time.time() - t0, 1)})
        if dist.rank == 0:
            print("EVAL " + json.dumps(m), flush=True)
            with open(os.path.join(a.out, "evals.jsonl"), "a") as f:
                f.write(json.dumps(m) + "\n")

    if a.eval_at_start and start == 0 and a.mode == "train":
        do_eval(0)
    shortest = int(ok_rows[np.argmin(data.len[ok_rows])])
    checked = False
    t_run = time.time()
    for step in range(start, steps):
        phase = "pre"
        try:
            ts = time.time()
            lr = ft.lr_at(step, a.lr, a.warmup, sched, a.lr_floor)
            for g in opt.param_groups:
                g["lr"] = lr
            items = [int(i) for i in order[step * dps:(step + 1) * dps]]
            plan = ft.plan_step(items, data.len, dist.world, a.pack_tokens)
            W_step = float(data.w[items].sum())
            if cuda:
                torch.cuda.reset_peak_memory_stats(dist.dev)
            opt.zero_grad(set_to_none=True)
            tot = torch.zeros(7, dtype=torch.float64, device=dist.dev)
            phase = "backward"
            for mb in plan[dist.rank]:
                loss, st = run_pack(a, model, W, cap, data, mb or [shortest], 1.0 if mb else 0.0, dist, W_step, True)
                loss.backward()
                tot += st
                del loss
            if not checked:                           # gradient reaches only the linear projections
                bad = [n for n, p in model.named_parameters() if (p.grad is not None) != p.requires_grad]
                frozen_with_grad = [n for n, p in model.named_parameters() if not p.requires_grad and p.grad is not None]
                if bad or frozen_with_grad:
                    raise SystemExit(f"GRAD_CHECK failed: {bad[:3]} {frozen_with_grad[:3]}")
                if dist.rank == 0:
                    names = sorted({n.rsplit(".", 2)[-2] for n, p in model.named_parameters() if p.grad is not None})
                    print(f"GRAD_CHECK ok: gradients on {sum(p.grad is not None for p in model.parameters())} tensors ({', '.join(names)}); "
                          f"none on {sum(not p.requires_grad for p in model.parameters())} frozen tensors", flush=True)
                checked = True
            gn = ft.clip_grads(params, a.clip, dist)
            if not math.isfinite(gn):
                raise SystemExit(f"non-finite gradient norm at step {step}")
            phase = "update"
            opt.step()
            phase = "post"
            dist.sum_([tot])
            peak = torch.tensor([torch.cuda.max_memory_allocated(dist.dev) / 2 ** 30 if cuda else 0.0], dtype=torch.float64, device=dist.dev)
            if dist.on:
                torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
            row = {"step": step, "loss": float(tot[0]) / W_step, "ce": float(tot[1]) / W_step, "kl": float(tot[2]) / W_step,
                   "acc": float(tot[5]) / max(len(items), 1), "gnorm": round(gn, 4), "lr": lr, "tokens": int(tot[6]),
                   "micro": len(plan[0]), "longest": int(max(data.len[i] for i in items)), "peak_gib": round(float(peak), 2),
                   "sec": round(time.time() - ts, 2)}
            if not math.isfinite(row["loss"]):
                raise SystemExit(f"non-finite loss at step {step}")
            if dist.rank == 0:
                print("STEP " + json.dumps(row), flush=True)
            if a.mode == "train":
                if (step + 1) % a.eval_every == 0 or step + 1 == steps:
                    do_eval(step + 1)
                act, code, why = fc.sync_flags(intr, dist)
                if act == fc.ACTION_STOP:
                    stop_save(step + 1, code, why)
                periodic = a.save_every and ((step + 1) % a.save_every == 0 or step + 1 == steps)
                if periodic or act == fc.ACTION_SAVE:
                    async_save(step + 1, "periodic" if periodic else why)
        except Exception as e:
            b_ = step if phase in ("pre", "backward") else step + 1
            if a.save_on_exception == "on" and a.mode == "train" and phase != "update" and b_ > start and not os.path.exists(
                    os.path.join(a.out, f"resume_step{b_}", "COMPLETE")):
                try:
                    print(f"EXCEPTION_SAVE rank {dist.rank} resume_step{b_} ({type(e).__name__}: {str(e)[:200]})", flush=True)
                    opt.zero_grad(set_to_none=True)
                    writer.save(b_, blob(b_), "full", sync=True, why=f"exception: {type(e).__name__}")
                except Exception as e2:              # noqa: BLE001
                    print(f"EXCEPTION_SAVE_FAILED rank {dist.rank}: {e2!r}", flush=True)
            raise
    writer.wait()
    if dist.on:
        torch.distributed.barrier()
    if watcher is not None:
        watcher.stop()
    fc.restore_signal_handlers(old_sig)
    if dist.rank == 0:
        print(f"TRAIN_END step {steps} in {time.time() - t_run:.0f} s", flush=True)
    if a.mode == "train" and not a.no_export:
        ft.reshard_all(model)
        state = ft.gather_full_state(model, dist.dev)
        if dist.rank == 0:
            exp = os.path.join(a.out, "export")
            n = ft.export_hf(exp + ".tmp", a.init_from, state)
            json.dump({"trainer": VERSION, "step": steps, "init_from": a.init_from, "layout": "karpo-blocks-v1", "attention": "bidirectional",
                       "data_rows_sha256": a.data_rows_sha256, "alpha": a.alpha, "lr": a.lr, "seed": a.seed, "licence_note": a.licence_note},
                      open(os.path.join(exp + ".tmp", "kl_export.json"), "w"), indent=1)
            os.rename(exp + ".tmp", exp)
            print(f"EXPORTED {exp} ({n} shards)", flush=True)
    if dist.on:
        torch.distributed.barrier()
    return 0


if __name__ == "__main__":
    sys.exit(main())
