#!/usr/bin/env python3
"""t4b_eval.py: Pre-training engine parity gate, unseen private-like scoring (gde1v3s + dev), and final summary.

Modes:
  --mode parity    Verify (1) prefix-cache vs full-sequence parity on multi-question requests from s4400,
                   and (2) T4BEngine merged-weight parity vs live PEFT checkpoint on 16 real rows.
  --mode unseen    Run T4BEngine on structured single-question requests (gde1v3s / dev) -> {"id", "p"} JSONL.
  --mode summarize Aggregate screen (gate_index), unseen (tz_unseen_score), and latency into t4b1_result.json.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t4b_layout as LYT
from t4b_engine import T4BEngine


def run_parity(a):
    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoConfig, Qwen3_5ForCausalLM

    eng_cached = T4BEngine(a.model, ckpt=a.ckpt, max_len=32768, device="cuda:0", use_prefix_cache=True, batch_multiq=True)
    eng_uncached = T4BEngine.__new__(T4BEngine)
    eng_uncached.__dict__ = dict(eng_cached.__dict__)
    eng_uncached.use_prefix_cache = False
    eng_uncached.batch_multiq = False

    # 1. Pick multi-question rows from sample4400 (with 2..20 questions) + single-question choice/noul rows
    multi_rows, single_rows = [], []
    for l in gzip.open(a.s4400, "rt", encoding="utf-8"):
        r = json.loads(l)
        req = r.get("request", r)
        qs = req["questions"]
        if 2 <= len(qs) <= 12 and len(multi_rows) < 6:
            multi_rows.append(r)
        elif len(qs) == 1 and len(single_rows) < 10:
            single_rows.append(r)
        if len(multi_rows) >= 6 and len(single_rows) >= 10:
            break

    cache_pdiffs, cache_same, cache_tot = [], 0, 0
    for r in multi_rows:
        req = r.get("request", r)
        c1 = eng_cached._compile(req["state"], req["questions"])
        z_c = eng_cached._score_compiled(c1)
        z_u = eng_uncached._score_compiled(c1)
        for qk in z_c:
            pc = torch.softmax(torch.tensor(z_c[qk], dtype=torch.float32), dim=-1)
            pu = torch.softmax(torch.tensor(z_u[qk], dtype=torch.float32), dim=-1)
            cache_pdiffs.append(float((pc - pu).abs().max().item()))
            cache_same += int(int(pc.argmax()) == int(pu.argmax()))
            cache_tot += 1

    # 2. Verify merged T4BEngine weights & probabilities vs PEFT merge_and_unload()
    peft_pdiffs, peft_same, peft_tot, max_weight_diff = [], 0, 0, 0.0
    if a.ckpt:
        cfg = AutoConfig.from_pretrained(a.model).get_text_config()
        base = Qwen3_5ForCausalLM.from_pretrained(a.model, config=cfg, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0")
        if eng_cached.bidir:
            base.set_attn_implementation(LYT.register_bidir())
        lcfg = LoraConfig.from_pretrained(a.ckpt)
        pm = get_peft_model(base, lcfg)
        set_peft_model_state_dict(pm, load_file(str(Path(a.ckpt) / "adapter.safetensors"), device="cuda:0"))
        pm_m = pm.merge_and_unload().eval()
        with torch.no_grad():
            for (_, p1), (_, p2) in zip(eng_cached.model.named_parameters(), pm_m.named_parameters()):
                d = float((p1.detach().float() - p2.detach().float()).abs().max().item())
                if d > max_weight_diff:
                    max_weight_diff = d
        bb = pm_m.model
        for r in single_rows:
            req = r.get("request", r)
            c1 = eng_uncached._compile(req["state"], req["questions"])
            z_eng = eng_uncached._score_compiled(c1)
            for qk, q, keys, ids in c1:
                t_ids = torch.tensor([ids], dtype=torch.long, device="cuda:0")
                with torch.inference_mode():
                    h = bb(input_ids=t_ids, use_cache=False).last_hidden_state[0, -1].float()
                    zp = eng_cached.head(h)[: len(keys)].cpu()
                pe = torch.softmax(torch.tensor(z_eng[qk], dtype=torch.float32), dim=-1)
                pp = torch.softmax(zp.float(), dim=-1)
                peft_pdiffs.append(float((pe - pp).abs().max().item()))
                peft_same += int(int(pe.argmax()) == int(pp.argmax()))
                peft_tot += 1
        del pm_m, pm, base
        torch.cuda.empty_cache()

    cache_pdiffs_s = sorted(cache_pdiffs)
    res = {
        "provenance": eng_cached.provenance,
        "cache_questions": cache_tot,
        "cache_same_top": cache_same,
        "cache_med_prob_diff": round(cache_pdiffs_s[len(cache_pdiffs_s) // 2] if cache_pdiffs_s else 0.0, 6),
        "cache_max_prob_diff": round(max(cache_pdiffs) if cache_pdiffs else 0.0, 6),
        "merge_max_weight_diff": round(max_weight_diff, 6),
        "peft_questions": peft_tot,
        "peft_same_top": peft_same,
        "peft_max_prob_diff": round(max(peft_pdiffs) if peft_pdiffs else 0.0, 6),
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print("T4B_PARITY", json.dumps(res), flush=True)
    assert cache_tot > 0 and cache_same == cache_tot and res["cache_max_prob_diff"] < 0.03, f"Cache parity failed: {res}"
    if a.ckpt:
        assert max_weight_diff <= 1e-4 and peft_tot > 0 and peft_same == peft_tot and res["peft_max_prob_diff"] < 0.01, f"Merge parity failed: {res}"
    print("T4B_PARITY_OK", flush=True)


def run_unseen(a):
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    rows = [r for r in (json.loads(l) for l in gzip.open(a.req, "rt", encoding="utf-8")) if r["id"] not in done]
    print(f"UNSEEN todo={len(rows)} done={len(done)}", flush=True)
    eng = T4BEngine(a.model, ckpt=a.ckpt or None, ckpt_spec=a.ckpt_spec or None, max_len=65536, device="cuda:0")
    t0 = time.time()
    nerr = 0
    with open(a.out, "a", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            try:
                resp, _ = eng(r["request"]["state"], r["request"]["questions"])
                f.write(json.dumps({"id": r["id"], "p": resp["answers"]["q"]["probabilities"]}) + "\n")
            except Exception as ex:
                nerr += 1
                f.write(json.dumps({"id": r["id"], "err": f"{type(ex).__name__}: {str(ex)[:160]}"}) + "\n")
            if (i + 1) % 500 == 0:
                f.flush()
                print(f"UNSEEN {i+1}/{len(rows)} {(i+1)/(time.time()-t0):.1f} rows/s errors={nerr}", flush=True)
    print(f"UNSEEN_DONE {len(rows)} rows in {time.time()-t0:.1f}s errors={nerr}", flush=True)


def run_summarize(a):
    out = {"arm": a.arm, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if a.data_stats and os.path.exists(a.data_stats):
        ds = json.load(open(a.data_stats))
        out["data"] = {k: ds.get(k) for k in ("total_rows", "sha256", "by_kind", "multi_requests_used", "multi_questions_emitted", "n_sources")}
    if a.train_log and os.path.exists(a.train_log):
        steps, devs = [], []
        for l in open(a.train_log):
            r = json.loads(l)
            if r.get("event") == "step":
                steps.append(r)
            elif r.get("event") == "dev_eval":
                devs.append(r)
        out["train"] = {
            "last_step": steps[-1] if steps else None,
            "dev_trajectory": devs,
        }
    if a.screen_dir and os.path.exists(a.screen_dir):
        sd = Path(a.screen_dir)
        scr = {}
        for split in ("full", "h0", "h1"):
            p = sd / f"{split}.json"
            if p.exists():
                scr[split] = json.load(open(p))
        out["screen_4400"] = scr
        # Latency stats from results.jsonl.gz
        rpath = sd / "results.jsonl.gz"
        if rpath.exists():
            lat, ok, err, unsup = [], 0, 0, 0
            for l in gzip.open(rpath, "rt", encoding="utf-8"):
                r = json.loads(l)
                st = r.get("status")
                if st == "ok":
                    ok += 1
                    ms = r.get("model_request_wall_ms") or r.get("total_wall_ms") or r.get("latency_ms") or r.get("wall_ms") or (r.get("timing") or {}).get("total_ms")
                    if ms is not None:
                        lat.append(float(ms))
                elif st == "unsupported":
                    unsup += 1
                else:
                    err += 1
            lat.sort()
            def pct(p_):
                return round(lat[min(len(lat) - 1, int(len(lat) * p_))], 1) if lat else None
            out["screen_latency_h200"] = {
                "ok": ok,
                "unsupported": unsup,
                "error": err,
                "n_lat": len(lat),
                "median_ms": round(statistics.median(lat), 1) if lat else None,
                "mean_ms": round(sum(lat) / len(lat), 1) if lat else None,
                "p80_ms": pct(0.80),
                "p95_ms": pct(0.95),
            }
    if a.unseen_json and os.path.exists(a.unseen_json):
        uj = json.load(open(a.unseen_json))
        out["unseen_private_proxies"] = {
            k: {
                "common": v["common"],
                "table": v["table"],
                "paired_t4b": v["paired"].get(a.arm),
                "paired_lg": v["paired"].get("lg"),
                "paired_kudhab": v["paired"].get("kudhab"),
            }
            for k, v in uj.items()
        }
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("SUMMARY_WRITTEN", a.out, flush=True)
    # Compact console headline
    if "screen_4400" in out and "full" in out["screen_4400"]:
        arms = out["screen_4400"]["full"].get("arms", {})
        for k, v in arms.items():
            print(f"  SCREEN {k:14s}: index={v.get('index')} areas={v.get('areas')}", flush=True)
    if "unseen_private_proxies" in out:
        for sname, sv in out["unseen_private_proxies"].items():
            print(f"  UNSEEN {sname}: t4b={sv['table'].get(a.arm)} karpo={sv['table'].get('karpo')} kudhab={sv['table'].get('kudhab')} lg={sv['table'].get('lg')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("parity", "unseen", "summarize"))
    ap.add_argument("--model", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--ckpt-spec", default="")
    ap.add_argument("--s4400", default="")
    ap.add_argument("--req", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="t4b1")
    ap.add_argument("--data-stats", default="")
    ap.add_argument("--train-log", default="")
    ap.add_argument("--screen-dir", default="")
    ap.add_argument("--unseen-json", default="")
    a = ap.parse_args()
    if a.mode == "parity":
        run_parity(a)
    elif a.mode == "unseen":
        run_unseen(a)
    else:
        run_summarize(a)


if __name__ == "__main__":
    main()
