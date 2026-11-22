#!/usr/bin/env python3
"""dh_tl_label.py (DH fork, copy of the MX fork's mx_tl_label.py with ONE change: every question of a request is written, so grouped
multi-question requests (round H, fs5h_multi) get {"id", "p": {question_key: {option_key: prob}}}; single-question requests keep {"id", "p": {option_key: prob}}).
Original docstring: mx_tl_label.py (MX fork, distillation line): teacher probabilities from karpo's OWN engine (karpo_decision.decider.Decider, the packed batch API:
one forward pass per pack of requests, same template / attention / readout as its kit engine) for structured requests made by mx_tl_convert.py.
One process per GPU; rows i % NSHARDS == SHARD, longest first; output parts are appended and flushed per batch, a restart skips ids already written.
  mx_tl_label.py CKPT REQUESTS.jsonl.gz OUT.jsonl SHARD NSHARDS [LIMIT]       OUT rows: {"id", "p": {option_key: prob}} or {"id", "err": reason}"""
import gzip, json, os, sys, time
import torch
from karpo_decision.decider import Decider
ck, src, out, shard, nsh = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]); limit = int(sys.argv[6]) if len(sys.argv) > 6 else 0
done = set()
if os.path.exists(out):
    for l in open(out):
        try: done.add(json.loads(l)["id"])
        except Exception: pass
rows = []
for i, l in enumerate(gzip.open(src, "rt")):
    if limit and i >= limit: break
    if i % nsh != shard: continue
    r = json.loads(l)
    if r["id"] not in done: rows.append(r)
rows.sort(key=lambda r: -(r.get("n_tokens") or 0))
print(f"LABEL shard {shard}/{nsh}: todo {len(rows)} done {len(done)}", flush=True)
dec = Decider.from_pretrained(ck, device="cuda"); BUDGET = int(os.environ.get("MX_TL_BUDGET", "24000")); MAXN = int(os.environ.get("MX_TL_MAXN", "96"))
def emit(f, r, resp):
    qs = r["request"]["questions"]
    if list(qs) == ["q"]:
        f.write(json.dumps({"id": r["id"], "p": resp["answers"]["q"]["probabilities"]}) + "\n")
    else:
        f.write(json.dumps({"id": r["id"], "p": {k: resp["answers"][k]["probabilities"] for k in qs}}) + "\n")
t0 = time.time(); n = 0; nerr = 0
with open(out, "a") as f, torch.inference_mode():
    i = 0
    while i < len(rows):
        batch = []; tok = 0
        while i < len(rows) and len(batch) < MAXN and (not batch or tok + (rows[i].get("n_tokens") or 512) <= BUDGET):
            batch.append(rows[i]); tok += rows[i].get("n_tokens") or 512; i += 1
        try:
            resps = dec.decide([b["request"] for b in batch])
            for b, r in zip(batch, resps): emit(f, b, r)
        except Exception as e:      # one bad request (length, contract) must not take the batch with it: one by one
            torch.cuda.empty_cache()
            for b in batch:
                try: emit(f, b, dec.decide(b["request"]))
                except Exception as e2: f.write(json.dumps({"id": b["id"], "err": f"{type(e2).__name__}: {str(e2)[:120]}"}) + "\n"); nerr += 1; torch.cuda.empty_cache()
        f.flush(); n += len(batch)
        if n % 2000 < len(batch): print(f"LABEL shard {shard}: {n}/{len(rows)} rows, {n / (time.time() - t0):.1f} rows/s, errors {nerr}", flush=True)
print(f"LABEL_DONE shard {shard}: {n} rows in {time.time() - t0:.0f} s, errors {nerr}", flush=True)
