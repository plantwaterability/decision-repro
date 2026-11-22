"""Which rows does the plain trainer put in which step? Replicates qv_train.py's Rows (order = list(range(n)) shuffled once by
random.Random(seed); logged step s uses rows [(s-1)*64, s*64) of that order) and, given train logs, checks that every logged step used 64
rows and skipped none; reports the ADE scene-count rows (source ade_cnt) among them. usage: ug1_rows_check.py <manifest> <seed> [train_log ...]
Last line = JSON; "ok" needs every logged step at rows == 64, skipped == 0."""
import json, random, sys
from collections import Counter

man, seed, logs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
src = [json.loads(l)["source"] for l in open(man, encoding="utf-8") if l.strip()]
n = len(src)
order = list(range(n))
random.Random(seed).shuffle(order)
R = 64
steps_total = n // R
ade_steps = [k // R + 1 for k, i in enumerate(order) if src[i] == "ade_cnt"]  # logged step numbers (1-based)
per1000 = Counter((s - 1) // 1000 for s in ade_steps)
out = {"manifest_rows": n, "steps_one_pass": steps_total, "ade_rows_total": len(ade_steps), "first_ade_step": min(ade_steps) if ade_steps else None,
       "steps_with_an_ade_row": len(set(ade_steps)), "ade_rows_in_steps_1_to_6": sum(1 for s in ade_steps if s <= 6),
       "ade_rows_per_1000_steps": [per1000.get(k, 0) for k in range((steps_total + 999) // 1000)]}
seen = {}
for p in logs:
    try:
        for l in open(p):
            if '"event": "step"' in l:
                r = json.loads(l)
                seen[r["step"]] = (r["rows"], r["skipped"])
    except FileNotFoundError:
        out.setdefault("missing_logs", []).append(p)
if logs:
    ss = sorted(seen)
    aset = Counter(ade_steps)
    out.update({"logged_steps": [ss[0], ss[-1]] if ss else None, "logged_step_count": len(ss), "rows_min": min((v[0] for v in seen.values()), default=None),
                "skipped_total": sum(v[1] for v in seen.values()), "steps_with_skips": sum(1 for v in seen.values() if v[1]),
                "ade_rows_in_logged_steps": sum(aset.get(s, 0) for s in ss),
                "ade_rows_in_logged_steps_without_skips": sum(aset.get(s, 0) for s in ss if seen[s][1] == 0),
                "sources_in_logged_steps": len({src[order[k]] for s in ss for k in range((s - 1) * R, min(n, s * R))})})
    out["ok"] = bool(ss) and all(v == (R, 0) for v in seen.values())
print(json.dumps(out))
