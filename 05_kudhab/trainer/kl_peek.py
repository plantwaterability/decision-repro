import gzip, json, sys, collections
sys.path.insert(0, "/data/local/mx/scripts"); import mx_tl_convert as C
K = "/data/local/kl/data/"
why = collections.Counter(); ex = {}; n = 0
for l in open(K + "mxa.jsonl"):
    r = json.loads(l); n += 1
    if n == 1: print("ROWKEYS", {k: (type(v).__name__, str(v)[:90]) for k, v in r.items() if k != "prompt"})
    req, w = C.parse(r)
    if req is None:
        kk = (w, r.get("kind"), len(r.get("option_order") or r["gold"])); why[kk] += 1
        if (w, r.get("kind")) not in ex: ex[(w, r.get("kind"))] = r
    if n >= 120000: break
print("FAIL", why.most_common(25))
for k, r in ex.items(): print("EX", k, r.get("option_order"), r.get("gold"), "\nPROMPT_TAIL", repr(r["prompt"][-900:]))
print("HEADER", open(K + "mxa.jsonl.header.json").read()[:1500])
r = json.loads(gzip.open("/data/local/mx/kroot/src/s4400_rows.jsonl.gz", "rt").readline()); print("KITROW", {k: str(v)[:160] for k, v in r.items()})
r = json.loads(gzip.open("/data/local/kl/ref/munku/prefix_results.jsonl.gz", "rt").readline()); print("RESROW", json.dumps(r)[:1500])
