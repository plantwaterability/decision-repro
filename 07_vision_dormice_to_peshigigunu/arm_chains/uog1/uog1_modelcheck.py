"""uog1 restage check of the base model: every shard named by model.safetensors.index.json must exist, be complete (safetensors header
vs file size) and hold exactly the tensors the index assigns to it; file sizes and sha256 are compared with the Hub's metadata at the
pinned revision (tokenless). The index file existing is NOT enough. usage: uog1_modelcheck.py <model dir> <revision> [expected shards=18]
Last line = JSON summary; exit 0 only if ok."""
import hashlib, json, os, struct, sys, time
from concurrent.futures import ThreadPoolExecutor

d, rev = sys.argv[1], sys.argv[2]
want_n = int(sys.argv[3]) if len(sys.argv) > 3 else 18
out = {"ok": False, "dir": d}


def done():
    print(json.dumps(out))
    sys.exit(0 if out["ok"] else 1)


ip = os.path.join(d, "model.safetensors.index.json")
if not os.path.isfile(ip):
    out["why"] = "no index file"
    done()
wm = json.load(open(ip))["weight_map"]
by = {}
for name, shard in wm.items():
    by.setdefault(shard, set()).add(name)
out["shards_in_index"] = len(by)
out["tensors_in_index"] = len(wm)
problems = []
if len(by) != want_n:
    problems.append(f"index names {len(by)} shards, expected {want_n}")


def struct_check(shard):
    p = os.path.join(d, shard)
    if not os.path.isfile(p):
        return f"{shard}: missing"
    size = os.path.getsize(p)
    with open(p, "rb") as f:
        head = f.read(8)
        if len(head) < 8:
            return f"{shard}: shorter than a header"
        n = struct.unpack("<Q", head)[0]
        if n <= 0 or n > min(size, 200_000_000):
            return f"{shard}: bad header length {n}"
        try:
            h = json.loads(f.read(n))
        except Exception as e:  # noqa: BLE001
            return f"{shard}: header not JSON ({str(e)[:60]})"
    names = {k for k in h if k != "__metadata__"}
    end = max((v["data_offsets"][1] for k, v in h.items() if k != "__metadata__"), default=0)
    if size != 8 + n + end:
        return f"{shard}: size {size} != header-implied {8 + n + end} (truncated or partial download)"
    if names != by[shard]:
        return f"{shard}: tensors differ from the index ({len(names - by[shard])} extra, {len(by[shard] - names)} missing)"
    return None


for shard in sorted(by):
    r = struct_check(shard)
    if r:
        problems.append(r)
out["structural_ok"] = not problems
# Hub metadata (tokenless): sizes of every file + sha256 of the LFS files
meta = None
for k in range(3):
    try:
        import urllib.request

        req = urllib.request.Request(f"https://huggingface.co/api/models/Qwen/Qwen3.8-27B/revision/{rev}?blobs=true", headers={"User-Agent": "model-check"})
        meta = json.load(urllib.request.urlopen(req, timeout=60))
        break
    except Exception as e:  # noqa: BLE001
        out["hub_meta_error"] = str(e)[:120]
        time.sleep(5 + 10 * k)
if meta and not problems:
    sib = {s["rfilename"]: s for s in meta.get("siblings", [])}
    out["hub_sha"] = meta.get("sha")
    if meta.get("sha") != rev:
        problems.append(f"hub revision {meta.get('sha')} != pinned {rev}")
    miss = [f for f in sib if not os.path.isfile(os.path.join(d, f)) and not f.startswith(".git")]
    if miss:
        problems.append(f"files of the revision missing locally: {miss[:5]}")

    def sha(f):
        h = hashlib.sha256()
        with open(os.path.join(d, f), "rb") as fh:
            while True:
                b = fh.read(1 << 24)
                if not b:
                    break
                h.update(b)
        return f, h.hexdigest()

    lfs = [f for f, s in sib.items() if s.get("lfs") and os.path.isfile(os.path.join(d, f))]
    for f in lfs:
        if os.path.getsize(os.path.join(d, f)) != sib[f]["lfs"]["size"]:
            problems.append(f"{f}: size differs from the Hub")
    if not problems:
        with ThreadPoolExecutor(min(24, max(1, len(lfs)))) as ex:
            for f, hx in ex.map(sha, lfs):
                if hx != sib[f]["lfs"]["sha256"]:
                    problems.append(f"{f}: sha256 differs from the Hub")
        out["sha256_checked_files"] = len(lfs)
        out["sha256_checked_shards"] = sum(1 for f in lfs if f in by)
        if out["sha256_checked_shards"] != len(by):
            problems.append(f"only {out['sha256_checked_shards']} of {len(by)} shards have Hub sha256 metadata")
elif not meta:
    out["sha256"] = "NOT checked (Hub metadata unreachable); structural check only"
out["problems"] = problems[:12]
out["ok"] = not problems
done()
