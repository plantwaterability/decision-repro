"""PixMo URL resolution test: a seeded random sample of rows, fetch with TLS verification on, compare sha256 with the dataset's image_sha256.
Also: pointing-method rows, count vs number of points."""
import collections, hashlib, json, random, sys, time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
import httpx
import pyarrow.parquet as pq
G = "/data/shared/visdata_g/dl"
UA = {"User-Agent": "Mozilla/5.0 (compatible; img-fetch/1.0)"}
t = pq.read_table([f"{G}/pixmo_points/train-0000{i}-of-00002.parquet" for i in (0, 1)], columns=["image_url", "image_sha256", "count", "label", "collection_method", "points"])
m = t.column("collection_method").to_pylist(); c = t.column("count").to_pylist()
idx_p = [i for i in range(len(m)) if m[i] == "pointing"]
rs = random.Random(1).sample(idx_p, 20000)
pts = t.column("points").take(rs).to_pylist(); cc = [c[i] for i in rs]
agree = collections.Counter(("none" if x is None else ("eq" if x == len(p or []) else "ne")) for x, p in zip(cc, pts))
print("pointing: count vs len(points) on 20,000 rows", dict(agree), "len(points) hist", dict(sorted(collections.Counter(min(len(p or []), 12) for p in pts).items())))
print("pointing count hist (non-None)", dict(sorted(collections.Counter(min(x, 12) for x in cc if x is not None).items())))
url = t.column("image_url").to_pylist(); sh = t.column("image_sha256").to_pylist()
def one(i):
    u = url[i]
    try:
        r = httpx.get(u, timeout=30, follow_redirects=True, headers=UA)
        if r.status_code != 200 or not r.content:
            return "http_%d" % r.status_code
        return "sha_ok" if hashlib.sha256(r.content).hexdigest() == sh[i] else "sha_differs"
    except httpx.ConnectError as e:
        return "connect_error" + (":tls" if "CERTIFICATE" in str(e).upper() or "SSL" in str(e).upper() else "")
    except Exception as e:
        return type(e).__name__
for name, pool in (("counting", [i for i in range(len(m)) if m[i] == "counting"]), ("pointing", idx_p)):
    imgs = {}
    for i in pool:
        imgs.setdefault(sh[i], i)
    samp = random.Random(f"urltest:{name}").sample(sorted(imgs.values()), 500)
    t0 = time.time()
    with ThreadPoolExecutor(32) as ex:
        res = list(ex.map(one, samp))
    cnt = collections.Counter(res)
    byhost = collections.defaultdict(collections.Counter)
    for i, r in zip(samp, res):
        byhost[urlparse(url[i]).netloc][r] += 1
    print(name, "sample 500 images", dict(cnt), "ok share %.3f" % (cnt["sha_ok"] / 500), "secs %.0f" % (time.time() - t0))
    print(" by host (top 15)", [(h, dict(v)) for h, v in sorted(byhost.items(), key=lambda kv: -sum(kv[1].values()))[:15]])
t2 = pq.read_table(f"{G}/pixmo_count/train-00000-of-00001.parquet", columns=["image_url", "image_sha256"])
u2 = t2.column("image_url").to_pylist(); s2 = t2.column("image_sha256").to_pylist()
url, sh = u2, s2
samp = random.Random("urltest:count").sample(range(len(u2)), 300)
with ThreadPoolExecutor(32) as ex:
    res = list(ex.map(one, samp))
cnt = collections.Counter(res); print("pixmo_count train sample 300", dict(cnt), "ok share %.3f" % (cnt["sha_ok"] / 300))
