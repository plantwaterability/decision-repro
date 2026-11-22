"""peek at PixMo-Points / PixMo-Count metadata (labels, counts, methods, url hosts); no image is fetched here"""
import collections, json, sys
from urllib.parse import urlparse
import pyarrow.parquet as pq
import pyarrow.compute as pc
G = "/data/shared/visdata_g/dl"
t = pq.read_table([f"{G}/pixmo_points/train-0000{i}-of-00002.parquet" for i in (0, 1)], columns=["image_url", "image_sha256", "count", "label", "collection_method"])
print("pixmo_points rows", t.num_rows)
m = t.column("collection_method").to_pylist(); c = t.column("count").to_pylist(); lab = t.column("label").to_pylist(); url = t.column("image_url").to_pylist(); sh = t.column("image_sha256").to_pylist()
print("methods", collections.Counter(m))
for meth in ("counting", "pointing"):
    idx = [i for i in range(len(m)) if m[i] == meth]
    imgs = {sh[i] for i in idx}
    print(meth, "rows", len(idx), "images", len(imgs))
    h = collections.Counter(min(c[i], 25) for i in idx); print(" count hist", dict(sorted(h.items())))
    lc = collections.Counter(lab[i].strip().lower() for i in idx); print(" labels distinct", len(lc)); print(" top labels", lc.most_common(80))
    nw = collections.Counter(min(len(lab[i].split()), 6) for i in idx); print(" words per label", dict(sorted(nw.items())))
    per = collections.Counter(sh[i] for i in idx); print(" labels per image", dict(sorted(collections.Counter(min(v, 15) for v in per.values()).items())))
    hosts = collections.Counter(urlparse(url[i]).netloc for i in idx); print(" hosts", len(hosts), hosts.most_common(25))
    small = [i for i in idx if 0 <= c[i] <= 10]; print(" rows with count 0-10", len(small), "images", len({sh[i] for i in small}))
t2 = pq.read_table(f"{G}/pixmo_count/train-00000-of-00001.parquet", columns=["image_url", "image_sha256", "count", "label"])
c2 = t2.column("count").to_pylist(); l2 = t2.column("label").to_pylist(); u2 = t2.column("image_url").to_pylist()
print("pixmo_count train rows", t2.num_rows, "images", len(set(t2.column("image_sha256").to_pylist())), "count hist", dict(sorted(collections.Counter(c2).items())))
print(" top labels", collections.Counter(x.lower() for x in l2).most_common(40)); print(" hosts", collections.Counter(urlparse(x).netloc for x in u2).most_common(15))
print("sample rows", [(lab[i], c[i], m[i], url[i][:90]) for i in range(0, len(m), len(m) // 12)])
