"""Round G dedupe gate + UG manifests (data fork). Same rules and draw function as round F's vf_mix.py / fork B's ve_mix.py.
usage: python vg_mix.py nevertrain      hash every image of the never-train sources fork A downloaded (hashes only) -> nevertrain_hashes.jsonl
       python vg_mix.py pool            dedupe every round G item list -> lists/<source>.jsonl (kept rows), drops, pool_stats.json
       python vg_mix.py <config.json>   manifest = a base manifest's rows UNCHANGED (same lines, same relative order) + round G rows
Config keys: name; base_path; base_sha256; line ("nc"); add {source: rows}; hm_add {source: rows} (drawn 50 / 50 by label);
extra_e {source: rows} (rows of a round E list that are NOT already in the base, drawn with a new tag).
Dedupe (sha256 of source and stored bytes + 64-bit pHash, Hamming <= 6) against: the 10,742 public eval images, the dev set, fork A's
private-like proxy list, CharXiv val + test figures, every image of the never-train sources on disk (CountBenchQA, What's Up, MME-RealWorld-Lite,
ChartQAPro, ChartMuseum, Multi3Hate); sha256 against the RBH list (MathVision test; the list carries no pHash)."""
import hashlib
import io
import json
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, "/data/local/visdata/scripts")
from vb_manifest import PH_T, public_hashes  # noqa: E402
from vb_build import phash  # noqa: E402

E = Path("/data/shared/visdata_e")
F = Path("/data/shared/visdata_f")
G = Path("/data/shared/visdata_g")
X = "/data/shared/visdata_priv/privlike_image_exclusion.jsonl"
RBH = F / "dl" / "rbh" / "rbh_image_exclusion.jsonl"
NT_DL = Path("/data/shared/visdata_priv/dl")
NT_SRC = ["vikhyatk__CountBenchQA", "amitakamath2__whatsup_vlms", "yifanzhang114__MME-RealWorld-Lite", "ahmed-masry__ChartQAPro", "lytang__ChartMuseum",
          "MinhDucBui__Multi3Hate"]
SEED = 20261114
HM = "Moderation (Hateful Memes)"
NEVER_WORDS = ("multi3hate", "sun rgb", "sunrgb", "countbench", "whatsup", "what's up", "mme-realworld", "chartmuseum", "chartqapro", "multi-chart",
               "visualwebbench", "sroie", "winoground", "mathvision", "ade20k")
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff")


def _hash_bytes(b):
    from PIL import Image

    try:
        im = Image.open(io.BytesIO(b))
        im.load()
        ph = f"{phash(im):016x}"
    except Exception:
        ph = None
    return hashlib.sha256(b).hexdigest(), ph


def nevertrain():
    """every image of the never-train sources fork A downloaded: loose files, zip members, parquet image columns (bytes); hashes only.
    One file per source, renamed into place only when the source is complete (a killed run redoes that source)."""
    import pyarrow.parquet as pq

    NT = G / "nevertrain"
    NT.mkdir(parents=True, exist_ok=True)
    for src in NT_SRC:
        fin = NT / f"{src}.jsonl"
        if fin.exists():
            continue
        part = NT / f"{src}.jsonl.part"
        d = NT_DL / src
        n = 0
        with open(part, "w") as o:
            for p in sorted(d.rglob("*")):
                if ".cache" in p.parts or not p.is_file():
                    continue
                if p.suffix.lower() in IMG_EXT:
                    sh, ph = _hash_bytes(p.read_bytes())
                    o.write(json.dumps({"src": src, "path": str(p.relative_to(d)), "sha256": sh, "phash": ph}) + "\n")
                    n += 1
                elif p.suffix.lower() == ".zip":
                    z = zipfile.ZipFile(p)
                    for m in z.namelist():
                        if m.lower().endswith(IMG_EXT):
                            sh, ph = _hash_bytes(z.read(m))
                            o.write(json.dumps({"src": src, "path": f"{p.name}:{m}", "sha256": sh, "phash": ph}) + "\n")
                            n += 1
                elif p.suffix.lower() == ".parquet":
                    pf = pq.ParquetFile(p)
                    cols = [f.name for f in pf.schema_arrow if "image" in f.name.lower() or f.name.lower() in ("img", "chart", "figure")]
                    for rg in range(pf.num_row_groups):
                        for ri, r in enumerate(pf.read_row_group(rg, columns=cols).to_pylist()):
                            for col in cols:
                                vals = r[col] if isinstance(r[col], list) else [r[col]]
                                for v in vals:
                                    b = v.get("bytes") if isinstance(v, dict) else v if isinstance(v, (bytes, bytearray)) else None
                                    if b:
                                        sh, ph = _hash_bytes(bytes(b))
                                        o.write(json.dumps({"src": src, "path": f"{p.name}:{rg}:{ri}:{col}", "sha256": sh, "phash": ph}) + "\n")
                                        n += 1
        part.rename(fin)
        print(json.dumps({"nevertrain_source": src, "images": n}), flush=True)
    with open(G / "nevertrain_hashes.jsonl", "w") as o:  # the merged list (the gate reads this file)
        for src in NT_SRC:
            o.write((NT / f"{src}.jsonl").read_text())
    print("G_NEVERTRAIN_DONE", flush=True)


def charxiv_hashes():
    return [json.loads(l) for l in open(F / "charxiv_hashes.jsonl")]  # round F's cache of princeton-nlp/CharXiv@f441eb63 val + test images


def gates():
    """-> [(name, sha set, pHash array)] in the order the reasons are reported"""
    pub = public_hashes()
    dv = [json.loads(l) for l in open(E / "dev_hashes.jsonl")]
    xr = [json.loads(l) for l in open(X)]
    rb = [json.loads(l) for l in open(RBH)]
    cx = charxiv_hashes()
    nt = [json.loads(l) for l in open(G / "nevertrain_hashes.jsonl")]
    assert {r["src"] for r in nt} == set(NT_SRC), "never-train hashes incomplete"
    arr = lambda rows: np.array([int(r["phash"], 16) for r in rows if r.get("phash")], dtype=np.uint64)  # noqa: E731
    g = [("public", {r["sha256"] for r in pub}, arr(pub)),
         ("dev", {r["sha256"] for r in dv}, arr(dv)),
         ("proxy", {r[k] for r in xr for k in ("sha256", "src_sha256") if isinstance(r.get(k), str)}, arr(xr)),
         ("rbh", {r[k] for r in rb for k in ("img_sha256", "img_src_sha256") if isinstance(r.get(k), str)}, np.zeros(0, dtype=np.uint64)),
         ("charxiv", {r["sha256"] for r in cx}, arr(cx)),
         ("nevertrain", {r["sha256"] for r in nt}, arr(nt))]
    sizes = {"public_images": len(pub), "dev_images": len(dv), "proxy_images": len(xr), "rbh_images": len(rb), "charxiv_val_test_images": len(cx),
             "nevertrain_images": dict(Counter(r["src"] for r in nt)),
             "proxy_list_md5": hashlib.md5(Path(X).read_bytes()).hexdigest(), "rbh_list_md5": hashlib.md5(RBH.read_bytes()).hexdigest(),
             "public_hashes_md5": hashlib.md5(Path("/data/local/visdata/public_hashes.jsonl").read_bytes()).hexdigest()}
    return g, sizes


def why(it, G_):
    for name, sha, ph in G_:
        for s1, s2, p in zip(it["img_src_sha256"], it["img_sha256"], it["img_phash"]):
            if s1 in sha or s2 in sha:
                return name
            if len(ph) and int(np.bitwise_count(ph ^ np.uint64(int(p, 16))).min()) <= PH_T:
                return name
    return None


def pool():
    G_, sizes = gates()
    stats, drops = defaultdict(Counter), []
    LS = G / "lists"
    LS.mkdir(parents=True, exist_ok=True)
    lic = {}
    for f in sorted((G / "items_g").glob("*.jsonl")):
        keep = []
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            s = r["source"]
            lic[s] = [r["line"], r["licence"]]
            stats[s]["built"] += 1
            w = why(r, G_)
            if w:
                stats[s]["drop_" + w] += 1
                drops.append({"id": r["id"], "source": s, "why": w})
            else:
                stats[s]["kept"] += 1
                keep.append(r)
        if not keep:
            continue
        s = keep[0]["source"]
        assert not any(wd in lic[s][1].lower() for wd in NEVER_WORDS), f"{s}: licence string names a never-train source"
        with open(LS / f"{s}.jsonl", "w", encoding="utf-8") as o:
            for r in keep:
                o.write(json.dumps(r, ensure_ascii=False) + "\n")
        cnt = [r for r in keep if r["subtask"] in ("Count", "Counting") and len(r["images"]) == 1 and all(x.isdigit() for x in r["options"])]
        if cnt:  # labeller input for the helper fork (vh_facts.py label), same fields as rounds E / F
            with open(LS / f"{s}.label_in.jsonl", "w", encoding="utf-8") as o:
                for r in cnt:
                    x = r.get("extra") or {}
                    o.write(json.dumps({"id": r["id"], "image": r["images"][0], "question": r["question"],
                                        **({"phrases": [x["class"].lower()]} if isinstance(x.get("class"), str) else {}),
                                        "gold_count": int(r["options"]["ABCDEFGH".index(r["gold"])]), "gold_basis": "human annotation"},
                                       ensure_ascii=False) + "\n")
        print(json.dumps({"source": s, **stats[s]}), flush=True)
    M = G / "manifests_g"
    M.mkdir(parents=True, exist_ok=True)
    with open(M / "g_dedupe_drops.jsonl", "w") as o:
        for r in drops:
            o.write(json.dumps(r) + "\n")
    (M / "pool_stats.json").write_text(json.dumps({"round_g_pool": {k: dict(v) for k, v in sorted(stats.items())}, "dedupe_against": sizes,
                                                    "phash_threshold": PH_T, "licences": lic,
                                                    "winoground": "never on disk, never in any list"}, indent=1, ensure_ascii=False))
    print("G_POOL_DONE", flush=True)


reps = {}


def draw(rows, n, tag):
    """ve_mix.draw: id-sorted, seeded shuffle, whole passes repeated with #r<k> ids when n exceeds the pool"""
    rows = sorted(rows, key=lambda r: r["id"])
    got, k = [], 0
    while len(got) < n and rows:
        order = rows[:]
        random.Random(f"{SEED}:{tag}:{k}").shuffle(order)
        need = n - len(got)
        got += [{**r, "id": r["id"] if k == 0 else f"{r['id']}#r{k}"} for r in order[:need]]
        k += 1
    reps[tag] = round(n / max(1, len(rows)), 2)
    return got


def balanced(rows, n, yes_share, tag):
    y = [r for r in rows if r["gold"] == "A"]
    no = [r for r in rows if r["gold"] == "B"]
    ny = round(n * yes_share)
    return draw(y, ny, tag + ":yes") + draw(no, n - ny, tag + ":no")


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def mix(cfg):
    base_p = Path(cfg["base_path"])
    base_bytes = base_p.read_bytes()
    assert hashlib.sha256(base_bytes).hexdigest() == cfg["base_sha256"], "base manifest sha256 differs from the config"
    base_lines = [l + b"\n" for l in base_bytes.split(b"\n") if l]  # bytes, split on \n only: the rows are written back untouched
    assert b"".join(base_lines) == base_bytes
    base_ids = [json.loads(l)["id"] for l in base_lines]
    base_idset = set(base_ids)
    assert len(base_idset) == len(base_ids)
    add, notes = [], []
    for s, k in cfg.get("add", {}).items():
        rows = load(G / "lists" / f"{s}.jsonl")
        assert len(rows) >= k, f"{s}: pool {len(rows)} < {k} (no repeats in round G)"
        add += draw(rows, k, f"G:{s}")
    for s, k in cfg.get("hm_add", {}).items():
        rows = load(G / "lists" / f"{s}.jsonl")
        add += balanced(rows, k, 0.5, f"GHM:{s}")
    for s, k in cfg.get("extra_e", {}).items():  # round E list rows that the base does not hold (fork B's gate already applied), new draw tag
        rows = [r for r in load(E / "lists" / f"{s}.jsonl") if r["id"] not in base_idset]
        assert len(rows) >= k, f"{s}: only {len(rows)} rows outside the base"
        add += draw(rows, k, f"GE:{s}")
        notes.append(f"{s}: {k} of the {len(rows)} round E list rows that the base does not hold")
    add_ids = [r["id"] for r in add]
    assert len(set(add_ids)) == len(add_ids) and not (set(add_ids) & base_idset), "duplicate ids"
    assert all("#r" not in i for i in add_ids), "a round G source repeated rows"
    total = len(base_lines) + len(add)
    random.Random(f"{SEED}:order:{cfg['name']}").shuffle(add)
    slots = set(random.Random(f"{SEED}:slots:{cfg['name']}").sample(range(total), len(add)))
    M = G / "manifests_g"
    M.mkdir(parents=True, exist_ok=True)
    p = M / f"{cfg['name']}.jsonl"
    bi, ai = iter(base_lines), iter(add)
    with open(p, "wb") as f:
        for i in range(total):
            f.write((json.dumps(next(ai), ensure_ascii=False) + "\n").encode("utf-8") if i in slots else next(bi))
    out = load(p)
    h = hashlib.sha256()
    for l in open(p, "rb"):  # the base rows, as they sit in the new file (picked by id), hash to the base manifest
        if json.loads(l)["id"] in base_idset:
            h.update(l)
    assert h.hexdigest() == cfg["base_sha256"], "base rows changed"
    assert len(out) == total and len({r["id"] for r in out}) == total and total % 64 == 0, (total, total % 64)
    G_, sizes = gates()
    final = Counter()
    for r in out:  # exact-bytes check of EVERY row of the manifest (base rows included) against the hard lists
        for name, sha, _ in G_:
            if name in ("proxy", "rbh", "charxiv", "nevertrain") and any(s in sha for s in r["img_src_sha256"] + r["img_sha256"]):
                final[f"{name}:{r['source']}"] += 1
    have, miss = {}, 0
    for r in add:  # every added image exists in the packed dirs (round G: img_g; round E rows: img_e)
        root = E / "img_e" if r["source"] in cfg.get("extra_e", {}) else G / "img_g"
        for q in r["images"]:
            d = q.split("/")[0]
            if (root, d) not in have:
                have[(root, d)] = {x.name for x in (root / d).iterdir()}
            miss += q.split("/", 1)[1] not in have[(root, d)]
    assert miss == 0, f"{miss} added images missing"
    fam = Counter(r["family"] for r in out)
    hmr = [r for r in out if r["family"] == HM]
    ps = json.loads((M / "pool_stats.json").read_text())
    new_src = sorted(set(cfg.get("add", {})) | set(cfg.get("hm_add", {})) | set(cfg.get("extra_e", {})))
    hdr = {"manifest": cfg["name"], "config": cfg, "rows": total, "steps_at_64": total / 64, "rows_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
           "base": {"path": str(base_p), "rows": len(base_lines), "rows_sha256": cfg["base_sha256"],
                    "note": "base rows are byte-identical and in the same relative order; the base ids, picked from the new file, reproduce the base file (asserted)"},
           "added_rows": len(add), "added_per_source": dict(sorted(Counter(r["source"] for r in add).items())),
           "added_per_family": dict(Counter(r["family"] for r in add)), "added_per_subtask": {f"{k[0]}|{k[1]}": v for k, v in sorted(Counter((r["family"], r["subtask"]) for r in add).items())},
           "per_family": dict(fam), "per_family_share": {k: round(v / total, 4) for k, v in fam.items()}, "repeat_factor": dict(reps),
           "per_source": dict(sorted(Counter(r["source"] for r in out).items())), "per_line": dict(Counter(r["line"] for r in out)),
           "hm_yes_share": round(sum(r["gold"] == "A" for r in hmr) / max(1, len(hmr)), 4),
           "hm_per_source_yes": {s: [sum(1 for r in hmr if r["source"] == s and r["gold"] == "A"), sum(1 for r in hmr if r["source"] == s)]
                                 for s in sorted({r["source"] for r in hmr})},
           "round_g_pool": ps["round_g_pool"], "dedupe": {**sizes, "phash_threshold": PH_T, "winoground": "never on disk, never in any list"},
           "exact_hash_hits_in_manifest_vs_proxy_rbh_charxiv_nevertrain": dict(final), "added_images_missing": miss,
           "new_sources": new_src, "licences": {r["source"]: [r["line"], r["licence"]] for r in out}, "notes": notes}
    (M / f"{cfg['name']}.header.json").write_text(json.dumps(hdr, indent=1, ensure_ascii=False))
    print(json.dumps({k: hdr[k] for k in ("manifest", "rows", "steps_at_64", "rows_sha256", "added_rows", "added_per_source", "added_per_family",
                                          "per_family", "per_line", "hm_yes_share", "repeat_factor", "exact_hash_hits_in_manifest_vs_proxy_rbh_charxiv_nevertrain")}))
    print(f"G_MIX_DONE {cfg['name']}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "pool":
        pool()
    elif sys.argv[1] == "nevertrain":
        nevertrain()
    else:
        mix(json.loads(Path(sys.argv[1]).read_text()))
