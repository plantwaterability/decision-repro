"""Vision recipe data build (ctl2, CPU). One family per call: python vb_build.py <family>

Each family writes items/<family>.jsonl (kit item fields + source/licence/line/family) and its images under img/<family>/.
Board-format conversions reuse the kit's own MIT code (decision_index.vision.build / convert at 25c6ec7) on TRAIN-side
splits that are disjoint from the board's pools; every saved image records sha256 of the source bytes, sha256 of the
saved bytes and a 64-bit DCT pHash for the dedupe gate (vb_manifest.py).
"""
import hashlib
import io
import json
import os
import random
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

KIT = "/data/local/visdata/kit/kit_vis"
sys.path.insert(0, KIT)
os.environ.setdefault("DECISION_INDEX_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
from decision_index.vision.build import m2w_convert, parse_lettered  # noqa: E402
from decision_index.vision.convert import (CORD_PRESENT_Q, CORD_VALUE_FIELDS, CORD_VALUE_Q, LETTERS, SEED, Letters, arrange,  # noqa: E402
                                           cord_fields, fmt_dec, fmt_int, funsd_type, norm, pick_by_rank, parse_amount)

from PIL import Image as _PILImage  # noqa: E402

_PILImage.MAX_IMAGE_PIXELS = None  # the kit's own Ctx does this; full-page Mind2Web screenshots exceed PIL's bomb limit
D = Path("/data/local/visdata")
DL = Path("/data/shared/visdata_dl")
IMG = D / "img"
ITEMS = D / "items"
MAXPIX = 1600 * 32 * 32

LIC = {
    "m2w": ("Mind2Web", "clean", "osunlp/Multimodal-Mind2Web train (card: OpenRAIL; Mind2Web annotations CC BY 4.0); website screenshots"),
    "infovqa_train": ("InfographicVQA", "nc", "InfographicVQA train via lmms-lab-encoder/DocVQA (upstream RRC research terms)"),
    "docvqa": ("InfographicVQA", "clean", "DocVQA train, pixparse@33136ef4 (MIT; UCSF industry documents)"),
    "cord_train": ("KIE (CORD+FUNSD)", "clean", "naver-clova-ix/cord-v2 train (CC BY 4.0)"),
    "funsd_train": ("KIE (CORD+FUNSD)", "nc", "FUNSD training_data (non-commercial research/education)"),
    "hm_train": ("Moderation (Hateful Memes)", "nc", "Hateful Memes train via neuralcatcher/hateful_memes (Facebook research licence; Getty images)"),
    "meme_v1": ("Moderation (Hateful Memes)", "clean", "meme_v1: CC BY 4.0 text (HateCheck, DynaHate, MHS, Gab, CAD) on DOCCI photos (CC BY 4.0)"),
    "coco_count": ("CV-Bench", "clean", "Nemotron-Image-Training-v3.5 coco_count_spatial_qa (OpenMDW-1.1); COCO train2017 images"),
    "vrd25": ("CV-Bench", "clean", "2.5VRD (CC BY 4.0); Open Images V4 per-image terms"),
    "vsr": ("BLINK", "clean", "VSR random-split train (CC BY 4.0); COCO train2017 images; BLINK test images removed at V1 build"),
    "sat": ("RealWorldQA", "clean", "Nemotron-Image-Training-v3.5 sat_spatial_qa (OpenMDW-1.1); SAT images"),
    "mindcube": ("BLINK", "clean", "MLL-Lab/MindCube train (MIT)"),
    "coco_wino": ("Winoground", "clean", "COCO captions train2017 (CC BY 4.0 annotations); COCO train2017 images; order-swapped negatives"),
    "chartnet": ("CharXiv", "clean", "ibm-granite/ChartNet grounded_qa (CDLA-Permissive-2.0; synthetic charts)"),
}

_DCT = None


def phash(im):
    import numpy as np
    from PIL import Image

    global _DCT
    if _DCT is None:
        n = 32
        k = np.arange(n)
        m = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
        m[0] *= 1 / np.sqrt(2)
        _DCT = m * np.sqrt(2 / n)
    a = np.asarray(im.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    c = (_DCT @ a @ _DCT.T)[:8, :8].flatten()
    bits = c > np.median(c[1:])
    return int("".join("1" if b else "0" for b in bits), 2)


class Saver:
    def __init__(self, fam):
        self.fam = fam
        self.dir = IMG / fam
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = {}

    def save(self, data, stem):
        from PIL import Image

        if stem in self.meta:
            return self.meta[stem]["path"]
        src_sha = hashlib.sha256(data).hexdigest()
        im = Image.open(io.BytesIO(data))
        im.load()
        fmt = im.format
        w, h = im.size
        ph = phash(im)
        if w * h > MAXPIX:
            s = (MAXPIX / (w * h)) ** 0.5
            im = im.convert("RGB").resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
            p = self.dir / f"{stem}.jpg"
            im.save(p, quality=93)
        else:
            ext = {"JPEG": "jpg", "MPO": "jpg", "PNG": "png", "WEBP": "webp", "GIF": "png", "BMP": "png"}.get(fmt, "png")
            p = self.dir / f"{stem}.{ext}"
            if ext == "png" and fmt != "PNG":
                im.convert("RGB").save(p)
            else:
                p.write_bytes(data)
        rel = str(p.relative_to(IMG))
        self.meta[stem] = {"path": rel, "src_sha256": src_sha, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "phash": f"{ph:016x}"}
        return rel

    def info(self, rel):
        for m in self.meta.values():
            if m["path"] == rel:
                return m
        return None


def item(fam, sid, images, question, options, gold, subtask, saver, extra=None, prompt=None):
    family, line, lic = LIC[fam]
    assert gold in LETTERS[:len(options)], (fam, sid, gold, options)
    metas = [saver.info(p) for p in images]
    it = {"id": f"{fam}:{sid}", "family": family, "source": fam, "line": line, "licence": lic, "subtask": subtask, "images": images,
          "question": question, "options": [str(o) for o in options], "gold": gold,
          "prompt": prompt or {"options_in_image": False, "readout": None},
          "img_src_sha256": [m["src_sha256"] for m in metas], "img_sha256": [m["sha256"] for m in metas],
          "img_phash": [m["phash"] for m in metas]}
    if extra:
        it["extra"] = extra
    return it


def iter_parquet(path, columns=None):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        for r in pf.read_row_group(rg, columns=columns).to_pylist():
            yield r


# ---------------------------------------------------------------- Mind2Web train (kit m2w_convert, unchanged)
def b_m2w():
    sv = Saver("m2w")

    class Ctx:
        img = sv.dir
        root = IMG
        font = os.environ["DECISION_INDEX_FONT"]

    cols = ["action_uid", "operation", "pos_candidates", "neg_candidates", "domain", "website", "annotation_id",
            "confirmed_task", "screenshot", "action_reprs", "target_action_index"]
    out, off = [], 0
    for p in sorted((DL / "m2w" / "data").glob("train-*.parquet")):
        for row in iter_parquet(p, cols):
            row["_off"] = off
            off += 1
            sid = row["action_uid"]
            it = m2w_convert(Ctx, row, sid, random.Random(f"{SEED}:train_mind2web:{sid}"))
            if not it:
                continue
            from PIL import Image

            rel = it["images"][0]
            data = (IMG / rel).read_bytes()
            sv.meta[rel] = {"path": rel, "src_sha256": hashlib.sha256(data).hexdigest(), "sha256": hashlib.sha256(data).hexdigest(),
                            "phash": f"{phash(Image.open(io.BytesIO(data))):016x}"}
            out.append(item("m2w", sid, [rel], it["question"], it["options"], it["gold"], it["subtask"], sv,
                            extra={"website": row["website"], "annotation_id": row["annotation_id"], "op": it.get("op")}))
    return out


# ---------------------------------------------------------------- InfographicVQA-style (kit full_infovqa rule, k per image)
def infovqa_like(fam, metas, image_bytes, per_image, seed_off, order_key=None):
    """metas: list of dicts with qid, question, answers, img_key. image_bytes(meta) -> bytes."""
    sv = Saver(fam)
    byimg = {}
    for m in metas:
        byimg.setdefault(m["img_key"], []).append(m)

    def isnum(a):
        return bool(re.search(r"\d", a))

    cands = []
    for m in metas:
        if not m["answers"]:
            continue
        gold = m["answers"][0].strip().lower()
        golds = {norm(a) for a in m["answers"]}
        pool = []
        for o in byimg[m["img_key"]]:
            if o["qid"] == m["qid"] or not o["answers"]:
                continue
            a = o["answers"][0].strip().lower()
            if isnum(a) != isnum(gold) or norm(a) in golds or not norm(a):
                continue
            if any(norm(a) in g or g in norm(a) for g in golds if g):
                continue
            pool.append(a)
        pool = list(dict.fromkeys(pool))
        if len(pool) >= 3 and gold:
            cands.append((m, gold, pool, "same_image"))
        elif gold:
            cands.append((m, gold, pool, "xdoc"))
    rng = random.Random(SEED + seed_off)
    glob_pool = {True: [], False: []}
    for m in metas:
        if m["answers"]:
            a = m["answers"][0].strip().lower()
            if norm(a):
                glob_pool[isnum(a)].append(a)
    for k in glob_pool:
        glob_pool[k] = list(dict.fromkeys(glob_pool[k]))
    rng.shuffle(cands)
    cands.sort(key=lambda cd: cd[3] != "same_image")  # same-image distractors (the kit's rule) first, cross-doc fill second
    per, pick = Counter(), []
    for cd in cands:
        if per[cd[0]["img_key"]] >= per_image:
            continue
        m, gold, pool, how = cd
        if how == "xdoc":
            golds = {norm(a) for a in m["answers"]}
            extra = [a for a in rng.sample(glob_pool[isnum(gold)], min(60, len(glob_pool[isnum(gold)])))
                     if norm(a) not in golds and not any(norm(a) in g or g in norm(a) for g in golds if g) and a not in pool]
            pool = (pool + extra)[:max(3, len(pool))]
            if len(pool) < 3:
                continue
        per[m["img_key"]] += 1
        pick.append((m, gold, pool, how))
    out = []
    if order_key is not None:  # read images in file order (one parquet image column load per file), as the kit does
        pick.sort(key=lambda cd: order_key(cd[0]))
    for m, gold, pool, how in pick:
        rel = sv.save(image_bytes(m), re.sub(r"[^A-Za-z0-9_.-]", "_", str(m["img_key"]))[:80])
        ds = rng.sample(pool, 3)
        opts = ds + [gold]
        rng.shuffle(opts)
        sub = ("numeric" if isnum(gold) else "text") + ("" if how == "same_image" else "_xdoc")
        out.append(item(fam, m["qid"], [rel], m["question"], opts, LETTERS[opts.index(gold)], sub, sv))
    return out


def b_infovqa_train():
    import pyarrow.parquet as pq

    files = sorted((DL / "infovqa" / "InfographicVQA").glob("train-*.parquet"))
    metas, where = [], {}
    for fi, f in enumerate(files):
        for ri, r in enumerate(pq.read_table(f, columns=["questionId", "question", "answers", "image_url"]).to_pylist()):
            metas.append({"qid": r["questionId"], "question": r["question"], "answers": r["answers"] or [], "img_key": r["image_url"]})
            where.setdefault(r["image_url"], (fi, ri))
    tables = {}

    def ib(m):
        fi, ri = where[m["img_key"]]
        if fi not in tables:
            tables.clear()
            tables[fi] = pq.read_table(files[fi], columns=["image"])
        return tables[fi].slice(ri, 1).to_pylist()[0]["image"]["bytes"]

    metas.sort(key=lambda m: where[m["img_key"]])
    return infovqa_like("infovqa_train", metas, ib, per_image=2, seed_off=1012, order_key=lambda m: where[m["img_key"]])


def b_docvqa():
    import pyarrow.parquet as pq

    files = sorted(DL.glob("docvqa/**/train-*.parquet"))
    metas, where = [], {}
    for fi, f in enumerate(files):
        for ri, r in enumerate(pq.read_table(f, columns=["question_id", "question", "answers", "other_metadata"]).to_pylist()):
            key = (r.get("other_metadata") or {}).get("image") or f"doc{(r.get('other_metadata') or {}).get('doc_id')}_{fi}_{ri}"
            metas.append({"qid": str(r["question_id"]), "question": r["question"], "answers": list(r["answers"] or []), "img_key": key})
            where.setdefault(key, (fi, ri))
    tables = {}

    def ib(m):
        fi, ri = where[m["img_key"]]
        if fi not in tables:
            tables.clear()
            tables[fi] = pq.read_table(files[fi], columns=["image"])
        return tables[fi].slice(ri, 1).to_pylist()[0]["image"]["bytes"]

    metas.sort(key=lambda m: where[m["img_key"]])
    return infovqa_like("docvqa", metas, ib, per_image=3, seed_off=1013, order_key=lambda m: where[m["img_key"]])


# ---------------------------------------------------------------- CORD train (kit full_cord rule on the train shards)
def b_cord_train():
    import pyarrow.parquet as pq

    sv = Saver("cord_train")
    paths = sorted((DL / "cord" / "data").glob("train-*.parquet"))
    recs = []
    for si, p in enumerate(paths):
        for i, gt in enumerate(pq.read_table(p, columns=["ground_truth"]).column("ground_truth").to_pylist()):
            recs.append((f"train{si}-{i}", si, i, json.loads(gt)["gt_parse"]))
    rng = random.Random(SEED + 1101)
    rng.shuffle(recs)
    plan = []
    for key, si, i, gp in recs:
        f, amounts, menu = cord_fields(gp)
        vals = {}
        for a_, k_ in amounts:
            vals.setdefault(a_, set()).add(k_)
        if len(vals) < 3:
            continue
        for fld in CORD_VALUE_FIELDS:
            if fld == "item_price":
                names = Counter(n.lower() for n, _ in menu)
                for j, (n, pr) in enumerate(menu):
                    if names[n.lower()] == 1 and 2 <= len(n) <= 40 and re.search(r"[A-Za-z]{2}", n):
                        plan.append((key, fld, j, n, pr, vals))
            else:
                g = parse_amount(f.get(fld))
                if g:
                    plan.append((key, fld, None, None, g, vals))
    n_elig = Counter(pl[1] for pl in plan)
    cap = {fld: -(-n_elig[fld] // 2) for fld in CORD_VALUE_FIELDS}
    letters = Letters(rng)
    rank_counts = {fld: Counter() for fld in CORD_VALUE_FIELDS}
    tables = {}
    where = {key: (si, i) for key, si, i, _ in recs}

    def image(key):
        si, i = where[key]
        if si not in tables:
            tables[si] = pq.read_table(paths[si], columns=["image"])
        return sv.save(tables[si].slice(i, 1).to_pylist()[0]["image"]["bytes"], key)

    out = []
    for key, fld, j, nm, gold_v, vals in plan:
        sid = f"{key}:value:{fld}" + (f":{j}" if j is not None else "")
        q = CORD_VALUE_Q[fld].format(nm=nm) if fld == "item_price" else CORD_VALUE_Q[fld]
        pool = [(v, None) for v in vals if abs(v - gold_v) > 1e-9]
        dec = any(isinstance(v, float) and v != int(v) for v in list(vals) + [gold_v])
        fm = fmt_dec if dec else fmt_int
        ch = pick_by_rank(gold_v, pool, 3, rng, rank_counts[fld], cap[fld])
        if ch is None:
            continue
        pos = letters.pick(len(ch) + 1)
        dv = [x[0] for x in ch]
        rng.shuffle(dv)
        ov = dv[:pos] + [gold_v] + dv[pos:]
        opts = [fm(v) for v in ov]
        if len(set(opts)) != len(opts):
            continue
        out.append(item("cord_train", sid, [image(key)], q, opts, LETTERS[pos], "field_value", sv, extra={"field": fld}))
    present = {}
    for key, si, i, gp in sorted(recs, key=lambda r: r[0]):
        f = cord_fields(gp)[0]
        for fld in CORD_PRESENT_Q:
            has = fld in f and parse_amount(f[fld] if not isinstance(f[fld], list) else f[fld][0]) is not None
            absent = fld not in f
            if not has and not absent:
                continue
            present.setdefault(fld, {}).setdefault("A" if has else "B", []).append(
                item("cord_train", f"{key}:present:{fld}", [image(key)], f"Does this receipt show {CORD_PRESENT_Q[fld]}?", ["Yes", "No"],
                     "A" if has else "B", "field_present", sv, extra={"field": fld}))
    for fld, g in present.items():  # the board's own balance rule: equal yes/no per field
        m = min(len(v) for v in g.values()) if len(g) == 2 else 0
        for v in g.values():
            out += sorted(v, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())[:m]
    return out


# ---------------------------------------------------------------- FUNSD training_data (kit full_funsd rule)
def b_funsd_train():
    sv = Saver("funsd_train")
    zf = zipfile.ZipFile(DL / "funsd_dataset.zip")
    base = "dataset/training_data"
    forms = sorted(Path(n).stem for n in zf.namelist() if n.startswith(f"{base}/annotations/") and n.endswith(".json"))
    value_c, present_c = [], []
    for fid in forms:
        ents = json.loads(zf.read(f"{base}/annotations/{fid}.json"))["form"]
        byid = {e["id"]: e for e in ents}
        answers = [e for e in ents if e["label"] == "answer" and 1 <= len(e["text"].strip()) <= 60]
        qtexts = Counter(e["text"].strip().lower() for e in ents if e["label"] == "question")
        for e in ents:
            if e["label"] != "question":
                continue
            qt = e["text"].strip()
            if qtexts[qt.lower()] != 1 or not re.search(r"[A-Za-z]{2}", qt) or len(qt) > 50:
                continue
            linked = [byid[b] for a_, b in e["linking"] if a_ == e["id"] and b in byid and byid[b]["label"] == "answer"]
            linked = [x for x in linked if x["text"].strip()]
            name = qt.strip(" :\"'").strip()
            if len(linked) == 1 and 1 <= len(linked[0]["text"].strip()) <= 60:
                gold = linked[0]["text"].strip()
                gt = funsd_type(gold)
                others = sorted({a_["text"].strip() for a_ in answers if a_["id"] != linked[0]["id"] and a_["text"].strip().lower() != gold.lower()})
                same = [o for o in others if funsd_type(o) == gt]
                if len(same) >= 3:
                    value_c.append((fid, e, name, gold, same, gt))
                present_c.append(("yes", fid, name, e))
            elif not linked and not any(byid.get(b, {}).get("label") == "answer" for _, b in e["linking"]):
                x0, y0, x1, y1 = e["box"]
                near = [a_ for a_ in answers if (abs((a_["box"][1] + a_["box"][3]) / 2 - (y0 + y1) / 2) < 12 and 0 <= a_["box"][0] - x1 < 250)
                        or (0 <= a_["box"][1] - y1 < 25 and abs(a_["box"][0] - x0) < 60)]
                if not near:
                    present_c.append(("no", fid, name, e))
    rng = random.Random(SEED + 1103)
    letters = Letters(rng)
    out = []
    for fid, e, name, gold, same, gt in value_c:
        ds = rng.sample(same, 3)
        pos = letters.pick(4)
        opts, gl = arrange(gold, ds, pos, rng)
        out.append(item("funsd_train", f"{fid}:value:{e['id']}", [sv.save(zf.read(f"{base}/images/{fid}.png"), fid)],
                        f"On this form, what is filled in for the field \"{name}\"?", opts, gl, "field_value", sv))
    g = {"A": [], "B": []}
    for lab, fid, name, e in present_c:
        g["A" if lab == "yes" else "B"].append(item("funsd_train", f"{fid}:present:{e['id']}", [sv.save(zf.read(f"{base}/images/{fid}.png"), fid)],
                                                   f"On this form, is the field \"{name}\" filled in?", ["Yes", "No"],
                                                   "A" if lab == "yes" else "B", "field_present", sv))
    m = min(len(g["A"]), len(g["B"]))
    for v in g.values():
        out += sorted(v, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())[:m]
    return out


# ---------------------------------------------------------------- Hateful Memes train (kit rule)
def b_hm_train():
    sv = Saver("hm_train")
    out = []
    for r in (json.loads(l) for l in open(DL / "hm" / "train.jsonl")):
        p = DL / "hm" / r["img"]
        if not p.exists():
            continue
        rel = sv.save(p.read_bytes(), str(r["id"]))
        out.append(item("hm_train", r["id"], [rel], "Is this meme hateful?", ["Yes", "No"], "A" if r["label"] == 1 else "B", "binary", sv))
    return out


# ---------------------------------------------------------------- V1 pools (structured items; images from v1_imgs.tar)
def v1_image_index():
    idx = {}
    for root in (D / "v1img", D / "memeimg"):
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file():
                idx.setdefault(p.name, p)
    return idx


def b_v1(fam, pool_file, convert):
    sv = Saver(fam)
    idx = v1_image_index()
    out, miss = [], 0
    for r in (json.loads(l) for l in open(D / "v1" / "pools" / pool_file)):
        res = convert(r)
        if res is None:
            continue
        img_name, q, opts, gold, sub = res
        p = idx.get(img_name)
        if p is None:
            miss += 1
            continue
        rel = sv.save(p.read_bytes(), Path(img_name).stem)
        out.append(item(fam, r["id"].split("|", 1)[-1], [rel], q, opts, gold, sub, sv))
    print(json.dumps({"family": fam, "missing_images": miss}), flush=True)
    return out


def v1_choice(r):
    q = r["questions"]["q1"]
    keys = list(q["criteria"])
    lab = r["labels"]["q1"]
    gk = max(keys, key=lambda k: lab[k])
    opts = [q["criteria"][k] for k in keys]
    state = r["state"].replace("<|image|>", "").strip()
    return keys, opts, gk, state, q["instructions"]


def conv_coco(r):
    keys, opts, gk, state, ins = v1_choice(r)
    return Path(r["image"]["path"]).name, state, opts, LETTERS[keys.index(gk)], "count_spatial"


def conv_sat(r):
    keys, opts, gk, state, ins = v1_choice(r)
    gold_opt = opts[keys.index(gk)]
    opts = opts[:]
    random.Random(f"{SEED}:sat:{r['id']}").shuffle(opts)  # the raw pool lists the correct option first (V1 shuffled at render)
    return Path(r["image"]["path"]).name, state, opts, LETTERS[opts.index(gold_opt)], "sat_spatial"


def conv_vrd(r):
    keys, opts, gk, state, ins = v1_choice(r)
    return Path(r["image"]["path"]).name, f"{state} {ins}".strip(), opts, LETTERS[keys.index(gk)], "relative_depth"


def conv_vsr(r):
    keys, opts, gk, state, ins = v1_choice(r)
    q = f"Is the following statement about the image true? \"{state}\""
    return Path(r["image"]["path"]).name, q, ["Yes", "No"], "A" if gk == "yes" else "B", "spatial_relation"


def b_meme_v1():
    sv = Saver("meme_v1")
    idx = v1_image_index()
    out, miss = [], 0
    for r in (json.loads(l) for l in open(D / "v1" / "pools" / "meme_items_train.jsonl")):
        p = idx.get(Path(r["images"][0]).name)
        if p is None:
            miss += 1
            continue
        rel = sv.save(p.read_bytes(), Path(r["images"][0]).stem)
        out.append(item("meme_v1", r["id"].split(":", 1)[-1], [rel], r["question"], r["options"], r["gold"], "binary", sv))
    print(json.dumps({"family": "meme_v1", "missing_images": miss}), flush=True)
    return out


# ---------------------------------------------------------------- MindCube train (multi-view, options inline)
def b_mindcube():
    sv = Saver("mindcube")
    zf = zipfile.ZipFile(DL / "mindcube" / "data.zip")
    names = set(zf.namelist())
    out = []
    for line in zf.read("data/raw/MindCube_train.jsonl").decode().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        stem, opts = parse_lettered(r["question"], r" ([A-F])\. ")
        if not opts or r.get("gt_answer") not in LETTERS[:len(opts)] or not 2 <= len(opts) <= 6:
            continue
        rels = []
        for p in r["images"][:4]:
            zp = f"data/{p}"
            if zp not in names:
                rels = None
                break
            rels.append(sv.save(zf.read(zp), re.sub(r"[^A-Za-z0-9_.-]", "_", p)[:90]))
        if not rels:
            continue
        q = stem
        for j in range(1, 5):
            q = q.replace(f"image {j}", f"[image {j}]")
        out.append(item("mindcube", r["id"], rels, q, opts, r["gt_answer"], r.get("type", "multi_view"), sv))
    return out


# ---------------------------------------------------------------- COCO caption order-swap (Winoground text subtask format)
PREPS = (r"(on top of|next to|in front of|behind|under|underneath|above|below|beside|near|on|in|inside|holding|riding|carrying|chasing|"
         r"watching|pulling|pushing|feeding|looking at|hanging from|leaning against|parked next to|walking past|following|facing|"
         r"petting|hugging|throwing|catching|kicking|hitting|eating|drinking|sitting on|standing on|lying on)")
SWAP = re.compile(rf"^(an?|the|two|three|some|several)\s+([a-z]+(?:\s[a-z]+)?)\s+(?:is\s+|are\s+|sitting\s+|standing\s+|lying\s+)?{PREPS}\s+(an?|the|two|three|some|several)\s+([a-z]+(?:\s[a-z]+)?)\s*\.?$")


def b_coco_wino():
    sv = Saver("coco_wino")
    zf = zipfile.ZipFile(DL / "annotations_trainval2017.zip")
    caps = json.loads(zf.read("annotations/captions_train2017.json"))
    fname = {im["id"]: im["file_name"] for im in caps["images"]}
    idx = v1_image_index()
    rng = random.Random(SEED + 1301)
    by = {}
    for a_ in caps["annotations"]:
        fn = fname[a_["image_id"]]
        cands = [n for n in (fn, f"train2017_{fn}") if n in idx]
        if not cands:
            continue
        c = a_["caption"].strip().lower().rstrip(".")
        m = SWAP.match(c)
        if not m or m.group(2) == m.group(5):
            continue
        sw = m.string[:m.start(1)] + f"{m.group(4)} {m.group(5)} {m.group(3)} {m.group(1)} {m.group(2)}"
        orig = f"{m.group(1)} {m.group(2)} {m.group(3)} {m.group(4)} {m.group(5)}"
        by.setdefault(cands[0], []).append((orig, sw))
    out = []
    for name, pairs in sorted(by.items()):
        pairs = list(dict.fromkeys(pairs))
        rng.shuffle(pairs)
        rel = sv.save(idx[name].read_bytes(), Path(name).stem)
        for j, (orig, sw) in enumerate(pairs[:3]):
            order = [0, 1]
            rng.shuffle(order)
            opts = [[orig, sw][k] for k in order]
            out.append(item("coco_wino", f"{Path(name).stem}:{j}", [rel], "Which caption describes this image?", opts, LETTERS[order.index(0)], "text", sv))
    return out


# ---------------------------------------------------------------- ChartNet grounded_qa -> CharXiv 4-way formats
LABEL_RE = re.compile(r"^The (.+?) is (.+?)(?: @ \[[^\]]*\])?$")


def b_chartnet():
    import pyarrow.parquet as pq
    from decision_index.vision.convert import NUM, numeric_distractors

    sv = Saver("chartnet")
    rng = random.Random(SEED + 1401)
    out = []
    for f in sorted((DL / "chartnet" / "grounded_qa").glob("grounded_qa_*.parquet")):
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            for r in pf.read_row_group(rg, columns=["id", "grounded_image", "template_qa", "reasoning_qa"]).to_pylist():
                labels = {}
                for q in r["template_qa"] or []:
                    if q.get("mode") not in ("WHAT", "BOTH"):
                        continue
                    m = LABEL_RE.match((q.get("ans") or "").strip())
                    if m and m.group(2).strip() and m.group(2).strip().upper() != "NA" and "exist" not in m.group(2):
                        labels[q["q"]] = m.group(2).strip()
                nums = []
                for q in r["reasoning_qa"] or []:
                    nums += [x.replace(",", "") for x in re.findall(r"-?\d[\d,]*(?:\.\d+)?", q.get("reasoning") or "")]
                nums = list(dict.fromkeys(n for n in nums if NUM.match(n)))
                made = []
                for q in r["reasoning_qa"] or []:
                    a = (q.get("answer") or "").strip().replace(",", "")
                    if NUM.match(a):
                        ds = numeric_distractors(a, [n for n in nums if n != a], rng)
                        if len(ds) == 3 and len(set(ds + [a])) == 4:
                            made.append(("number-in-chart", q["question"], a, ds))
                            break
                vals = [v for v in dict.fromkeys(labels.values()) if not NUM.match(v.replace(",", ""))]
                for qq, gold in labels.items():
                    if NUM.match(gold.replace(",", "")):
                        continue
                    pool = [v for v in vals if v.lower() != gold.lower()]
                    if len(pool) >= 3:
                        made.append(("text-in-chart", qq, gold, rng.sample(pool, 3)))
                        break
                if not made:
                    continue
                rel = sv.save(r["grounded_image"]["bytes"], r["id"])
                for sub, q, gold, ds in made:
                    opts = ds + [gold]
                    rng.shuffle(opts)
                    out.append(item("chartnet", f"{r['id']}:{sub}", [rel], q, opts, LETTERS[opts.index(gold)], sub, sv))
    return out


BUILDERS = {
    "m2w": b_m2w, "infovqa_train": b_infovqa_train, "docvqa": b_docvqa, "cord_train": b_cord_train, "funsd_train": b_funsd_train,
    "hm_train": b_hm_train, "meme_v1": b_meme_v1, "chartnet": b_chartnet, "mindcube": b_mindcube, "coco_wino": b_coco_wino,
    "coco_count": lambda: b_v1("coco_count", "coco_count.jsonl", conv_coco),
    "sat": lambda: b_v1("sat", "sat.jsonl", conv_sat),
    "vrd25": lambda: b_v1("vrd25", "vrd25.jsonl", conv_vrd),
    "vsr": lambda: b_v1("vsr", "vsr.jsonl", conv_vsr),
}

if __name__ == "__main__":
    import time

    fam = sys.argv[1]
    t = time.time()
    ITEMS.mkdir(parents=True, exist_ok=True)
    items = BUILDERS[fam]()
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)), f"{fam}: duplicate ids"
    with open(ITEMS / f"{fam}.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(json.dumps({"family": fam, "items": len(items), "gold": dict(Counter(it["gold"] for it in items)),
                      "seconds": round(time.time() - t, 1)}), flush=True)
