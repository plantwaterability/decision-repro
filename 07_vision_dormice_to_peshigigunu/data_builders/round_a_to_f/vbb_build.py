"""Round-B gap data (ctl2). python vbb_build.py <family>; reuses vb_build's saver/item/pHash, writes items_b/ + img_b/.

Formats added this round:
- R-Bench-M style: question + lettered options as text, readout row ("The answer is ("), and the benchmark's
  "Other options are incorrect" option (sometimes the gold, after the true answer is removed);
- MMMU-Pro vision style: question + options rendered INTO the image (options_in_image, the kit's vision_only instruction),
  options padded to 6-10 with same-pool distractors, readout row.
"""
import hashlib
import io
import json
import random
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vb_build as B  # noqa: E402
from decision_index.vision.convert import NUM, SEED, numeric_distractors  # noqa: E402

D = B.D
DLB = Path("/data/shared/visdata_dl_b")
RB_V2 = __import__("os").environ.get("VBB_RB_V2") == "1"  # A3: R-Bench rows only, fixed "Other" construction, separate dirs
B.IMG = D / ("img_b3" if RB_V2 else "img_b")
B.ITEMS = D / ("items_b3" if RB_V2 else "items_b")
READOUT = "The answer is ("
LETTERS = "ABCDEFGHIJKL"
OTHER = "Other options are incorrect"
B.LIC.update({
    "ai2d_rb2": ("R-Bench-M", "nc", "AI2D (ai2d-all.zip license.txt: AI2 License for Non-Commercial Use); A3 'Other' construction"),
    "sqa_rb2": ("R-Bench-M", "nc", "ScienceQA train (CC BY-NC-SA 4.0); A3 'Other' construction"),
    "chartnet_rb2": ("R-Bench-M", "clean", "ChartNet reasoning_qa (CDLA-Permissive-2.0); A3 'Other' construction"),
    "ai2d_rb": ("R-Bench-M", "nc", "AI2D (ai2d-all.zip license.txt: AI2 License for Non-Commercial Use; web pages say CC BY-SA - the shipped licence governs)"),
    "ai2d_mmmu": ("MMMU-Pro vision", "nc", "AI2D (ai2d-all.zip license.txt: AI2 License for Non-Commercial Use)"),
    "sqa_rb": ("R-Bench-M", "nc", "ScienceQA train (CC BY-NC-SA 4.0)"),
    "sqa_mmmu": ("MMMU-Pro vision", "nc", "ScienceQA train (CC BY-NC-SA 4.0)"),
    "chartnet_b": ("CharXiv", "clean", "ibm-granite/ChartNet grounded_qa shards 02-05 (CDLA-Permissive-2.0; synthetic charts)"),
    "chartnet_rb": ("R-Bench-M", "clean", "ChartNet reasoning_qa in R-Bench-M format (CDLA-Permissive-2.0)"),
    "chartnet_mmmu": ("MMMU-Pro vision", "clean", "ChartNet reasoning_qa rendered MMMU-Pro-vision style (CDLA-Permissive-2.0)"),
})
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def wrap(draw, text, font, width):
    lines = []
    for para in str(text).split("\n"):
        cur = ""
        for w in para.split(" "):
            t = (cur + " " + w).strip()
            if draw.textlength(t, font=font) <= width or not cur:
                cur = t
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


def composite(img_bytes, question, options, context=None):
    """MMMU-Pro-vision-like page: figure on top, then the question and lettered options as text, all one image."""
    from PIL import Image, ImageDraw, ImageFont

    fig = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    W = max(900, min(1400, fig.width))
    if fig.width > W:
        fig = fig.resize((W, max(1, int(fig.height * W / fig.width))), Image.LANCZOS)
    fs = 26
    font = ImageFont.truetype(FONT, fs)
    tmp = ImageDraw.Draw(Image.new("RGB", (W, 10)))
    body = []
    if context:
        body += wrap(tmp, f"Context: {context}", font, W - 40) + [""]
    body += wrap(tmp, question, font, W - 40) + [""]
    for i, o in enumerate(options):
        body += wrap(tmp, f"({LETTERS[i]}) {o}", font, W - 40)
    th = (fs + 8) * len(body) + 40
    page = Image.new("RGB", (W, fig.height + th + 20), "white")
    page.paste(fig, ((W - fig.width) // 2, 10))
    d = ImageDraw.Draw(page)
    y = fig.height + 30
    for line in body:
        d.text((20, y), line, fill="black", font=font)
        y += fs + 8
    buf = io.BytesIO()
    page.save(buf, format="PNG")
    return buf.getvalue()


def rbench_options(gold, distractors, rng):
    """R-Bench-M flavour: sometimes add 'Other options are incorrect'; sometimes it is the gold (true answer removed)."""
    opts = distractors[:] + [gold]
    rng.shuffle(opts)
    g = gold
    r = rng.random()
    if r < 0.30:
        if r < 0.10 and len(distractors) >= 2:
            opts = [o for o in opts if o != gold]
            g = OTHER
        opts.append(OTHER)
    return opts, LETTERS[opts.index(g)]


def rbench_options_v2(gold, distractors, pool, rng):
    """A3 construction (R-Bench paper 2505.02018 sec. 2.4: every question carries the 'other answers are incorrect' option; no
    removal rate is documented -> neutral 1/n, n = options shown incl. OTHER). OTHER is in every row at a random position and is
    the gold ONLY when the true answer was removed (replaced by a same-pool distractor so n stays fixed). Same rule for every source."""
    opts = distractors[:] + [gold]
    n = len(opts) + 1
    g = gold
    if rng.random() < 1.0 / n:
        repl = [p for p in dict.fromkeys(pool) if p.lower() != gold.lower() and p not in distractors and p != OTHER]
        if repl:
            opts = distractors[:] + [rng.choice(repl)]
            g = OTHER
    rng.shuffle(opts)
    opts.insert(rng.randrange(len(opts) + 1), OTHER)
    return opts, LETTERS[opts.index(g)]


def mmmu_options(gold, distractors, pad_pool, rng):
    k = rng.randint(6, 10)
    pool = [p for p in dict.fromkeys(pad_pool) if p.lower() != gold.lower() and p not in distractors]
    extra = rng.sample(pool, min(max(0, k - 1 - len(distractors)), len(pool)))
    opts = distractors[:] + extra + [gold]
    rng.shuffle(opts)
    return opts, LETTERS[opts.index(gold)]


def emit_pair(fam_rb, fam_mm, sv_rb, sv_mm, sid, img_bytes, stem, question, gold, distractors, pad_pool, rng, context=None):
    """Each source question goes to exactly one of the two styles (hash split), never both."""
    if RB_V2 and int(hashlib.sha256(sid.encode()).hexdigest(), 16) % 2 != 0:
        return None  # v2 rebuilds only the R-Bench half (the MMMU half keeps its round-B rows; a question is never in both)
    if int(hashlib.sha256(sid.encode()).hexdigest(), 16) % 2 == 0:
        rel = sv_rb.save(img_bytes, stem)
        q = f"{context}\n{question}" if context else question
        opts, g = rbench_options_v2(gold, distractors, pad_pool, rng) if RB_V2 else rbench_options(gold, distractors, rng)
        sub = "other_correct" if opts[LETTERS.index(g)] == OTHER else "content_correct"
        return B.item(fam_rb, sid, [rel], q, opts, g, sub, sv_rb, prompt={"options_in_image": False, "readout": READOUT})
    opts, g = mmmu_options(gold, distractors, pad_pool, rng)
    page = composite(img_bytes, question, opts, context)
    rel = sv_mm.save(page, stem + "_page")
    k = len(opts)
    return B.item(fam_mm, sid, [rel], f"Answer the multiple-choice question shown in the image (options A-{LETTERS[k - 1]}).",
                  list(LETTERS[:k]), g, "vision_only", sv_mm, prompt={"options_in_image": True, "readout": READOUT})


def b_ai2d():
    z = zipfile.ZipFile(DLB / "ai2d-all.zip")
    names = set(z.namelist())
    cats = json.loads(z.read("ai2d/categories.json"))
    rng = random.Random(SEED + 2001)
    qfiles = sorted(n for n in names if n.startswith("ai2d/questions/") and n.endswith(".json"))
    pools, recs = {}, []
    for qf in qfiles:
        j = json.loads(z.read(qf))
        img = j["imageName"]
        for qt, q in j["questions"].items():
            at = [str(a).strip() for a in q["answerTexts"]]
            if len(at) < 2 or not (0 <= q["correctAnswer"] < len(at)) or len(set(a.lower() for a in at)) != len(at):
                continue
            cat = cats.get(img, "other")
            pools.setdefault(cat, []).extend(at)
            recs.append((q["questionId"], img, qt, at, q["correctAnswer"], cat))
    sv_rb, sv_mm = B.Saver("ai2d_rb2" if RB_V2 else "ai2d_rb"), B.Saver("ai2d_mmmu")
    out = []
    for qid, img, qt, at, ci, cat in recs:
        p = f"ai2d/images/{img}"
        if p not in names:
            continue
        gold = at[ci]
        ds = [a for i, a in enumerate(at) if i != ci]
        it = emit_pair("ai2d_rb2" if RB_V2 else "ai2d_rb", "ai2d_mmmu", sv_rb, sv_mm, qid, z.read(p), re.sub(r"[^A-Za-z0-9_.-]", "_", qid), qt, gold, ds, pools[cat], rng)
        out.append(it)
    return out


def b_sqa():
    import pyarrow.parquet as pq

    f = sorted(DLB.glob("scienceqa/**/train-*.parquet"))[0]
    rng = random.Random(SEED + 2002)
    rows = [r for r in pq.read_table(f).to_pylist() if r.get("image") and (r["image"] or {}).get("bytes")]
    pools = {}
    for r in rows:
        pools.setdefault(r.get("subject") or "other", []).extend(str(c) for c in r["choices"])
    sv_rb, sv_mm = B.Saver("sqa_rb2" if RB_V2 else "sqa_rb"), B.Saver("sqa_mmmu")
    out = []
    for i, r in enumerate(rows):
        ch = [str(c).strip() for c in r["choices"]]
        if len(ch) < 2 or len(set(c.lower() for c in ch)) != len(ch):
            continue
        gold = ch[int(r["answer"])]
        ds = [c for j, c in enumerate(ch) if j != int(r["answer"])]
        ctx = (r.get("hint") or "").strip() or None
        it = emit_pair("sqa_rb2" if RB_V2 else "sqa_rb", "sqa_mmmu", sv_rb, sv_mm, f"train{i}", r["image"]["bytes"], f"sqa_train{i}", r["question"], gold, ds,
                       pools[r.get("subject") or "other"], rng, context=ctx)
        out.append(it)
    return out


def b_chartnet_b():
    """Shards 02-05: CharXiv 4-way formats (as round A) plus reasoning items in R-Bench / MMMU-Pro-vision formats."""
    import pyarrow.parquet as pq

    sv = B.Saver("chartnet_b")
    sv_rb, sv_mm = B.Saver("chartnet_rb2" if RB_V2 else "chartnet_rb"), B.Saver("chartnet_mmmu")
    rng = random.Random(SEED + 2003)
    out = []
    seen = set()
    for line in open(D / "items" / "chartnet.jsonl"):  # round A's charts never come back in round B
        seen.add(json.loads(line)["id"].split(":")[1])
    for f in sorted((DLB / "chartnet" / "grounded_qa").glob("grounded_qa_*.parquet")):
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            for r in pf.read_row_group(rg, columns=["id", "grounded_image", "template_qa", "reasoning_qa"]).to_pylist():
                if r["id"] in seen:  # chart ids repeat across shards
                    continue
                seen.add(r["id"])
                labels = {}
                for q in r["template_qa"] or []:
                    if q.get("mode") not in ("WHAT", "BOTH"):
                        continue
                    m = B.LABEL_RE.match((q.get("ans") or "").strip())
                    if m and m.group(2).strip() and m.group(2).strip().upper() != "NA" and "exist" not in m.group(2):
                        labels[q["q"]] = m.group(2).strip()
                nums = []
                for q in r["reasoning_qa"] or []:
                    nums += [x.replace(",", "") for x in re.findall(r"-?\d[\d,]*(?:\.\d+)?", q.get("reasoning") or "")]
                nums = list(dict.fromkeys(n for n in nums if NUM.match(n)))
                numq = []
                for q in r["reasoning_qa"] or []:
                    a = (q.get("answer") or "").strip().replace(",", "")
                    if NUM.match(a):
                        ds = numeric_distractors(a, [n for n in nums if n != a], rng)
                        if len(ds) == 3 and len(set(ds + [a])) == 4:
                            numq.append((q["question"], a, ds))
                img = r["grounded_image"]["bytes"]
                h = int(hashlib.sha256(r["id"].encode()).hexdigest(), 16) % 3
                if h == 0 and numq:  # reasoning question in R-Bench / MMMU-Pro-vision format
                    q, a, ds = numq[0]
                    out.append(emit_pair("chartnet_rb2" if RB_V2 else "chartnet_rb", "chartnet_mmmu", sv_rb, sv_mm, f"{r['id']}:r", img, r["id"], q, a, ds, nums, rng))
                    continue
                if RB_V2:
                    continue  # v2: R-Bench rows only
                made = []
                if numq:
                    made.append(("number-in-chart",) + numq[0])
                vals = [v for v in dict.fromkeys(labels.values()) if not NUM.match(v.replace(",", ""))]
                for qq, gold in labels.items():
                    pool = [v for v in vals if v.lower() != gold.lower()]
                    if not NUM.match(gold.replace(",", "")) and len(pool) >= 3:
                        made.append(("text-in-chart", qq, gold, rng.sample(pool, 3)))
                        break
                if not made:
                    continue
                rel = sv.save(img, r["id"])
                for sub, q, gold, ds in made:
                    opts = ds + [gold]
                    rng.shuffle(opts)
                    out.append(B.item("chartnet_b", f"{r['id']}:{sub}", [rel], q, opts, LETTERS[opts.index(gold)], sub, sv))
    return out


B.LIC.update({
    "vrd_b": ("BLINK", "clean", "2.5VRD within-image train annotations (CC BY 4.0); Open Images V4 train images (per-image terms, the trawl's CLEAN line)"),
    "coco_b": ("CV-Bench", "clean", "Nemotron-Image-Training-v3.5 coco_count_spatial_qa (OpenMDW-1.1); COCO train2017 images (V1 precedent)"),
})


def fetch_many(urls_paths, workers=16):
    import httpx
    from concurrent.futures import ThreadPoolExecutor

    def one(up):
        u, p = up
        if p.exists() and p.stat().st_size > 0:
            return True
        try:
            r = httpx.get(u, timeout=60, follow_redirects=True)
            if r.status_code == 200 and r.content:
                p.write_bytes(r.content)
                return True
        except Exception:
            pass
        return False

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, urls_paths))


def b_vrd_b(n_new=8000):
    import csv
    from decision_index.vision.convert import draw_marks

    objs = {}
    for r in csv.DictReader(open(DLB / "vrd_within_image_objects_train.csv")):
        objs[(r["image_id"], r["object_id"])] = (r["entity"], [float(r[k]) for k in ("xmin", "ymin", "xmax", "ymax")])
    pairs = list(csv.DictReader(open(DLB / "vrd_within_image_vrd_train.csv")))
    # infer the distance-code -> closer-object mapping from V1's 5,747 labelled pairs (gold known), require >= 99% consistency
    v1 = {}
    for l in open(D / "v1" / "pools" / "vrd25.jsonl"):
        r = json.loads(l)
        img = r["id"].split("|")[1]
        lab = r["labels"]["q1"]
        v1[img] = max(lab, key=lab.get)  # "A" = object of the first-named option
    agree = Counter()
    for p in pairs:
        if p["image_id_1"] in v1 and p["distance"] in ("1", "2"):
            agree[(p["distance"], v1[p["image_id_1"]])] += 1
    m1 = "A" if agree[("1", "A")] >= agree[("1", "B")] else "B"
    tot = sum(agree.values())
    cons = (agree[("1", m1)] + agree[("2", "B" if m1 == "A" else "A")]) / max(1, tot)
    print(json.dumps({"vrd_mapping_votes": {f"{k[0]}->{k[1]}": v for k, v in agree.items()}, "consistency": round(cons, 4)}), flush=True)
    if cons < 0.99:
        raise SystemExit("2.5VRD distance-code mapping not consistent with V1 labels; refusing to build")
    first_closer = {"1": m1 == "A", "2": m1 != "A"}  # True -> object_1 is closer
    cand = [p for p in pairs if p["distance"] in ("1", "2") and p["image_id_1"] not in v1 and p["image_id_1"] == p["image_id_2"]
            and (p["image_id_1"], p["object_id_1"]) in objs and (p["image_id_2"], p["object_id_2"]) in objs]
    cand.sort(key=lambda p: hashlib.sha256((p["image_id_1"] + p["object_id_1"] + p["object_id_2"]).encode()).hexdigest())
    seen, pick = set(), []
    for p in cand:
        if p["image_id_1"] in seen:
            continue
        seen.add(p["image_id_1"])
        pick.append(p)
        if len(pick) >= int(n_new * 1.15):
            break
    raw = DLB / "oid_train"
    raw.mkdir(exist_ok=True)
    ok = fetch_many([(f"https://open-images-dataset.s3.amazonaws.com/train/{p['image_id_1']}.jpg", raw / f"{p['image_id_1']}.jpg") for p in pick])
    print(json.dumps({"oid_fetch_ok": sum(ok), "of": len(ok)}), flush=True)
    from PIL import Image

    sv = B.Saver("vrd_b")
    rng = random.Random(SEED + 2004)
    out = []
    for p, good in zip(pick, ok):
        if not good or len(out) >= n_new:
            continue
        try:
            im = Image.open(raw / f"{p['image_id_1']}.jpg").convert("RGB")
        except Exception:
            continue
        W, H = im.size
        b1 = objs[(p["image_id_1"], p["object_id_1"])][1]
        b2 = objs[(p["image_id_2"], p["object_id_2"])][1]
        px = [[b[0] * W, b[1] * H, b[2] * W, b[3] * H] for b in (b1, b2)]
        order = [0, 1]
        rng.shuffle(order)  # which object gets mark A
        marked = draw_marks(im, [px[i] for i in order], B.os.environ.get("DECISION_INDEX_FONT"))
        buf = io.BytesIO()
        marked.save(buf, format="JPEG", quality=92)
        rel = sv.save(buf.getvalue(), p["image_id_1"])
        closer = 0 if first_closer[p["distance"]] else 1
        gold = "AB"[order.index(closer)]
        out.append(B.item("vrd_b", f"{p['image_id_1']}:{p['object_id_1']}:{p['object_id_2']}", [rel],
                          "Two objects are marked with boxes A and B. Which marked object is closer to the camera?",
                          ["the object in box A", "the object in box B"], gold, "relative_depth", sv))
    return out


def b_coco_b():
    from decision_index.vision.build import parse_lettered

    used = set()
    for l in open(D / "v1" / "pools" / "coco_count.jsonl"):
        used.add(json.loads(l)["group_id"])
    recs = []
    for l in open(DLB / "nemotron" / "coco_count_spatial_qa" / "coco_count_spatial_qa.jsonl"):
        r = json.loads(l)
        u = r["messages"][0]["content"]
        img = next((c["image"] for c in u if isinstance(c, dict) and c.get("type") == "image"), None)
        txt = "".join(c for c in u if isinstance(c, str)).strip()
        ans = "".join(c for c in r["messages"][1]["content"] if isinstance(c, str))
        m = re.search(r"\(([A-F])\)", ans)
        if not img or img in used or not m or "val2017" in img:
            continue
        body = txt.split("Select from the following choices.")[0].strip()
        rest = txt.split("Select from the following choices.")[-1]
        stem, opts = parse_lettered("X\n" + rest.strip(), r"\n?\(([A-F])\) ")
        if not opts or m.group(1) not in LETTERS[:len(opts)]:
            continue
        recs.append((r["id"], img, body, opts, m.group(1)))
    raw = DLB / "coco_train2017"
    raw.mkdir(exist_ok=True)
    imgs = sorted({r[1] for r in recs})
    ok = dict(zip(imgs, fetch_many([(f"http://images.cocodataset.org/{im}", raw / im.split("/")[-1]) for im in imgs])))
    print(json.dumps({"coco_records": len(recs), "coco_images": len(imgs), "fetched": sum(ok.values())}), flush=True)
    sv = B.Saver("coco_b")
    out = []
    for rid, img, body, opts, g in recs:
        if not ok.get(img):
            continue
        rel = sv.save((raw / img.split("/")[-1]).read_bytes(), img.split("/")[-1].rsplit(".", 1)[0])
        sub = "count" if body.lower().startswith("count") or "how many" in body.lower() else "relation"
        out.append(B.item("coco_b", rid, [rel], body, opts, g, sub, sv))
    return out


BUILDERS = {"ai2d": b_ai2d, "sqa": b_sqa, "chartnet_b": b_chartnet_b, "vrd_b": b_vrd_b, "coco_b": b_coco_b}

if __name__ == "__main__":
    import time

    fam = sys.argv[1]
    t = time.time()
    B.ITEMS.mkdir(parents=True, exist_ok=True)
    items = [it for it in BUILDERS[fam]() if it is not None]
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)), f"{fam}: duplicate ids"
    with open(B.ITEMS / f"{fam}.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(json.dumps({"family": fam, "items": len(items), "per_source": dict(Counter(it["source"] for it in items)),
                      "gold": dict(Counter(it["gold"] for it in items)), "seconds": round(time.time() - t, 1)}), flush=True)
