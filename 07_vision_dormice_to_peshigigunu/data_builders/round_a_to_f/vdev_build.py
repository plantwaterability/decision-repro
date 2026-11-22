"""NON-public vision dev set (~400 rows per board family) from held-out / validation splits of our training sources (never
their train rows, never public board rows), board-format through the kit's converters, deduped (sha256 + pHash <= 6) against
the public images AND every training pool (rounds A, B, B3, C, D). usage: python vdev_build.py <download|build|rows>"""
import hashlib
import io
import json
import random
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/data/local/visdata/scripts")
import vbb_build as VB  # noqa: E402  (import first: it repoints vb_build's dirs, which we override right below)
import vb_build as B  # noqa: E402
from decision_index.vision.build import m2w_convert  # noqa: E402
from decision_index.vision.convert import NUM, SEED, norm, numeric_distractors, draw_marks  # noqa: E402

DV = Path("/data/shared/visdata_dev")
DL = DV / "dl"
B.IMG = DV / "img_dev"
B.ITEMS = DV / "items_dev"
LET = "ABCDEFGHIJ"
LIC_EVAL = "eval-only dev set (NC fine in evaluation)"
B.LIC.update({k: (fam, "eval", f"{src}; {LIC_EVAL}") for k, fam, src in [
    ("dev_m2w", "Mind2Web", "Multimodal-Mind2Web test_domain (board uses test_website)"),
    ("dev_chartnet", "CharXiv", "ChartNet grounded_qa_06 (unused shard; CDLA-Permissive-2.0)"),
    ("dev_doc_info", "InfographicVQA", "DocVQA validation (pixparse, MIT) - InfoVQA proxy"),
    ("dev_doc_kie", "KIE (CORD+FUNSD)", "DocVQA validation (pixparse, MIT), disjoint docs - KIE proxy"),
    ("dev_pixmo_cnt", "CV-Bench", "PixMo-count validation (ODC-BY; web images)"),
    ("dev_vqa_cnt", "CV-Bench", "VQAv2 val 'how many' (CC BY 4.0; COCO val2014)"),
    ("dev_vrd", "BLINK", "2.5VRD validation pairs (CC BY 4.0; Open Images validation)"),
    ("dev_mindcube", "BLINK", "MindCube rows not in MindCube_train (MIT)"),
    ("dev_vqa_rw", "RealWorldQA", "VQAv2 val (CC BY 4.0; COCO val2014) - weak RWQA proxy"),
    ("dev_hm", "Moderation (Hateful Memes)", "Hateful Memes dev_unseen + test_unseen (board uses the seen splits; research licence)"),
    ("dev_sqa_rb", "R-Bench-M", "ScienceQA validation (CC BY-NC-SA 4.0), A3 'Other' construction"),
    ("dev_sqa_mmmu", "MMMU-Pro vision", "ScienceQA test (CC BY-NC-SA 4.0), MMMU-Pro vision format"),
    ("dev_chartnet_mmmu", "MMMU-Pro vision", "ChartNet grounded_qa_06 reasoning, MMMU-Pro vision format"),
]})


def get(url, path):
    import httpx

    if path.exists() and path.stat().st_size > 0:
        return path
    with httpx.stream("GET", url, timeout=900, follow_redirects=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for c in r.iter_bytes(1 << 20):
                f.write(c)
    return path


def download():
    from huggingface_hub import snapshot_download, list_repo_files

    DL.mkdir(parents=True, exist_ok=True)
    m2w = sorted(f for f in list_repo_files("osunlp/Multimodal-Mind2Web", repo_type="dataset") if "test_domain" in f)[:2]
    snapshot_download("osunlp/Multimodal-Mind2Web", repo_type="dataset", local_dir=DL / "m2w", allow_patterns=m2w, token=False)
    snapshot_download("ibm-granite/ChartNet", repo_type="dataset", local_dir=DL / "chartnet", allow_patterns=["grounded_qa/grounded_qa_06.parquet"], token=False)
    snapshot_download("allenai/pixmo-count", repo_type="dataset", local_dir=DL / "pixmo", allow_patterns=["data/validation-*"], token=False)
    fs = list_repo_files("derek-thomas/ScienceQA", repo_type="dataset")
    snapshot_download("derek-thomas/ScienceQA", repo_type="dataset", local_dir=DL / "sqa", allow_patterns=[f for f in fs if ("validation" in f or "test" in f) and f.endswith(".parquet")], token=False)
    fs = list_repo_files("pixparse/docvqa-single-page-questions", repo_type="dataset")
    snapshot_download("pixparse/docvqa-single-page-questions", repo_type="dataset", local_dir=DL / "docvqa",
                      allow_patterns=[f for f in fs if "validation" in f and f.endswith(".parquet")], token=False)
    for u in ["https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip", "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
              "https://raw.githubusercontent.com/google-research-datasets/2.5vrd/main/data/within_image_objects_validation.csv",
              "https://raw.githubusercontent.com/google-research-datasets/2.5vrd/main/data/within_image_vrd_validation.csv"]:
        get(u, DL / u.rsplit("/", 1)[-1])
    print("DOWNLOAD_OK", flush=True)


def fetch(pairs):
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

    with ThreadPoolExecutor(16) as ex:
        return list(ex.map(one, pairs))


def it(fam, sid, rel, q, opts, gold, sub, sv, prompt=None):
    return B.item(fam, sid, rel if isinstance(rel, list) else [rel], q, opts, gold, sub, sv, prompt=prompt)


def b_m2w(n=480):
    import pyarrow.parquet as pq

    sv = B.Saver("dev_m2w")

    class Ctx:
        img = sv.dir
        root = B.IMG
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    out, off = [], 0
    for p in sorted((DL / "m2w" / "data").glob("test_domain-*.parquet")):
        pf = pq.ParquetFile(p)
        for rg in range(pf.num_row_groups):
            for row in pf.read_row_group(rg, columns=["action_uid", "operation", "pos_candidates", "neg_candidates", "domain", "annotation_id",
                                                      "confirmed_task", "screenshot", "action_reprs", "target_action_index"]).to_pylist():
                row["_off"] = off
                off += 1
                x = m2w_convert(Ctx, row, row["action_uid"], random.Random(f"{SEED}:dev_m2w:{row['action_uid']}"))
                if not x:
                    continue
                rel = x["images"][0]
                data = (B.IMG / rel).read_bytes()
                from PIL import Image
                sv.meta[rel] = {"path": rel, "src_sha256": hashlib.sha256(data).hexdigest(), "sha256": hashlib.sha256(data).hexdigest(),
                                "phash": f"{B.phash(Image.open(io.BytesIO(data))):016x}"}
                out.append(it("dev_m2w", row["action_uid"], rel, x["question"], x["options"], x["gold"], x["subtask"], sv))
                if len(out) >= n:
                    return out
    return out


def b_chartnet():
    """ChartNet shard 06 -> CharXiv 4-way (dev_chartnet) + reasoning in MMMU-Pro-vision style (dev_chartnet_mmmu); disjoint charts."""
    import pyarrow.parquet as pq

    sv, sv_mm = B.Saver("dev_chartnet"), B.Saver("dev_chartnet_mmmu")
    rng = random.Random(SEED + 5001)
    out = {"cx": [], "mm": []}
    pf = pq.ParquetFile(DL / "chartnet" / "grounded_qa" / "grounded_qa_06.parquet")
    for rg in range(pf.num_row_groups):
        for r in pf.read_row_group(rg, columns=["id", "grounded_image", "template_qa", "reasoning_qa"]).to_pylist():
            if len(out["cx"]) >= 480 and len(out["mm"]) >= 240:
                return out["cx"], out["mm"]
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
            if not numq:
                continue
            q, a, ds = numq[0]
            img = r["grounded_image"]["bytes"]
            if int(hashlib.sha256(r["id"].encode()).hexdigest(), 16) % 3 == 0 and len(out["mm"]) < 240:
                opts, g = VB.mmmu_options(a, ds, nums, rng)
                rel = sv_mm.save(VB.composite(img, q, opts), r["id"] + "_page")
                k = len(opts)
                out["mm"].append(it("dev_chartnet_mmmu", r["id"], rel, f"Answer the multiple-choice question shown in the image (options A-{LET[k - 1]}).",
                                    list(LET[:k]), g, "vision_only", sv_mm, prompt={"options_in_image": True, "readout": "The answer is ("}))
            elif len(out["cx"]) < 480:
                opts = ds + [a]
                rng.shuffle(opts)
                rel = sv.save(img, r["id"])
                out["cx"].append(it("dev_chartnet", r["id"], rel, q, opts, LET[opts.index(a)], "number-in-chart", sv))
    return out["cx"], out["mm"]


def b_docvqa():
    import pyarrow.parquet as pq

    files = sorted((DL / "docvqa").rglob("validation-*.parquet"))
    metas, where = [], {}
    for fi, f in enumerate(files):
        cols = pq.ParquetFile(f).schema_arrow.names
        qid_c = "question_id" if "question_id" in cols else "questionId"
        for ri, r in enumerate(pq.read_table(f, columns=[qid_c, "question", "answers", "other_metadata"] if "other_metadata" in cols else [qid_c, "question", "answers"]).to_pylist()):
            key = ((r.get("other_metadata") or {}).get("image") if r.get("other_metadata") else None) or f"doc_{fi}_{ri}"
            metas.append({"qid": str(r[qid_c]), "question": r["question"], "answers": list(r["answers"] or []), "img_key": key})
            where.setdefault(key, (fi, ri))
    docs = sorted({m["img_key"] for m in metas})
    random.Random(SEED + 5002).shuffle(docs)
    half = set(docs[: len(docs) // 2])
    tables = {}

    def ib(m):
        fi, ri = where[m["img_key"]]
        if fi not in tables:
            tables.clear()
            tables[fi] = pq.read_table(files[fi], columns=["image"])
        return tables[fi].slice(ri, 1).to_pylist()[0]["image"]["bytes"]

    info = B.infovqa_like("dev_doc_info", [m for m in metas if m["img_key"] in half], ib, per_image=1, seed_off=5003, order_key=lambda m: where[m["img_key"]])
    kie = B.infovqa_like("dev_doc_kie", [m for m in metas if m["img_key"] not in half], ib, per_image=1, seed_off=5004, order_key=lambda m: where[m["img_key"]])
    rng = random.Random(SEED + 5005)
    rng.shuffle(info)
    rng.shuffle(kie)
    return info[:480], kie[:480]


def b_counting():
    import pyarrow.parquet as pq

    rng = random.Random(SEED + 5006)
    rows = pq.read_table(sorted((DL / "pixmo").rglob("validation-*.parquet"))[0]).to_pylist()
    raw = DL / "pixmo_img"
    raw.mkdir(exist_ok=True)
    ok = fetch([(r["image_url"], raw / f"{i}.jpg") for i, r in enumerate(rows)])
    sv = B.Saver("dev_pixmo_cnt")
    out = []
    for i, (r, good) in enumerate(zip(rows, ok)):
        if not good:
            continue
        try:
            data = (raw / f"{i}.jpg").read_bytes()
            if r.get("image_sha256") and hashlib.sha256(data).hexdigest() != r["image_sha256"]:
                continue
            rel = sv.save(data, f"pixmo_{i}")
        except Exception:
            continue
        g = str(int(r["count"]))
        ds = numeric_distractors(g, [], rng)
        opts = ds[:3] + [g]
        rng.shuffle(opts)
        out.append(it("dev_pixmo_cnt", str(i), rel, f"How many {r['label']} are in the image?", opts, LET[opts.index(g)], "count", sv))
    return out[:240]


def vqa_val():
    qs = json.loads(zipfile.ZipFile(DL / "v2_Questions_Val_mscoco.zip").read("v2_OpenEnded_mscoco_val2014_questions.json"))["questions"]
    an = json.loads(zipfile.ZipFile(DL / "v2_Annotations_Val_mscoco.zip").read("v2_mscoco_val2014_annotations.json"))["annotations"]
    qmap = {q["question_id"]: q for q in qs}
    return [(a, qmap[a["question_id"]]) for a in an if sum(1 for x in a["answers"] if x["answer"] == a["multiple_choice_answer"]) >= 8]


def b_vqa(recs):
    import sys as _s
    _s.path.insert(0, "/data/local/visdata/scripts")
    from vbd_build import mc, prefix
    B.IMG, B.ITEMS = DV / "img_dev", DV / "items_dev"  # vbd_build's import repoints vb_build's dirs; restore the dev ones

    rng = random.Random(SEED + 5007)
    rng.shuffle(recs)
    pools = {}
    for a, q in recs:
        pools.setdefault(prefix(q["question"]), []).append(a["multiple_choice_answer"])
    cnt, rw, used = [], [], set()
    for a, q in recs:
        if a["image_id"] in used:
            continue
        is_cnt = a["answer_type"] == "number" and q["question"].lower().startswith("how many")
        if is_cnt and len(cnt) < 300:
            cnt.append((a, q)); used.add(a["image_id"])
        elif not is_cnt and len(rw) < 600:
            rw.append((a, q)); used.add(a["image_id"])
        if len(cnt) >= 300 and len(rw) >= 600:
            break
    raw = DL / "coco_val2014"
    raw.mkdir(exist_ok=True)
    ims = sorted(used)
    ok = dict(zip(ims, fetch([(f"http://images.cocodataset.org/val2014/COCO_val2014_{i:012d}.jpg", raw / f"{i:012d}.jpg") for i in ims])))
    out = {}
    for fam, lst in (("dev_vqa_cnt", cnt), ("dev_vqa_rw", rw)):
        sv = B.Saver(fam)
        res = []
        for a, q in lst:
            if not ok.get(a["image_id"]):
                continue
            opts, g = mc(q["question"], a["multiple_choice_answer"], list(dict.fromkeys(pools[prefix(q["question"])]))[:400], rng)
            if not opts:
                continue
            rel = sv.save((raw / f"{a['image_id']:012d}.jpg").read_bytes(), f"{a['image_id']:012d}")
            res.append(it(fam, str(q["question_id"]), rel, q["question"], opts, g, a["answer_type"], sv))
        out[fam] = res
    return out["dev_vqa_cnt"][:240], out["dev_vqa_rw"][:480]


def b_vrd(n=240):
    import csv
    from PIL import Image

    objs = {(r["image_id"], r["object_id"]): [float(r[k]) for k in ("xmin", "ymin", "xmax", "ymax")]
            for r in csv.DictReader(open(DL / "within_image_objects_validation.csv"))}
    pairs = [p for p in csv.DictReader(open(DL / "within_image_vrd_validation.csv")) if p["distance"] in ("1", "2")
             and (p["image_id_1"], p["object_id_1"]) in objs and (p["image_id_2"], p["object_id_2"]) in objs and p["image_id_1"] == p["image_id_2"]]
    rng = random.Random(SEED + 5008)
    rng.shuffle(pairs)
    seen, pick = set(), []
    for p in pairs:
        if p["image_id_1"] not in seen:
            seen.add(p["image_id_1"]); pick.append(p)
        if len(pick) >= int(n * 1.3):
            break
    raw = DL / "oid_val"
    raw.mkdir(exist_ok=True)
    ok = fetch([(f"https://open-images-dataset.s3.amazonaws.com/validation/{p['image_id_1']}.jpg", raw / f"{p['image_id_1']}.jpg") for p in pick])
    sv = B.Saver("dev_vrd")
    out = []
    for p, good in zip(pick, ok):
        if not good or len(out) >= n:
            continue
        im = Image.open(raw / f"{p['image_id_1']}.jpg").convert("RGB")
        W, H = im.size
        px = [[b[0] * W, b[1] * H, b[2] * W, b[3] * H] for b in (objs[(p["image_id_1"], p["object_id_1"])], objs[(p["image_id_2"], p["object_id_2"])])]
        order = [0, 1]
        rng.shuffle(order)
        buf = io.BytesIO()
        draw_marks(im, [px[i] for i in order], "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf").save(buf, format="JPEG", quality=92)
        rel = sv.save(buf.getvalue(), p["image_id_1"])
        closer = 0 if p["distance"] == "1" else 1  # mapping verified on V1 labels (1 -> first object closer)
        out.append(it("dev_vrd", f"{p['image_id_1']}:{p['object_id_1']}:{p['object_id_2']}", rel,
                      "Two objects are marked with boxes A and B. Which marked object is closer to the camera?",
                      ["the object in box A", "the object in box B"], "AB"[order.index(closer)], "relative_depth", sv))
    return out


def b_mindcube(n=240):
    from decision_index.vision.build import parse_lettered

    z = zipfile.ZipFile("/data/shared/visdata_dl/mindcube/data.zip")
    names = set(z.namelist())
    train_ids = {json.loads(l)["id"] for l in z.read("data/raw/MindCube_train.jsonl").decode().splitlines() if l.strip()}
    rows = [json.loads(l) for l in z.read("data/raw/MindCube.jsonl").decode().splitlines() if l.strip()]
    rows = [r for r in rows if r["id"] not in train_ids]
    random.Random(SEED + 5009).shuffle(rows)
    sv = B.Saver("dev_mindcube")
    out = []
    for r in rows:
        stem, opts = parse_lettered(r["question"], r" ([A-F])\. ")
        if not opts or r.get("gt_answer") not in LET[:len(opts)] or not 2 <= len(opts) <= 6:
            continue
        rels = []
        for p in r["images"][:4]:
            if f"data/{p}" not in names:
                rels = None
                break
            rels.append(sv.save(z.read(f"data/{p}"), re.sub(r"[^A-Za-z0-9_.-]", "_", p)[:90]))
        if not rels:
            continue
        q = stem
        for j in range(1, 5):
            q = q.replace(f"image {j}", f"[image {j}]")
        out.append(it("dev_mindcube", r["id"], rels, q, opts, r["gt_answer"], r.get("type", "multi_view"), sv))
        if len(out) >= n:
            break
    return out


def b_hm(n=480):
    from huggingface_hub import hf_hub_download

    rows = [json.loads(l) for s in ("dev_unseen", "test_unseen") for l in open(f"/data/shared/visdata_dl/hm/{s}.jsonl")]
    random.Random(SEED + 5010).shuffle(rows)
    sv = B.Saver("dev_hm")
    out = []
    for r in rows[: int(n * 1.4)]:
        p = Path("/data/shared/visdata_dl/hm") / r["img"]
        if not p.exists():
            try:
                hf_hub_download("neuralcatcher/hateful_memes", r["img"], repo_type="dataset", local_dir="/data/shared/visdata_dl/hm", token=False)
            except Exception:
                continue
        rel = sv.save(p.read_bytes(), str(r["id"]))
        out.append(it("dev_hm", str(r["id"]), rel, "Is this meme hateful?", ["Yes", "No"], "A" if r["label"] == 1 else "B", "binary", sv))
        if len(out) >= n:
            break
    return out


def b_sqa():
    import pyarrow.parquet as pq

    out = {"rb": [], "mm": []}
    for split, kind, cap in (("validation", "rb", 480), ("test", "mm", 240)):
        f = sorted((DL / "sqa").rglob(f"{split}-*.parquet"))[0]
        rows = [r for r in pq.read_table(f).to_pylist() if r.get("image") and (r["image"] or {}).get("bytes")]
        pool = [str(c) for r in rows for c in r["choices"]]
        rng = random.Random(SEED + (5011 if kind == "rb" else 5012))
        rng.shuffle(rows)
        fam = "dev_sqa_rb" if kind == "rb" else "dev_sqa_mmmu"
        sv = B.Saver(fam)
        for i, r in enumerate(rows):
            if len(out[kind]) >= cap:
                break
            ch = [str(c).strip() for c in r["choices"]]
            if len(ch) < 2 or len(set(c.lower() for c in ch)) != len(ch):
                continue
            gold = ch[int(r["answer"])]
            ds = [c for j, c in enumerate(ch) if j != int(r["answer"])]
            ctx = (r.get("hint") or "").strip() or None
            if kind == "rb":
                opts, g = VB.rbench_options_v2(gold, ds, pool, rng)
                rel = sv.save(r["image"]["bytes"], f"{split}{i}")
                q = f"{ctx}\n{r['question']}" if ctx else r["question"]
                out[kind].append(it(fam, f"{split}{i}", rel, q, opts, g, "other_correct" if opts[LET.index(g)] == VB.OTHER else "content_correct", sv,
                                    prompt={"options_in_image": False, "readout": "The answer is ("}))
            else:
                opts, g = VB.mmmu_options(gold, ds, pool, rng)
                rel = sv.save(VB.composite(r["image"]["bytes"], r["question"], opts, ctx), f"{split}{i}_page")
                k = len(opts)
                out[kind].append(it(fam, f"{split}{i}", rel, f"Answer the multiple-choice question shown in the image (options A-{LET[k - 1]}).",
                                    list(LET[:k]), g, "vision_only", sv, prompt={"options_in_image": True, "readout": "The answer is ("}))
    return out["rb"], out["mm"]


def build():
    B.ITEMS.mkdir(parents=True, exist_ok=True)
    fams = {}
    fams["dev_m2w"] = b_m2w()
    fams["dev_chartnet"], fams["dev_chartnet_mmmu"] = b_chartnet()
    fams["dev_doc_info"], fams["dev_doc_kie"] = b_docvqa()
    fams["dev_pixmo_cnt"] = b_counting()
    fams["dev_vqa_cnt"], fams["dev_vqa_rw"] = b_vqa(vqa_val())
    fams["dev_vrd"] = b_vrd()
    fams["dev_mindcube"] = b_mindcube()
    fams["dev_hm"] = b_hm()
    fams["dev_sqa_rb"], fams["dev_sqa_mmmu"] = b_sqa()
    for k, v in fams.items():
        with open(B.ITEMS / f"{k}.jsonl", "w") as f:
            for x in v:
                f.write(json.dumps(x) + "\n")
        print(json.dumps({"family": k, "items": len(v)}), flush=True)


def rows():
    """Dedupe vs public images AND every training pool, cap 400/board family, convert to kit rows (sha-pinned)."""
    import numpy as np
    from vb_manifest import PH_T, public_hashes

    pub = public_hashes()
    sha = {r["sha256"] for r in pub}
    phs = [int(r["phash"], 16) for r in pub if r["phash"]]
    tr_files = (list(Path("/data/local/visdata/items").glob("*.jsonl")) + list(Path("/data/local/visdata/items_b").glob("*.jsonl"))
                + list(Path("/data/local/visdata/items_b3").glob("*.jsonl")) + list(Path("/data/shared/visdata_c/items_c").glob("*.jsonl"))
                + list(Path("/data/shared/visdata_d/items_d").glob("*.jsonl")))
    n_tr = 0
    for f in tr_files:
        for l in open(f):
            r = json.loads(l)
            sha.update(r["img_src_sha256"]); sha.update(r["img_sha256"]); phs += [int(p, 16) for p in r["img_phash"]]
            n_tr += 1
    PH = np.array(phs, dtype=np.uint64)
    by = {}
    drops = Counter()
    for f in sorted(B.ITEMS.glob("*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            bad = any(s1 in sha or s2 in sha or int(np.bitwise_count(PH ^ np.uint64(int(p, 16))).min()) <= PH_T
                      for s1, s2, p in zip(r["img_src_sha256"], r["img_sha256"], r["img_phash"]))
            if bad:
                drops[r["source"]] += 1
                continue
            by.setdefault(r["family"], []).append(r)
    out = []
    for fam, lst in sorted(by.items()):
        srcs = sorted({r["source"] for r in lst})
        per = 400 // len(srcs)
        take = []
        for s in srcs:  # equal share per source within a family, topped up to 400
            take += [r for r in lst if r["source"] == s][:per]
        rest = [r for r in lst if r not in take]
        take += rest[: max(0, 400 - len(take))]
        for r in take[:400]:
            k = len(r["options"])
            vis = (r.get("prompt") or {}).get("options_in_image")
            crit = {LET[i]: (LET[i] if vis else str(o)) for i, o in enumerate(r["options"])}
            out.append({"id": r["id"], "benchmark": fam, "set": r["source"], "subtask": r["subtask"], "images": r["images"], "state": {},
                        "questions": {"answer": {"type": "choice", "instructions": r["question"], "criteria": crit}},
                        "prompt": r.get("prompt") or {"options_in_image": False, "readout": None}, "expected": {"answer": r["gold"]},
                        "_evaluation": {"run_id": r["id"], "benchmark": fam, "set": r["source"]}, "licence": r["licence"]})
    p = DV / "dev_rows.jsonl"
    with open(p, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    hdr = {"rows": len(out), "rows_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "per_family": dict(Counter(r["benchmark"] for r in out)),
           "per_source": dict(Counter(r["set"] for r in out)), "dedupe_drops": dict(drops), "dedupe_against": {"public_images": len(pub), "training_items": n_tr},
           "image_root": str(B.IMG)}
    (DV / "dev_rows.header.json").write_text(json.dumps(hdr, indent=1))
    print(json.dumps(hdr))


if __name__ == "__main__":
    {"download": download, "build": build, "rows": rows}[sys.argv[1]]()
