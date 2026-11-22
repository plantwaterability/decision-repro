"""Round D (A4 data, ctl2): real-world photo QA for the RealWorldQA gap + more counting.
usage: python vbd_build.py <download|vqa2_rw|vg_rw|tally_cnt>
- vqa2_rw  (clean): VQAv2 train (annotations CC BY 4.0, visualqa.org/terms; COCO train2014 images, fetched as train2017 by id)
- vg_rw    (clean): Visual Genome question_answers (CC BY 4.0; VG images)
- tally_cnt (NC): TallyQA train (repo Apache-2.0, but the 11-06 coordinator NC ruling stands; COCO / VG images)
MC conversion: yes/no -> ["Yes","No"] balanced; numeric -> kit numeric_distractors; other -> 3 distractors from the answer pool of the
same question prefix (first three words), never a synonym-substring of the gold. RWQA-style rows use the kit prompt shape."""
import io
import json
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/data/local/visdata/scripts")
import vb_build as B  # noqa: E402
from decision_index.vision.convert import NUM, SEED, numeric_distractors, norm  # noqa: E402

DLD = Path("/data/shared/visdata_dl_d")
B.IMG = Path("/data/shared/visdata_d/img_d")
B.ITEMS = Path("/data/shared/visdata_d/items_d")
B.LIC.update({
    "vqa2_rw": ("RealWorldQA", "clean", "VQAv2 train annotations (CC BY 4.0, visualqa.org/terms); COCO train2014 images"),
    "vg_rw": ("RealWorldQA", "clean", "Visual Genome question_answers (CC BY 4.0); Visual Genome images"),
    "tally_cnt": ("CV-Bench", "nc", "TallyQA train (repo LICENSE Apache-2.0; kept NC per the 11-06 coordinator ruling); COCO/VG images"),
})
LETTERS = "ABCDEFGHIJ"


def get(url, path):
    import httpx

    if path.exists() and path.stat().st_size > 0:
        return path
    with httpx.stream("GET", url, timeout=600, follow_redirects=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for c in r.iter_bytes(1 << 20):
                f.write(c)
    return path


def download():
    DLD.mkdir(parents=True, exist_ok=True)
    for u in ["https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
              "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
              "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/question_answers.json.zip",
              "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip",
              "https://github.com/manoja328/tallyqa/blob/master/tallyqa.zip?raw=true"]:
        name = u.split("/")[-1].split("?")[0]
        get(u, DLD / name)
        print(json.dumps({"downloaded": name, "bytes": (DLD / name).stat().st_size}), flush=True)


def mc(q, gold, pool, rng):
    g = gold.strip()
    if g.lower() in ("yes", "no"):
        return ["Yes", "No"], "A" if g.lower() == "yes" else "B"
    if NUM.match(g):
        ds = numeric_distractors(g, [p for p in pool if NUM.match(p)], rng)
        if len(ds) < 3:
            return None, None
        opts = ds + [g]
    else:
        cand = [p for p in dict.fromkeys(pool) if p.lower() != g.lower() and not NUM.match(p)
                and norm(p) and norm(g) and norm(p) not in norm(g) and norm(g) not in norm(p)]
        if len(cand) < 3:
            return None, None
        opts = rng.sample(cand, 3) + [g]
    rng.shuffle(opts)
    return opts, LETTERS[opts.index(g)]


def prefix(q):
    return " ".join(re.findall(r"[a-z]+", q.lower())[:3])


def fetch_imgs(pairs, workers=16):
    """Threaded fetch (same logic as round B's; NOT imported from vbb_build, whose import repoints B.IMG/B.ITEMS)."""
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
        return list(ex.map(one, pairs))


def b_vqa2(n=9000):
    qs = json.loads(zipfile.ZipFile(DLD / "v2_Questions_Train_mscoco.zip").read("v2_OpenEnded_mscoco_train2014_questions.json"))["questions"]
    an = json.loads(zipfile.ZipFile(DLD / "v2_Annotations_Train_mscoco.zip").read("v2_mscoco_train2014_annotations.json"))["annotations"]
    qmap = {q["question_id"]: q for q in qs}
    pools = defaultdict(list)
    recs = []
    for a in an:
        g = a["multiple_choice_answer"]
        agree = sum(1 for x in a["answers"] if x["answer"] == g)
        if agree < 7:
            continue
        q = qmap[a["question_id"]]
        pools[prefix(q["question"])].append(g)
        recs.append((a["question_id"], a["image_id"], q["question"], g, a["answer_type"]))
    rng = random.Random(SEED + 4001)
    rng.shuffle(recs)
    yn = Counter()
    pick = []
    per_img = Counter()
    for r in recs:
        if per_img[r[1]] >= 1:
            continue
        if r[4] == "yes/no":
            if yn[r[3]] >= n * 0.15:  # yes/no capped at 30% of the family, balanced yes/no
                continue
            yn[r[3]] += 1
        pick.append(r)
        per_img[r[1]] += 1
        if len(pick) >= int(n * 1.25):
            break
    raw = DLD / "coco_train2017"
    raw.mkdir(exist_ok=True)
    imgs = sorted({r[1] for r in pick})
    ok = dict(zip(imgs, fetch_imgs([(f"http://images.cocodataset.org/train2017/{i:012d}.jpg", raw / f"{i:012d}.jpg") for i in imgs])))
    sv = B.Saver("vqa2_rw")
    out = []
    for qid, iid, q, g, at in pick:
        if not ok.get(iid) or len(out) >= n:
            continue
        opts, gl = mc(q, g, list(dict.fromkeys(pools[prefix(q)]))[:400], rng)
        if not opts:
            continue
        rel = sv.save((raw / f"{iid:012d}.jpg").read_bytes(), f"{iid:012d}")
        out.append(B.item("vqa2_rw", str(qid), [rel], q, opts, gl, at, sv))
    return out


def b_vg(n=6000):
    imgs = {d["image_id"]: d for d in json.loads(zipfile.ZipFile(DLD / "image_data.json.zip").read("image_data.json"))}
    data = json.loads(zipfile.ZipFile(DLD / "question_answers.json.zip").read("question_answers.json"))
    pools = defaultdict(list)
    recs = []
    for d in data:
        for qa in d.get("qas", []):
            q, g = qa["question"].strip(), qa["answer"].strip().rstrip(".")
            if not q or not g or len(g.split()) > 4:
                continue
            pools[prefix(q)].append(g.lower())
            recs.append((qa["qa_id"], d["id"], q, g.lower()))
    rng = random.Random(SEED + 4002)
    rng.shuffle(recs)
    per_img, pick = Counter(), []
    for r in recs:
        if per_img[r[1]] >= 1 or r[1] not in imgs or imgs[r[1]].get("coco_id"):  # VG-only photos (COCO ones come via VQAv2)
            continue
        per_img[r[1]] += 1
        pick.append(r)
        if len(pick) >= int(n * 1.4):
            break
    raw = DLD / "vg_images"
    raw.mkdir(exist_ok=True)
    iids = sorted({r[1] for r in pick})
    ok = dict(zip(iids, fetch_imgs([(imgs[i]["url"], raw / f"{i}.jpg") for i in iids])))
    sv = B.Saver("vg_rw")
    out = []
    for qaid, iid, q, g in pick:
        if not ok.get(iid) or len(out) >= n:
            continue
        opts, gl = mc(q, g, list(dict.fromkeys(pools[prefix(q)]))[:400], rng)
        if not opts:
            continue
        rel = sv.save((raw / f"{iid}.jpg").read_bytes(), str(iid))
        out.append(B.item("vg_rw", str(qaid), [rel], q, opts, gl, "vg_qa", sv))
    return out


def b_tally(n=7000):
    z = zipfile.ZipFile(DLD / "tallyqa.zip")
    name = next(x for x in z.namelist() if x.endswith("train.json"))
    data = json.loads(z.read(name))
    rng = random.Random(SEED + 4003)
    rng.shuffle(data)
    per_img, pick = Counter(), []
    for r in data:
        if per_img[r["image"]] >= 1 or not (0 <= int(r["answer"]) <= 15):
            continue
        if "val2014" in r["image"]:  # COCO val2014 overlaps val2017 (CV-Bench's pool)
            continue
        per_img[r["image"]] += 1
        pick.append(r)
        if len(pick) >= int(n * 1.3):
            break
    raw = DLD / "tally_images"
    raw.mkdir(exist_ok=True)

    def url(p):
        if p.startswith("train2014/"):
            cid = int(re.findall(r"(\d+)\.jpg", p)[0])
            return "http://images.cocodataset.org/train2017/%012d.jpg" % cid
        return "https://cs.stanford.edu/people/rak248/" + p

    ok = fetch_imgs([(url(r["image"]), raw / r["image"].replace("/", "_")) for r in pick])
    sv = B.Saver("tally_cnt")
    out = []
    for r, good in zip(pick, ok):
        if not good or len(out) >= n:
            continue
        g = str(int(r["answer"]))
        ds = numeric_distractors(g, [], rng)
        if len(ds) < 3:
            continue
        opts = ds + [g]
        rng.shuffle(opts)
        rel = sv.save((raw / r["image"].replace("/", "_")).read_bytes(), Path(r["image"]).stem)
        out.append(B.item("tally_cnt", str(r.get("question_id", r["image"] + r["question"])), [rel], r["question"], opts, LETTERS[opts.index(g)],
                          "simple" if r.get("issimple") else "complex", sv))
    return out


if __name__ == "__main__":
    fam = sys.argv[1]
    if fam == "download":
        download()
        sys.exit(0)
    B.ITEMS.mkdir(parents=True, exist_ok=True)
    items = {"vqa2_rw": b_vqa2, "vg_rw": b_vg, "tally_cnt": b_tally}[fam]()
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)), "duplicate ids"
    with open(B.ITEMS / f"{fam}.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(json.dumps({"family": fam, "items": len(items), "gold": dict(Counter(it["gold"] for it in items)),
                      "subtasks": dict(Counter(it["subtask"] for it in items))}), flush=True)
