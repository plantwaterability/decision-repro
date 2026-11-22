"""Round G (data fork, ctl2 CPU): counting-first vision rows for the weak HELD families. Reuses round F / round E code unchanged
(vf_build: kit Saver / item via fork B's FS; ve_build.count_options; vf_build3 text_mc / num_mc / numeric_distractors).
usage: python vg_build.py <builder>
  fhm          fhm_fg    (NC)   : Fine-grained Hateful Memes TRAIN annotations -> new [Yes, No] constructs on HM TRAIN images
  chartqa_h    chartqa_h (NC)   : ChartQA train, human-authored questions only (round F's conversion rules; separate pool for the A/B)
  hs_cnt       hs_cnt    (clean): Hypersim train-scene instance counts (NYU40 classes), round F's exposure rule
  pixmo_plan                    : PixMo-Points candidates (plain countable nouns, counts 0-10, one row per image) -> dl/pixmo/plan.jsonl
  pixmo_fetch                   : fetch the planned images by URL (TLS verified); keep only bytes whose sha256 = the row's image_sha256
  pixmo        pixmo_cnt (NC)   : rows from the verified images (flat-illustration screen, board wording, 4-6 numeric options)
Every image goes through the kit Saver (sha256 of source + saved bytes, 64-bit pHash) for the dedupe gate in vg_mix.py."""
import csv
import hashlib
import io
import json
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vf_build as VF  # noqa: E402  (imports fork B's ve_build and the kit's vb_build; sets the kit path)
import vf_build3 as VF3  # noqa: E402
from vf_build import B, FS, LETTERS  # noqa: E402
from decision_index.vision.convert import NUM, norm  # noqa: E402

V = VF.V
G = Path("/data/shared/visdata_g")
DLG = G / "dl"
DLF = VF.DLF  # round F downloads (ChartQA zip)
B.IMG = G / "img_g"
B.ITEMS = G / "items_g"
SEED = 20261114
HM = "Moderation (Hateful Memes)"
B.LIC.update({
    "pixmo_cnt": ("CV-Bench", "nc", "PixMo-Points allenai/pixmo-points@2b5c6931 (ODC-BY-1.0, 'intended for research and educational use'), human point "
                                    "annotations; images are third-party web images fetched by URL (bytes = the row's image_sha256), under their own terms"),
    "hs_cnt": ("CV-Bench", "clean", VF.HS_LIC + "instance + semantic masks (NYU40 classes)"),
    "fhm_fg": (HM, "nc", "Fine-grained hateful memes train annotations facebookresearch/fine_grained_hateful_memes@845c5f86 (Apache-2.0) on Hateful Memes "
                         "TRAIN images (Facebook HM dataset licence: research; Getty images)"),
    "chartqa_h": ("CharXiv", "nc", "ChartQA train, human-authored questions only, via ahmed-masry/ChartQA@af8b6f5c (GPL-3.0; chart images from Statista / "
                                   "Pew / OWID / OECD)"),
    "gqa_rw": ("RealWorldQA", "clean", "GQA balanced train questions + scene graphs v1.1 (cs.stanford.edu/people/dorarad/gqa, CC BY 4.0); Visual Genome "
                                       "images (CC BY 4.0; Flickr / COCO photos)"),
})
COUNT_Q = "How many {p} are in the image?"


def write(name, res, summ):
    B.ITEMS.mkdir(parents=True, exist_ok=True)
    (G / "summ").mkdir(parents=True, exist_ok=True)
    rep = {"builder": name, "summary": summ, "sources": {}}
    for fam, items in res.items():
        ids = [it["id"] for it in items]
        assert len(ids) == len(set(ids)), f"duplicate ids in {fam}"
        with open(B.ITEMS / f"{fam}.jsonl", "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        rep["sources"][fam] = {"items": len(items), "images": len({p for it in items for p in it["images"]}),
                               "gold": dict(sorted(Counter(it["gold"] for it in items).items())),
                               "n_options": dict(sorted(Counter(len(it["options"]) for it in items).items())),
                               "subtask": dict(Counter(it["subtask"] for it in items)), "line": items[0]["line"] if items else None}
    (G / "summ" / f"{name}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    print(json.dumps(rep, ensure_ascii=False, default=str)[:6000], flush=True)
    print(f"G_BUILD_DONE {name}", flush=True)


# ---------------------------------------------------------------- Fine-grained Hateful Memes (train annotations only)
FHM_Q = {
    "race": "Does this meme attack people because of their race?",
    "religion": "Does this meme attack people because of their religion?",
    "nationality": "Does this meme attack people because of their nationality?",
    "sex": "Does this meme attack people because of their sex or gender?",
    "disability": "Does this meme attack people because of a disability?",
    "dehumanizing": "Does this meme dehumanize a group of people?",
    "mocking": "Does this meme mock a group of people?",
    "contempt": "Does this meme express contempt for a group of people?",
    "inferiority": "Does this meme present a group of people as inferior?",
    "slurs": "Does this meme use a slur?",
    "exclusion": "Does this meme call for excluding a group of people?",
    "inciting_violence": "Does this meme incite violence against a group of people?",
}
PC = ("race", "religion", "nationality", "sex", "disability")


def b_fhm(max_per_meme=3):
    raw = (DLG / "fhm" / "train.json").read_text(encoding="utf-8")
    try:
        recs = json.loads(raw)
        recs = recs if isinstance(recs, list) else list(recs.values())
    except json.JSONDecodeError:
        recs = [json.loads(l) for l in raw.splitlines() if l.strip()]
    hm = B.DL / "hm"
    ev = set()
    for sp in ("dev_seen", "dev_unseen", "test_seen", "test_unseen"):  # ids only: every HM eval-split id must be absent from the train file
        ev |= {int(json.loads(l)["id"]) for l in open(hm / f"{sp}.jsonl")}
    hm_train = {int(json.loads(l)["id"]) for l in open(hm / "train.jsonl")}
    c = Counter()
    assert all(r.get("set_name") == "train" for r in recs), Counter(r.get("set_name") for r in recs)
    assert not ({int(r["id"]) for r in recs} & ev), "FHM train id in an HM eval split"
    c["records"] = len(recs)
    c["ids_in_hm_train"] = sum(int(r["id"]) in hm_train for r in recs)
    rng = random.Random(f"{SEED}:fhm")
    pos, neg_nh, neg_h = defaultdict(list), defaultdict(list), defaultdict(list)
    meta = {}
    for r in sorted(recs, key=lambda r: int(r["id"])):
        mid = int(r["id"])
        if mid not in hm_train:
            c["not_in_hm_train"] += 1
            continue
        name = Path(r["img"]).name
        p = hm / "img" / name
        if not p.exists():
            p = V.DLE / "hm_exp" / "img" / name
        if not p.exists():
            c["image_missing"] += 1
            continue
        hate = "hateful" in (r.get("gold_hate") or [])
        labs = set(r.get("gold_pc") or []) | set(r.get("gold_attack") or [])
        meta[mid] = (p, r.get("text", ""), sorted(set(r.get("gold_pc") or [])), sorted(set(r.get("gold_attack") or [])), hate)
        for k in FHM_Q:
            if k in labs:
                pos[k].append(mid)
            elif hate:
                neg_h[k].append(mid)
            else:
                neg_nh[k].append(mid)
    per_meme, rows = Counter(), []
    for k in FHM_Q:  # every positive; as many negatives, half from non-hateful memes and half from hateful memes without this label
        P = pos[k][:]
        rng.shuffle(P)
        P = [m for m in P if per_meme[m] < max_per_meme]
        for m in P:
            per_meme[m] += 1
        nh, h = neg_nh[k][:], neg_h[k][:]
        rng.shuffle(nh)
        rng.shuffle(h)
        nh = [m for m in nh if per_meme[m] < max_per_meme]
        h = [m for m in h if per_meme[m] < max_per_meme]
        n_h = min(len(h), len(P) // 2)
        N = h[:n_h] + nh[:len(P) - n_h]
        for m in N:
            per_meme[m] += 1
        rows += [(m, k, "A") for m in P] + [(m, k, "B") for m in N]
        c[f"{k}_yes"] = len(P)
        c[f"{k}_no_hateful"] = n_h
        c[f"{k}_no_not_hateful"] = len(N) - n_h
    sv = FS("fhm_fg")
    out = []
    for m, k, gold in sorted(rows):
        p, text, gpc, gat, hate = meta[m]
        rel = sv.save(p.read_bytes(), str(m))
        out.append(B.item("fhm_fg", f"{m}_{k}", [rel], FHM_Q[k], ["Yes", "No"], gold, "fine_target" if k in PC else "fine_attack", sv,
                          extra={"text": text, "construct": k, "gold_pc": gpc, "gold_attack": gat, "hateful": hate}))
    c["memes_used"] = len({m for m, _, _ in rows})
    return {"fhm_fg": out}, dict(c)


# ---------------------------------------------------------------- ChartQA train, human-authored questions only
def b_chartqa_h(cap=2):
    z = zipfile.ZipFile(DLF / "chartqa" / "ChartQA_Dataset.zip")
    root = "ChartQA Dataset/train/"
    names = set(z.namelist())
    hu = json.loads(z.read(root + "train_human.json"))
    tcache = {}

    def table(img):  # round F's table reader, unchanged
        if img not in tcache:
            nums, labs = [], []
            try:
                for row in csv.reader(io.StringIO(z.read(root + "tables/" + Path(img).stem + ".csv").decode("utf-8", "replace"))):
                    for cell in row:
                        s = cell.strip()
                        t = s.replace(",", "").replace("$", "").rstrip("%").strip()
                        if re.fullmatch(r"-?\d+(\.\d+)?", t):
                            nums.append(t)
                        elif 1 <= len(s) <= 60:
                            labs.append(s)
            except KeyError:
                pass
            tcache[img] = (nums, labs)
        return tcache[img]

    c, per, out = Counter(), Counter(), []
    sv = FS("chartqa_h")
    rng = random.Random(f"{SEED}:chartqa_h")
    qs = sorted(hu, key=lambda r: (r["imgname"], r["query"]))
    rng.shuffle(qs)
    for r in qs:
        img, lab = r["imgname"], str(r["label"]).strip()
        if per[img] >= cap or root + "png/" + img not in names:
            c["cap_or_missing"] += 1
            continue
        rr = random.Random(f"{SEED}:chartqa_h:{img}:{r['query']}")
        nums, labs = table(img)
        a = lab.replace(",", "")
        if lab.lower() in ("yes", "no"):
            c["skip_yes_no"] += 1
            continue
        if NUM.match(a):
            pct = a.endswith("%")
            opts, gold = VF3.num_mc(a, [x + ("%" if pct else "") for x in nums], rr)
            sub = "number-in-chart"
        else:
            if len(lab) > 60 or lab.startswith("["):
                c["skip_list_or_long"] += 1
                continue
            opts, gold = VF3.text_mc(lab, [[x for x in labs if not NUM.match(x)]], {norm(lab)}, rr, 4)
            sub = "text-in-chart"
        if not opts:
            c["no_distractors"] += 1
            continue
        rel = sv.save(z.read(root + "png/" + img), Path(img).stem)
        per[img] += 1
        out.append(B.item("chartqa_h", f"human_{Path(img).stem}_{per[img]}", [rel], " ".join(r["query"].split()), opts, gold, sub, sv, extra={"set": "human"}))
    return {"chartqa_h": out}, {"human_questions": len(hu), "charts": len(per), **c}


# ---------------------------------------------------------------- Hypersim instance counts (train scenes, NYU40 classes)
HS_CNT = {5: ("chair", "chairs"), 18: ("pillow", "pillows"), 35: ("lamp", "lamps"), 11: ("picture", "pictures"), 9: ("window", "windows"),
          8: ("door", "doors"), 3: ("cabinet", "cabinets"), 7: ("table", "tables"), 27: ("towel", "towels"), 29: ("box", "boxes"),
          6: ("sofa", "sofas"), 4: ("bed", "beds"), 19: ("mirror", "mirrors"), 34: ("sink", "sinks"), 33: ("toilet", "toilets"),
          32: ("nightstand", "nightstands"), 25: ("television", "televisions"), 14: ("desk", "desks"), 37: ("bag", "bags"),
          24: ("refrigerator", "refrigerators"), 36: ("bathtub", "bathtubs"), 17: ("dresser", "dressers")}
HS_W = {0: .05, 1: .10, 2: .17, 3: .16, 4: .14, 5: .11, 6: .09, 7: .07, 8: .05, 9: .03, 10: .03}


def b_hs_cnt(n=5000, per_frame=2):
    import h5py
    import numpy as np
    from PIL import Image

    RAWC, HF, EXPO = Path("/data/shared/visdata_c/hypersim"), DLF / "hypersim", {"blown_over": 0.30, "dark_over": 0.50, "median_under": 14}
    c, cands = Counter(), []
    scenes = sorted(p.parent.name for p in HF.glob("ai_*/.done"))
    for si, sd in enumerate(scenes):
        fdir = RAWC / sd / "images" / "scene_cam_00_final_preview"
        ldir = HF / sd / "images" / "scene_cam_00_geometry_hdf5"
        for fp in sorted(ldir.glob("frame.*.semantic_instance.hdf5")):
            fr = fp.name.split(".")[1]
            try:
                ins = h5py.File(fp)["dataset"][:]
                sem = h5py.File(ldir / f"frame.{fr}.semantic.hdf5")["dataset"][:]
                jp = fdir / f"frame.{fr}.color.jpg"
                im = Image.open(jp).convert("RGB")
            except Exception:
                c["frame_read_error"] += 1
                continue
            H, W = ins.shape
            if im.size != (W, H):
                c["frame_shape_mismatch"] += 1
                continue
            c["frames"] += 1
            lum = np.asarray(im.convert("L"), dtype=np.float32)
            th = lum[::4, ::4]
            if float((th >= 250).mean()) > EXPO["blown_over"] or float((th <= 12).mean()) > EXPO["dark_over"] or float(np.median(th)) < EXPO["median_under"]:
                c["frame_exposure_skip"] += 1
                continue
            tot = H * W
            big, amb = Counter(), set()
            for i in np.unique(ins):
                if i < 0:
                    continue
                mk = ins == i
                nn = int(mk.sum())
                vals, cnts = np.unique(sem[mk], return_counts=True)
                cls = int(vals[cnts.argmax()])
                if cls not in HS_CNT:
                    continue
                if cnts.max() < 0.8 * nn:  # instance of mixed semantics: the class count is not reliable in this frame
                    amb.add(cls)
                    continue
                if nn >= 0.0015 * tot and 20 <= float(lum[mk].mean()) <= 240:
                    big[cls] += 1
                elif nn >= 0.0001 * tot:  # a sliver a person might or might not count: leave the class out of this frame
                    amb.add(cls)
            present = set(int(v) for v in np.unique(sem))
            for cls in HS_CNT:
                if cls in amb:
                    continue
                stray = int(((sem == cls) & (ins < 0)).sum())  # class pixels outside any instance: unknown objects, leave the class out
                if stray > 0.001 * tot:
                    continue
                k = big.get(cls, 0)
                if k == 0 and cls in present:
                    continue
                if k <= 10:
                    cands.append((f"{sd}_{fr}", cls, k, str(jp)))
        if si % 40 == 0:
            print(json.dumps({"hs_cnt_progress": si, "scene": sd, "cands": len(cands), **c}), flush=True)
    rng = random.Random(f"{SEED}:hs_cnt")
    cands.sort()
    rng.shuffle(cands)
    quota = {k: round(n * w) for k, w in HS_W.items()}
    got, per, cat, taken = [], Counter(), Counter(), set()
    for relax in (False, True):
        for key, cls, k, jp in cands:
            if len(got) >= n:
                break
            if per[key] >= per_frame or (key, cls) in taken or cat[cls] >= (0.18 if relax else 0.12) * n:
                continue
            if not relax and quota.get(k, 0) <= 0:
                continue
            if relax and not 1 <= k <= 8:
                continue
            if not relax:
                quota[k] -= 1
            taken.add((key, cls))
            per[key] += 1
            cat[cls] += 1
            got.append((key, cls, k, jp))
    sv = FS("hs_cnt")
    out = []
    for key, cls, k, jp in sorted(got):
        r = random.Random(f"{SEED}:hs_cnt:{key}:{cls}")
        rel = sv.save(Path(jp).read_bytes(), key)
        opts, gold = V.count_options(k, r)
        out.append(B.item("hs_cnt", f"{key}_{HS_CNT[cls][0]}", [rel], COUNT_Q.format(p=HS_CNT[cls][1]), opts, gold, "Count", sv,
                          extra={"count": k, "class": HS_CNT[cls][0]}))
    return {"hs_cnt": out}, {"scenes": len(scenes), "candidates": len(cands), **c, "count_hist": dict(sorted(Counter(g[2] for g in got).items())),
                             "classes": Counter(HS_CNT[g[1]][0] for g in got).most_common(), "rule": "instance >= 0.15% of the frame and lit; class skipped "
                             "if any instance is a sliver (0.01-0.15%), of mixed semantics, or class pixels lie outside instances; zero = class absent"}


# ---------------------------------------------------------------- PixMo-Points (human exhaustive points), counting rows
# plain countable nouns: (singular, plural, group); group "in" = ADE-style indoor / household / room things, "out" = other countable things
LEX = [(s, p, "in") for s, p in [
    ("chair", "chairs"), ("table", "tables"), ("window", "windows"), ("door", "doors"), ("lamp", "lamps"), ("pillow", "pillows"),
    ("cushion", "cushions"), ("book", "books"), ("bottle", "bottles"), ("cup", "cups"), ("mug", "mugs"), ("glass", "glasses"),
    ("plate", "plates"), ("bowl", "bowls"), ("vase", "vases"), ("plant", "plants"), ("potted plant", "potted plants"), ("picture", "pictures"),
    ("painting", "paintings"), ("picture frame", "picture frames"), ("frame", "frames"), ("mirror", "mirrors"), ("shelf", "shelves"),
    ("cabinet", "cabinets"), ("drawer", "drawers"), ("bed", "beds"), ("sofa", "sofas"), ("couch", "couches"), ("armchair", "armchairs"),
    ("stool", "stools"), ("bench", "benches"), ("desk", "desks"), ("computer", "computers"), ("monitor", "monitors"), ("laptop", "laptops"),
    ("keyboard", "keyboards"), ("television", "televisions"), ("tv", "tvs"), ("clock", "clocks"), ("candle", "candles"), ("basket", "baskets"),
    ("box", "boxes"), ("towel", "towels"), ("curtain", "curtains"), ("rug", "rugs"), ("light", "lights"), ("ceiling light", "ceiling lights"),
    ("light bulb", "light bulbs"), ("chandelier", "chandeliers"), ("sink", "sinks"), ("faucet", "faucets"), ("toilet", "toilets"),
    ("speaker", "speakers"), ("jar", "jars"), ("can", "cans"), ("spoon", "spoons"), ("fork", "forks"), ("knife", "knives"),
    ("napkin", "napkins"), ("pot", "pots"), ("pan", "pans"), ("teapot", "teapots"), ("kettle", "kettles"), ("wine glass", "wine glasses"),
    ("glass jar", "glass jars"), ("hanger", "hangers"), ("shoe", "shoes"), ("sneaker", "sneakers"), ("boot", "boots"), ("hat", "hats"),
    ("bag", "bags"), ("handbag", "handbags"), ("backpack", "backpacks"), ("suitcase", "suitcases"), ("umbrella", "umbrellas"), ("toy", "toys"),
    ("stuffed animal", "stuffed animals"), ("teddy bear", "teddy bears"), ("doll", "dolls"), ("ball", "balls"), ("balloon", "balloons"),
    ("pen", "pens"), ("pencil", "pencils"), ("remote", "remotes"), ("phone", "phones"), ("cell phone", "cell phones"), ("tray", "trays"),
    ("container", "containers"), ("flower pot", "flower pots"), ("flowerpot", "flowerpots"), ("pot plant", "pot plants"), ("bucket", "buckets"),
    ("chair leg", "chair legs"), ("seat", "seats"), ("drawer handle", "drawer handles"), ("knob", "knobs"), ("outlet", "outlets"),
    ("light switch", "light switches"), ("ceiling fan", "ceiling fans"), ("fan", "fans"), ("radiator", "radiators"), ("stair", "stairs"),
    ("column", "columns"), ("pillar", "pillars"), ("plant pot", "plant pots"), ("shelf unit", "shelf units"), ("dish", "dishes"),
    ("bowl of food", "bowls of food"), ("wine bottle", "wine bottles"), ("can of soda", "cans of soda"), ("coaster", "coasters"),
    ("placemat", "placemats"), ("tablecloth", "tablecloths"), ("blanket", "blankets"), ("lampshade", "lampshades"), ("sconce", "sconces"),
    ("bookshelf", "bookshelves"), ("bookcase", "bookcases"), ("nightstand", "nightstands"), ("dresser", "dressers"), ("wardrobe", "wardrobes"),
    ("stove", "stoves"), ("oven", "ovens"), ("microwave", "microwaves"), ("refrigerator", "refrigerators"), ("fridge", "fridges"),
    ("bathtub", "bathtubs"), ("shower", "showers"), ("trash can", "trash cans"), ("bin", "bins"), ("tissue box", "tissue boxes"),
    ("cupboard", "cupboards"), ("countertop", "countertops"), ("stool", "stools"), ("coat", "coats"), ("jacket", "jackets"), ("shirt", "shirts"),
    ("sock", "socks"), ("glove", "gloves"), ("dress", "dresses")]] + [(s, p, "out") for s, p in [
    ("person", "people"), ("people", "people"), ("man", "men"), ("woman", "women"), ("child", "children"), ("kid", "kids"), ("boy", "boys"),
    ("girl", "girls"), ("car", "cars"), ("truck", "trucks"), ("bus", "buses"), ("bicycle", "bicycles"), ("bike", "bikes"),
    ("motorcycle", "motorcycles"), ("boat", "boats"), ("vehicle", "vehicles"), ("dog", "dogs"), ("cat", "cats"), ("bird", "birds"),
    ("horse", "horses"), ("cow", "cows"), ("sheep", "sheep"), ("duck", "ducks"), ("animal", "animals"), ("tree", "trees"),
    ("building", "buildings"), ("house", "houses"), ("tower", "towers"), ("street light", "street lights"), ("streetlight", "streetlights"),
    ("traffic light", "traffic lights"), ("sign", "signs"), ("pole", "poles"), ("flag", "flags"), ("wheel", "wheels"), ("tire", "tires"),
    ("apple", "apples"), ("orange", "oranges"), ("banana", "bananas"), ("egg", "eggs"), ("cookie", "cookies"), ("donut", "donuts"),
    ("doughnut", "doughnuts"), ("cupcake", "cupcakes"), ("cake", "cakes"), ("pizza", "pizzas"), ("sandwich", "sandwiches"),
    ("strawberry", "strawberries"), ("lemon", "lemons"), ("tomato", "tomatoes"), ("cherry", "cherries"), ("grape", "grapes"),
    ("coin", "coins"), ("tennis ball", "tennis balls"), ("rock", "rocks"), ("stone", "stones"), ("flower", "flowers"), ("rose", "roses"),
    ("tulip", "tulips"), ("sunflower", "sunflowers"), ("leaf", "leaves"), ("mushroom", "mushrooms"), ("candy", "candies"), ("muffin", "muffins"),
    ("pumpkin", "pumpkins"), ("potato", "potatoes"), ("carrot", "carrots"), ("pepper", "peppers"), ("onion", "onions"), ("bread roll", "bread rolls"),
    ("fish", "fish"), ("chicken", "chickens"), ("goat", "goats"), ("pig", "pigs"), ("elephant", "elephants"), ("giraffe", "giraffes"),
    ("zebra", "zebras"), ("deer", "deer"), ("penguin", "penguins"), ("butterfly", "butterflies"), ("bee", "bees"), ("car tire", "car tires"),
    ("cone", "cones"), ("traffic cone", "traffic cones"), ("bench", "benches"), ("lamp post", "lamp posts"), ("fence post", "fence posts"),
    ("tent", "tents"), ("kite", "kites"), ("surfboard", "surfboards"), ("skateboard", "skateboards"), ("airplane", "airplanes"),
    ("plane", "planes"), ("train", "trains"), ("player", "players"), ("soldier", "soldiers"), ("cyclist", "cyclists"), ("motorcyclist", "motorcyclists")]]
NOUN = {}
for s_, p_, g_ in LEX:
    NOUN.setdefault(s_, (p_, g_))
    NOUN.setdefault(p_, (p_, g_))
ART = re.compile(r"^(the|a|an|all the|all|every|each|any)\s+")
# v2 (after the first pool's contact sheets): ONLY collection_method "counting" rows. The "pointing" rows carry no count field and their points
# are not exhaustive (sheet: "trays = 3" on one tray, "windows = 1" on a wall of panes, "seats = 10" on a double-decker bus), so the number of
# points is not a count. Counting rows hold counts of 0 and 4 and above only.
PX_W = {0: .06, 4: .20, 5: .18, 6: .16, 7: .13, 8: .11, 9: .08, 10: .08}


def px_label(s):
    s = " ".join(str(s or "").lower().replace("_", " ").split()).strip(" .,:;!?\"'")
    return ART.sub("", s)


def b_pixmo_plan(n_plan=26000, in_share=0.62):
    import pyarrow.parquet as pq

    t = pq.read_table([DLG / "pixmo_points" / f"train-0000{i}-of-00002.parquet" for i in (0, 1)],
                      columns=["image_url", "image_sha256", "count", "label", "collection_method", "points"])
    url, sh, cnt, lab, meth = (t.column(k).to_pylist() for k in ("image_url", "image_sha256", "count", "label", "collection_method"))
    import pyarrow.compute as pc

    npts = [0 if v is None else v for v in pc.list_value_length(t.column("points")).to_pylist()]
    del t
    per_img_expr, by = Counter(), defaultdict(lambda: defaultdict(set))
    c = Counter()
    for i in range(len(url)):
        per_img_expr[sh[i]] += 1
        L = px_label(lab[i])
        if L not in NOUN:
            continue
        if meth[i] != "counting" or cnt[i] is None:  # v2: counting rows only (see PX_W)
            continue
        k = cnt[i]
        if k != npts[i]:  # the row's count and its number of points must agree
            c["count_differs_from_points"] += 1
            continue
        by[sh[i]][NOUN[L][0]].add((int(k), url[i], meth[i]))
    cands = []
    for s, d in by.items():
        for p, vals in d.items():
            ks = {v[0] for v in vals}
            if len(ks) != 1:  # two annotations of the same noun disagree on the count: ambiguous, left out
                c["conflicting_counts"] += 1
                continue
            k = ks.pop()
            if not 0 <= k <= 10:
                continue
            v = sorted(vals)[0]
            cands.append({"sha": s, "url": v[1], "plural": p, "group": NOUN[p][1], "count": k, "method": v[2], "n_expr": per_img_expr[s]})
    c["candidates"] = len(cands)
    c["candidate_images"] = len({x["sha"] for x in cands})
    rng = random.Random(f"{SEED}:pixmo_plan")
    cands.sort(key=lambda x: (x["sha"], x["plural"]))
    rng.shuffle(cands)
    cands.sort(key=lambda x: 0 if x["n_expr"] >= 3 else 1)  # clutter first (stable: the seeded order holds within each tier)
    quota = {(g, k): round(n_plan * (in_share if g == "in" else 1 - in_share) * w) for g in ("in", "out") for k, w in PX_W.items()}
    cap = {"people": 0.05, "cars": 0.03}
    got, used, cat = [], set(), Counter()
    for x in cands:
        if len(got) >= n_plan:
            break
        q = (x["group"], x["count"])
        if x["sha"] in used or quota.get(q, 0) <= 0 or cat[x["plural"]] >= cap.get(x["plural"], 0.025) * n_plan:
            continue
        quota[q] -= 1
        used.add(x["sha"])
        cat[x["plural"]] += 1
        got.append(x)
    P = DLG / "pixmo"
    P.mkdir(parents=True, exist_ok=True)
    with open(P / "plan.jsonl", "w") as f:
        for x in got:
            f.write(json.dumps(x) + "\n")
    summ = {**c, "planned": len(got), "group": dict(Counter(x["group"] for x in got)), "count_hist": dict(sorted(Counter(x["count"] for x in got).items())),
            "method": dict(Counter(x["method"] for x in got)), "top": cat.most_common(40), "unfilled_quota": {f"{g}:{k}": v for (g, k), v in quota.items() if v > 0}}
    (P / "plan_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ)[:3000], flush=True)
    print("G_PIXMO_PLAN_DONE", flush=True)


UA = {"User-Agent": "Mozilla/5.0 (compatible; img-fetch/1.0)"}


def b_pixmo_fetch(workers=48):
    import time
    from concurrent.futures import ThreadPoolExecutor

    import httpx

    P = DLG / "pixmo"
    D = P / "img"
    D.mkdir(parents=True, exist_ok=True)
    plan = [json.loads(l) for l in open(P / "plan.jsonl")]
    st_p = P / "fetch_status.jsonl"
    done = {}
    if st_p.exists():
        for l in open(st_p):
            try:
                r = json.loads(l)
                done[r["sha"]] = r["st"]
            except Exception:
                pass
    todo = [x for x in plan if x["sha"] not in done]
    print(json.dumps({"planned": len(plan), "already": len(done), "todo": len(todo)}), flush=True)
    cli = httpx.Client(timeout=httpx.Timeout(20.0, connect=10.0), follow_redirects=True, headers=UA, limits=httpx.Limits(max_connections=workers * 2))

    def one(x):
        try:
            r = cli.get(x["url"])
            if r.status_code != 200 or not r.content:
                return x["sha"], f"http_{r.status_code}"
            if hashlib.sha256(r.content).hexdigest() != x["sha"]:
                return x["sha"], "sha_differs"
            (D / x["sha"]).write_bytes(r.content)
            return x["sha"], "ok"
        except httpx.ConnectError as e:
            return x["sha"], "connect_error" + (":tls" if "CERTIFICATE" in str(e).upper() else "")
        except Exception as e:
            return x["sha"], type(e).__name__

    t0, n = time.time(), 0
    with open(st_p, "a") as f, ThreadPoolExecutor(workers) as ex:
        for s, st in ex.map(one, todo):
            f.write(json.dumps({"sha": s, "st": st}) + "\n")
            n += 1
            if n % 1000 == 0:
                f.flush()
                print(json.dumps({"fetched": n, "secs": round(time.time() - t0)}), flush=True)
    allst = Counter()
    for l in open(st_p):
        allst[json.loads(l)["st"]] += 1
    print(json.dumps({"status": dict(allst.most_common())}), flush=True)
    print("G_PIXMO_FETCH_DONE", flush=True)


def photo_stats(im):
    """flat-illustration screen: colours after 5-bit quantisation in a 128 x 128 thumbnail, and the share of the 10 most common colours"""
    import numpy as np

    a = np.asarray(im.convert("RGB").resize((128, 128)), dtype=np.uint8) >> 3
    k = (a[..., 0].astype(np.int32) << 10) | (a[..., 1].astype(np.int32) << 5) | a[..., 2]
    u, cnt = np.unique(k, return_counts=True)
    top = np.sort(cnt)[::-1][:10].sum() / k.size
    return int(len(u)), round(float(top), 3)


def b_pixmo(n=16000, min_side=224, min_colours=300, max_top10=0.50):
    from PIL import Image

    P = DLG / "pixmo"
    plan = [json.loads(l) for l in open(P / "plan.jsonl")]
    sv = FS("pixmo_cnt")
    out, c, scr = [], Counter(), []
    for x in plan:
        if len(out) >= n:
            break
        p = P / "img" / x["sha"]
        if not p.exists():
            c["not_fetched"] += 1
            continue
        data = p.read_bytes()
        try:
            im = Image.open(io.BytesIO(data))
            im.load()
        except Exception:
            c["unreadable"] += 1
            continue
        if min(im.size) < min_side:
            c["too_small"] += 1
            continue
        u, top = photo_stats(im)
        scr.append({"sha": x["sha"], "colours": u, "top10": top})
        if u < min_colours and top > max_top10:  # very flat AND dominated by a few colours (vector art, renders on white); calibrated on a sheet
            c["flat_illustration"] += 1
            continue
        r = random.Random(f"{SEED}:pixmo:{x['sha']}")
        rel = sv.save(data, x["sha"][:24])
        opts, gold = V.count_options(x["count"], r)
        out.append(B.item("pixmo_cnt", f"{x['sha'][:24]}_{x['plural'].replace(' ', '_')}", [rel], COUNT_Q.format(p=x["plural"]), opts, gold, "Count", sv,
                          extra={"count": x["count"], "class": x["plural"], "group": x["group"], "method": x["method"], "n_expr": x["n_expr"],
                                 "image_sha256": x["sha"]}))
    with open(P / "photo_screen.jsonl", "w") as f:
        for s in scr:
            f.write(json.dumps(s) + "\n")
    return {"pixmo_cnt": out}, {**c, "rows": len(out), "group": dict(Counter(it["extra"]["group"] for it in out)),
                                "count_hist": dict(sorted(Counter(it["extra"]["count"] for it in out).items())),
                                "top": Counter(it["extra"]["class"] for it in out).most_common(40),
                                "screen": {"min_side": min_side, "rule": f"dropped if colours (5-bit, 128 px) < {min_colours} AND top-10 colour share > {max_top10}"}}


BUILDERS = {"fhm": b_fhm, "chartqa_h": b_chartqa_h, "hs_cnt": b_hs_cnt, "pixmo": b_pixmo}
STEPS = {"pixmo_plan": b_pixmo_plan, "pixmo_fetch": b_pixmo_fetch}

if __name__ == "__main__":
    name = sys.argv[1]
    if name in STEPS:
        STEPS[name]()
    else:
        res, summ = BUILDERS[name]()
        write(name, res, summ)
