"""Round G: gqa_rw (clean) = RealWorldQA-style rows from GQA balanced TRAIN questions on Visual Genome photos (spatial / relational only).
usage: python vg_gqa.py plan    -> dl/gqa/plan.jsonl (questions chosen, options built, one row per image first)
       python vg_gqa.py fetch   -> dl/gqa/img/<vg id>.jpg from Visual Genome's own hosting (TLS verified; resumable)
       python vg_gqa.py build   -> items_g/gqa_rw.jsonl + img_g/gqa_rw/ (kit Saver: sha256 + pHash for the dedupe gate)
Rules: question text names a spatial relation or position AND its GQA type is a relation / position type; choose -> its two alternatives
(order as in the question); verify / exist -> [Yes, No] drawn 50 / 50; query -> gold + 2-3 distractors = answers of other questions with the
same opening words that name nothing annotated in this image's scene graph (so they are wrong here); position queries -> [left, right].
VG images that are COCO images are used only if the COCO id is in train2017; GQA val / testdev / test questions and images are never read."""
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
import vg_build as VG  # noqa: E402
from vg_build import B, FS, LETTERS  # noqa: E402

D = VG.DLG / "gqa"
SEED = VG.SEED
SP = re.compile(r"\b(left|right|above|below|behind|in front of|on top of|under|underneath|beneath|next to|beside|near|top|bottom)\b")
CHOOSE_T = {"relChooser", "positionChoose", "categoryRelOChoose"}
VERIFY_T = {"relVerify", "relVerifyCo", "relVerifyCr", "relVerifyCop", "positionVerify", "positionVerifyC", "existRelS", "existRelSC", "existRelSRC"}
QUERY_T = {"relS", "relO", "categoryRelS", "categoryRelO"}
SP_TYPES = CHOOSE_T | VERIFY_T | QUERY_T
YESNO = {"yes", "no"}
SYN = [{"couch", "sofa"}, {"man", "guy", "gentleman", "men"}, {"woman", "lady", "women"}, {"person", "people", "man", "woman", "guy", "lady", "boy", "girl",
       "child", "kid", "player", "skier", "surfer", "skateboarder", "snowboarder", "biker", "rider", "pedestrian", "tourist", "spectator", "catcher",
       "batter", "umpire"}, {"child", "kid", "boy", "girl", "children", "kids"}, {"cell phone", "phone", "cellphone", "mobile phone", "smartphone"},
       {"tv", "television", "monitor", "screen", "computer monitor"}, {"cup", "mug", "glass", "coffee cup"}, {"bike", "bicycle"},
       {"motorcycle", "motorbike", "scooter"}, {"plane", "airplane", "aircraft", "jet"}, {"car", "automobile", "taxi", "cab", "suv", "vehicle"},
       {"shirt", "t-shirt", "tee shirt", "tshirt", "polo shirt"}, {"cabinet", "cupboard", "cabinets"}, {"rug", "carpet", "mat"},
       {"pants", "jeans", "trousers", "shorts"}, {"hat", "cap", "helmet"}, {"purse", "handbag", "bag", "backpack", "tote bag"}, {"dog", "puppy"},
       {"cat", "kitten"}, {"sneakers", "shoes", "sneaker", "shoe", "boots", "boot"}, {"pillow", "cushion"}, {"lamp", "light", "lamp shade", "lampshade"},
       {"stone", "rock", "rocks", "stones"}, {"freezer", "refrigerator", "fridge"}, {"laptop", "computer", "notebook"}, {"remote", "controller",
       "remote control", "game controller"}, {"skateboard", "board", "surfboard", "snowboard"}, {"truck", "pickup", "pickup truck"},
       {"table", "desk", "coffee table", "dining table"}, {"chair", "seat", "stool", "armchair"}, {"bush", "bushes", "shrub", "plant", "plants",
       "tree", "trees"}, {"jacket", "coat"}, {"street sign", "sign", "signs"}, {"pole", "post"}, {"fence", "railing"}, {"vase", "pot", "flower pot"},
       {"bowl", "plate", "dish"}, {"sofa", "loveseat"}, {"donut", "doughnut"}, {"sandwich", "burger", "hamburger"}, {"ocean", "sea", "water"},
       {"hill", "mountain"}, {"road", "street"}, {"sidewalk", "pavement"}, {"cow", "cattle", "calf", "bull"}, {"horse", "pony"}, {"sheep", "lamb"},
       {"bird", "seagull", "pigeon", "duck"}, {"animal", "animals"}, {"bear", "teddy bear"}, {"keyboard", "computer keyboard"},
       {"mouse", "computer mouse"}, {"towel", "napkin", "paper towel"}, {"glasses", "eyeglasses", "sunglasses"}, {"drawer", "drawers"},
       {"shelf", "shelves", "bookshelf", "bookcase"}, {"bed", "mattress"}, {"wall", "walls"}, {"curtain", "curtains", "drapes", "blinds"}]


def syn(a, p):
    return any(a in g and p in g for g in SYN)


def iter_json_dict(s):
    dec = json.JSONDecoder()
    i = s.index("{") + 1
    while True:
        while s[i] in " \n\r\t,":
            i += 1
        if s[i] == "}":
            return
        k, i = dec.raw_decode(s, i)
        while s[i] in " \n\r\t:":
            i += 1
        v, i = dec.raw_decode(s, i)
        yield k, v


def words(q, k):
    return " ".join(re.findall(r"[a-z]+", q.lower())[:k])


def variants(n):
    n = n.lower().strip()
    out = {n}
    if n.endswith("es"):
        out.add(n[:-2])
    if n.endswith("s"):
        out.add(n[:-1])
    out.add(n + "s")
    out.add(n + "es")
    return out


def choose_alts(q, ans, op, arg):
    """the two alternatives of a GQA choose question from its program's last operation, answer-shaped, in the order the question says them"""
    if op == "choose rel":  # "woman,to the right of|to the left of,s (3296405)"
        parts = arg.split(",")
        if len(parts) < 3:
            return None
        alts = parts[1].split("|")
    elif op.startswith("choose"):  # "left|right", "top|bottom", "cell phone|speaker"
        alts = arg.split("|")
    else:
        return None
    if len(alts) != 2:
        return None

    def short(x):
        x = re.sub(r"^(to the|on the|in the)\s+", "", x.strip().lower())
        return re.sub(r"\s+of$", "", x)

    ans = ans.lower().strip()
    for forms in ([short(x) for x in alts], [x.strip().lower() for x in alts]):
        if ans in forms and forms[0] != forms[1]:
            ql = q.lower()
            pos = [ql.find(f) for f in forms]
            if min(pos) < 0 or len(max(forms, key=len)) > 30:
                return None
            return [f for _, f in sorted(zip(pos, forms))]
    return None


def plan(n=18000, share=None):
    share = share or {"choose": 0.34, "verify": 0.30, "query": 0.36}
    z = zipfile.ZipFile(D / "questions1.2.zip")
    s = z.read("train_balanced_questions.json").decode("utf-8")
    zi = zipfile.ZipFile(D / "vg_image_data.json.zip")
    im = {str(r["image_id"]): r for r in json.loads(zi.read(zi.namelist()[0]))}
    coco_train = {int(Path(x).stem) for x in zipfile.ZipFile(VG.V.DLE / "coco" / "train2017.zip").namelist() if x.endswith(".jpg")}
    c, cand = Counter(), []
    pref = defaultdict(list)
    for qid, v in iter_json_dict(s):
        q, a, t = v["question"].strip(), str(v["answer"]).strip().lower(), v["types"]
        pref[words(q, 4)].append(a)
        if not SP.search(q.lower()) or t["detailed"] not in SP_TYPES:
            continue
        iid = v["imageId"]
        r = im.get(iid)
        if r is None:
            c["image_not_in_vg_list"] += 1
            continue
        if r.get("coco_id") and int(r["coco_id"]) not in coco_train:
            c["coco_image_not_train2017"] += 1
            continue
        last = v["semantic"][-1] if v.get("semantic") else {"operation": "", "argument": ""}
        cand.append((qid, iid, q, a, t["structural"], t["detailed"], last["operation"], last["argument"]))
    del s
    pref = {k: list(dict.fromkeys(x for x in v if x not in YESNO and len(x) <= 30)) for k, v in pref.items()}
    c["spatial_candidates"] = len(cand)
    c["by_type"] = dict(Counter(x[5] for x in cand))
    zs = zipfile.ZipFile(D / "sceneGraphs.zip")
    sg = json.loads(zs.read("train_sceneGraphs.json"))
    names = {iid: {n for o in g["objects"].values() for n in variants(o["name"])} for iid, g in sg.items()}
    del sg
    rng = random.Random(f"{SEED}:gqa")
    cand.sort()
    rng.shuffle(cand)
    quota = {k: round(n * w) for k, w in share.items()}
    vcap = quota["verify"] / 2  # yes and no each at most half of the verify rows
    yn = Counter()
    gold_pos = Counter()
    got, per = [], Counter()
    for cap in (1, 2):  # one question per image first, a second only to fill
        for qid, iid, q, a, st, det, op, arg in cand:
            if len(got) >= n:
                break
            if per[iid] != cap - 1 or iid not in names:
                continue
            kind = "verify" if det in VERIFY_T else "choose" if det in CHOOSE_T else "query" if det in QUERY_T else None
            if kind is None or quota[kind] <= 0:
                continue
            rr = random.Random(f"{SEED}:gqa:{qid}")
            if kind == "verify":
                if a not in YESNO or yn[a] >= vcap:
                    continue
                opts, gold = ["Yes", "No"], "A" if a == "yes" else "B"
                yn[a] += 1
            elif kind == "choose":
                alts = choose_alts(q, a, op, arg)
                if not alts:
                    c["choose_unparsed"] += 1
                    continue
                gi = alts.index(a)
                if gold_pos[gi] > gold_pos[1 - gi] + 50:  # keep the gold position near 50 / 50
                    continue
                gold_pos[gi] += 1
                opts, gold = alts, LETTERS[gi]
            else:
                if a in ("left", "right"):
                    opts, gold = ["left", "right"], "A" if a == "left" else "B"
                else:
                    here = names[iid]
                    pool = [p for p in pref.get(words(q, 4), []) if p != a and not (variants(p) & here) and a not in p and p not in a and not syn(a, p)]
                    if len(pool) < 3:
                        pool = [p for p in pref.get(words(q, 3), []) if p != a and not (variants(p) & here) and a not in p and p not in a
                                and not syn(a, p)]
                    k = 4 if rr.random() < 0.5 else 3
                    if len(pool) < k - 1:
                        c["query_few_distractors"] += 1
                        continue
                    ds = rr.sample(pool, k - 1)
                    opts = ds + [a]
                    rr.shuffle(opts)
                    gold = LETTERS[opts.index(a)]
            quota[kind] -= 1
            per[iid] += 1
            got.append({"qid": qid, "image": iid, "url": im[iid]["url"], "question": q, "answer": a, "kind": kind, "type": det, "options": opts, "gold": gold})
    with open(D / "plan.jsonl", "w") as f:
        for x in got:
            f.write(json.dumps(x) + "\n")
    summ = {**c, "planned": len(got), "images": len(per), "kind": dict(Counter(x["kind"] for x in got)), "type": dict(Counter(x["type"] for x in got)),
            "yes_no": dict(yn), "choose_gold_pos": dict(gold_pos), "n_options": dict(Counter(len(x["options"]) for x in got)), "unfilled": {k: v for k, v in quota.items() if v > 0}}
    (D / "plan_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ), flush=True)
    print("G_GQA_PLAN_DONE", flush=True)


def fetch(workers=24):
    import hashlib
    import time
    from concurrent.futures import ThreadPoolExecutor

    import httpx

    P = D / "img"
    P.mkdir(parents=True, exist_ok=True)
    plan_ = [json.loads(l) for l in open(D / "plan.jsonl")]
    urls = {}
    for x in plan_:
        urls.setdefault(x["image"], x["url"])
    todo = [(i, u) for i, u in sorted(urls.items()) if not (P / f"{i}.jpg").exists()]
    print(json.dumps({"images": len(urls), "todo": len(todo)}), flush=True)
    cli = httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0), follow_redirects=True, headers=VG.UA)

    def one(iu):
        i, u = iu
        u = u.replace("http://", "https://")
        for k in range(3):
            try:
                r = cli.get(u)
                if r.status_code == 200 and r.content:
                    tmp = P / f"{i}.part"
                    tmp.write_bytes(r.content)
                    tmp.rename(P / f"{i}.jpg")
                    return "ok", hashlib.sha256(r.content).hexdigest()
                if r.status_code == 404:
                    return "http_404", None
            except Exception:
                time.sleep(2 + 3 * k)
        return "failed", None

    t0, st = time.time(), Counter()
    with ThreadPoolExecutor(workers) as ex:
        for j, (s_, _) in enumerate(ex.map(one, todo)):
            st[s_] += 1
            if (j + 1) % 2000 == 0:
                print(json.dumps({"fetched": j + 1, "secs": round(time.time() - t0), **st}), flush=True)
    print(json.dumps({"status": dict(st)}), flush=True)
    print("G_GQA_FETCH_DONE", flush=True)


def build():
    plan_ = [json.loads(l) for l in open(D / "plan.jsonl")]
    sv = FS("gqa_rw")
    out, c = [], Counter()
    for x in plan_:
        p = D / "img" / f"{x['image']}.jpg"
        if not p.exists():
            c["image_missing"] += 1
            continue
        try:
            rel = sv.save(p.read_bytes(), x["image"])
        except Exception:
            c["bad_image"] += 1
            continue
        sub = {"verify": "yesno", "choose": "choose", "query": "mc"}[x["kind"]]
        out.append(B.item("gqa_rw", x["qid"], [rel], x["question"], x["options"], x["gold"], sub, sv,
                          extra={"gqa_type": x["type"], "kind": x["kind"], "vg_image_id": x["image"]}))
    return {"gqa_rw": out}, {"planned": len(plan_), **c, "rows": len(out), "kind": dict(Counter(it["extra"]["kind"] for it in out)),
                             "gold": dict(Counter(it["gold"] for it in out))}


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "plan":
        plan()
    elif step == "fetch":
        fetch()
    else:
        res, summ = build()
        VG.write("gqa", res, summ)
