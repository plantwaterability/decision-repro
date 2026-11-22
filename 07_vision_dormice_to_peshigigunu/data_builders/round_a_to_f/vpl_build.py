"""Private-like vision proxy (fork A, 11-10): board-format rows for the 8 private families we can source FRESH images for
(CV-Bench, BLINK, RealWorldQA, CharXiv, InfographicVQA, Mind2Web, KIE, Hateful Memes; Winoground has no fresh equivalent),
from sources that are in NO training pool and are not the benchmarks' own remaining items. Item formats / option counts
mirror the kit's public suite; kit helpers (convert.py) are reused. Deduped (sha256 + pHash <= 6) against the public images,
every training pool (A, B, B3, C, D) and the dev set. usage: python vpl_build.py <build|rows|debug_sun> [family ...]"""
import hashlib
import io
import json
import math
import random
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/data/local/visdata/scripts")
import vb_build as B  # noqa: E402
from decision_index.vision.convert import (CORD_PRESENT_Q, NUM, SEED, Letters, arrange, draw_marks, fmt_dec, font,  # noqa: E402
                                           numeric_distractors, overlaps, parse_amount, pick_by_rank)

PV = Path("/data/shared/visdata_priv")
DL = PV / "dl"
B.IMG = PV / "img_priv"
B.ITEMS = PV / "items_priv"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
LET = "ABCDEFGHIJ"
LIC_EVAL = "eval-only private-like proxy (NC fine in evaluation); never a training source"
B.LIC.update({k: (fam, "eval", f"{src}; {LIC_EVAL}") for k, fam, src in [
    ("pl_ade_count", "CV-Bench", "ADE20K train shard 3 (1aurent/ADE20K@bbf21e26; ADE20K terms, non-commercial research) - Count; board uses ADE20K val + COCO"),
    ("pl_ade_rel", "CV-Bench", "ADE20K train shard 3 (1aurent/ADE20K@bbf21e26; ADE20K terms) - Relation"),
    ("pl_sun_dist", "CV-Bench", "SUN RGB-D 3D boxes (kcz358/SUNRGBD-3D@a5baef8e; SUN RGB-D research terms) - Distance, boxes projected by us"),
    ("pl_cb_count", "BLINK", "CountBenchQA (vikhyatk/CountBenchQA@76d60030; LAION web images) - Counting"),
    ("pl_whatsup", "BLINK", "What's Up controlled images (amitakamath2/whatsup_vlms@373b3870, MIT) - Spatial_Relation yes/no"),
    ("pl_sun_depth", "BLINK", "SUN RGB-D 3D boxes (kcz358/SUNRGBD-3D@a5baef8e) - Relative_Depth, two circled object centres"),
    ("pl_ade_corr", "BLINK", "ADE20K train shard 7 (1aurent/ADE20K@bbf21e26) - Visual_Correspondence, synthetic homography + photometric change"),
    ("pl_mme_rw", "RealWorldQA", "MME-RealWorld-Lite (yifanzhang114@0e378f64; driving / monitoring / OCR-in-the-wild; source-image terms NC) cut to 3 options"),
    ("pl_cmuseum", "CharXiv", "ChartMuseum test (lytang/ChartMuseum@462d46de, CC BY-SA 4.0), numeric answers + kit numeric_distractors"),
    ("pl_cqp", "CharXiv", "ChartQAPro test (ahmed-masry/ChartQAPro@e27c2874, MIT): native 4-option MCQ + numeric factoid"),
    ("pl_mcinfo", "InfographicVQA", "multi-chart-infographic-reasoning test (xychen-zh@50e542ff, MIT), kit same-image distractor rule"),
    ("pl_vwb", "Mind2Web", "VisualWebBench action_ground + element_ground (visualwebbench@9dace8ac, Apache-2.0), kit 4-box marks"),
    ("pl_sroie", "KIE (CORD+FUNSD)", "ICDAR2019-SROIE test receipts (jsdnrs@bffe40c2, CC BY 4.0), CORD-style field_value / field_present"),
    ("pl_m3h", "Moderation (Hateful Memes)", "Multi3Hate English memes, US-annotator majority label (MinhDucBui/Multi3Hate@e3692a90; research use)"),
]})
QUOTA = {"pl_ade_count": 160, "pl_ade_rel": 130, "pl_sun_dist": 150, "pl_cb_count": 120, "pl_whatsup": 120, "pl_sun_depth": 100, "pl_ade_corr": 100,
         "pl_mme_rw": 450, "pl_cmuseum": 200, "pl_cqp": 250, "pl_mcinfo": 400, "pl_vwb": 400, "pl_sroie": 450, "pl_m3h": 300}


def it(fam, sid, rel, q, opts, gold, sub, sv):
    return B.item(fam, sid, rel if isinstance(rel, list) else [rel], q, opts, gold, sub, sv)


def jpg(im, q=92):
    b = io.BytesIO()
    im.convert("RGB").save(b, "JPEG", quality=q)
    return b.getvalue()


def rgroups(path, columns):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        yield from pf.read_row_group(rg, columns=columns).to_pylist()


# ---------------------------------------------------------------- CV-Bench Count / Relation (ADE20K train)
THINGS = {k: (v.split("/") if "/" in v else [v, v + "s"]) for k, v in {
    "person": "person/people", "chair": "chair", "car": "car", "windowpane": "window", "table": "table", "lamp": "lamp", "cushion": "cushion",
    "bottle": "bottle", "painting": "painting", "vase": "vase", "pillow": "pillow", "plate": "plate", "stool": "stool", "armchair": "armchair",
    "sofa": "sofa", "bed": "bed", "door": "door", "boat": "boat", "pot": "pot", "bus": "bus/buses", "truck": "truck", "bicycle": "bicycle",
    "mirror": "mirror", "box": "box/boxes", "basket": "basket", "bench": "bench/benches", "van": "van", "flag": "flag", "streetlight": "streetlight",
    "desk": "desk", "sink": "sink", "toilet": "toilet", "television receiver": "television", "refrigerator": "refrigerator", "stove": "stove",
    "bathtub": "bathtub", "fireplace": "fireplace", "clock": "clock", "chandelier": "chandelier", "piano": "piano", "computer": "computer",
    "microwave": "microwave", "oven": "oven", "fan": "fan", "towel": "towel", "airplane": "airplane", "umbrella": "umbrella",
    # the two shards we hold are outdoor (nature landscape, urban): outdoor things with a clear extent
    "building": "building", "house": "house", "tree": "tree", "signboard": "sign", "bridge": "bridge", "fountain": "fountain", "tower": "tower",
    "pole": "pole", "rock": "rock", "sculpture": "sculpture", "bird": "bird", "dog": "dog", "horse": "horse/horses", "cow": "cow", "sheep": "sheep/sheep",
    "palm": "palm tree", "awning": "awning", "ashcan": "trash can", "traffic light": "traffic light", "ship": "ship", "minibike": "motorbike",
    "backpack": "backpack", "bag": "bag", "beacon": "lighthouse", "sun": "sun/suns", "pier": "pier", "tent": "tent", "kite": "kite", "balloon": "balloon",
    "windmill": "windmill", "statue": "statue", "booth": "booth/booths", "hovel": "hut", "column": "column", "animal": "animal"}.items()}


def b_ade():
    from PIL import Image

    svc, svr = B.Saver("pl_ade_count"), B.Saver("pl_ade_rel")
    rng = random.Random(SEED + 9101)
    letc, letr = Letters(rng), Letters(rng)
    cnt, rel = [], []
    want_c, want_r = int(QUOTA["pl_ade_count"] * 1.2), int(QUOTA["pl_ade_rel"] * 1.2)
    import itertools

    cols = ["image", "filename", "objects"]
    a3 = list(rgroups(DL / "1aurent__ADE20K/data/train-00003-of-00010.parquet", ["filename"]))
    order = [(3, i) for i in range(len(a3))] + [(7, i) for i in range(0, 2557, 2)]  # odd rows of shard 7 belong to b_corr
    rng.shuffle(order)
    want = {3: {i for sh, i in order if sh == 3}, 7: {i for sh, i in order if sh == 7}}
    its = [((sh, i), r) for sh in (7, 3) for i, r in enumerate(rgroups(DL / f"1aurent__ADE20K/data/train-0000{sh}-of-00010.parquet", cols)) if i in want[sh]]
    rng.shuffle(its)
    for _, row in its:
        if len(cnt) >= want_c and len(rel) >= want_r:
            break
        data = row["image"]["bytes"]
        W, H = Image.open(io.BytesIO(data)).size
        if min(W, H) < 256:
            continue
        objs = []
        for o in row["objects"]:
            if o["parts"]["part_level"] != 0 or not o["polygon"]["x"]:
                continue
            xs, ys = o["polygon"]["x"], o["polygon"]["y"]
            bb = [max(0, min(xs)), max(0, min(ys)), min(W, max(xs)), min(H, max(ys))]
            objs.append((o["name"].split(",")[0].strip(), bb, (bb[2] - bb[0]) * (bb[3] - bb[1]) / (W * H)))
        names = Counter(n for n, _, _ in objs)
        stem = row["filename"].rsplit(".", 1)[0]
        cc = [n for n, k in names.items() if n in THINGS and 2 <= k <= 7 and all(a >= 0.002 for m, _, a in objs if m == n)]
        uni = [(n, bb) for n, bb, a in objs if n in THINGS and names[n] == 1 and 0.008 <= a <= 0.5]
        pairs = []
        for i in range(len(uni)):
            for j in range(i + 1, len(uni)):
                (na, a), (nb, b) = uni[i], uni[j]
                if a[2] + 0.03 * W <= b[0] or b[2] + 0.03 * W <= a[0]:
                    pairs.append((na, a, nb, b, "lr"))
                elif a[3] + 0.03 * H <= b[1] or b[3] + 0.03 * H <= a[1]:
                    pairs.append((na, a, nb, b, "ab"))
        # one question per image; alternate so Count and Relation never share an image
        if cc and len(cnt) < want_c and (len(cnt) <= len(rel) * want_c / max(1, want_r) or not pairs):
            n = rng.choice(sorted(cc))
            g = names[n]
            k = rng.choices([4, 5, 6], weights=[63, 20, 17])[0]
            lo = max(0, g - rng.randint(0, k - 1))
            opts = [str(v) for v in range(lo, lo + k)]
            ds = [o for o in opts if o != str(g)]
            opts, gold = arrange(str(g), ds, letc.pick(k), rng)
            cnt.append(it("pl_ade_count", stem, svc.save(data, stem), f"How many {THINGS[n][1]} are in the image?", opts, gold, "Count", svc))
        elif pairs and len(rel) < want_r:
            na, a, nb, b, kind = rng.choice(pairs)
            pos = letr.pick(2)  # options are fixed-order as in CV-Bench; pick which object is asked about to balance gold
            first = (a[0] < b[0]) if kind == "lr" else (a[1] < b[1])  # a is left / above
            if first != (pos == 0):
                na, nb = nb, na
            opts = ["left", "right"] if kind == "lr" else ["above", "below"]
            q = (f"Considering the relative positions of the {THINGS[na][0]} and the {THINGS[nb][0]} in the image provided, "
                 f"where is the {THINGS[na][0]} located with respect to the {THINGS[nb][0]}?")
            rel.append(it("pl_ade_rel", stem, svr.save(data, stem), q, opts, LET[pos], "Relation", svr))
    return cnt, rel


# ---------------------------------------------------------------- SUN RGB-D: CV-Bench Distance + BLINK Relative_Depth
SUN_SKIP = {"wall", "floor", "ceiling", "", "unknown"}


class SunCal:
    """per-scene intrinsics, tilt and depth map from the full SUNRGBD.zip (the parquet has the 3D boxes but no calibration)."""

    def __init__(self):
        self.z = zipfile.ZipFile(DL / "ZHAO11564__SUNRGBD.zip/SUNRGBD.zip")
        self.ext, self.dep = {}, {}
        for n in sorted(self.z.namelist()):
            if n.endswith(".txt") and "/extrinsics/" in n:
                self.ext[n.split("/extrinsics/")[0]] = n  # sorted -> the latest file wins
            elif n.endswith(".png") and "/depth_bfx/" in n:
                self.dep.setdefault(n.split("/depth_bfx/")[0], n)

    def get(self, sp):
        import numpy as np

        try:
            K = np.array(self.z.read(sp + "/intrinsics.txt").split(), dtype=float).reshape(3, 3)
            E = np.array(self.z.read(self.ext[sp]).split(), dtype=float).reshape(3, 4)[:, :3]
        except (KeyError, ValueError):
            return None
        return K, E

    def depth(self, sp):
        import numpy as np
        from PIL import Image

        if sp not in self.dep:
            return None
        d = np.asarray(Image.open(io.BytesIO(self.z.read(self.dep[sp])))).astype(np.uint16)
        return ((d >> 3) | (d << 13)).astype(np.float32) / 1000.0  # SUN RGB-D depth encoding -> metres


def sun_objs(row, cal, mode="Et", yawsign=1.0):
    """-> objects with 2D box / centre from the 3D box (gravity-aligned camera axes: x right, y down, z forward) un-tilted into the camera."""
    import numpy as np

    W, H = row["image_width"], row["image_height"]
    kc = cal.get(row["scene_path"])
    if kc is None:
        return []
    K, E = kc
    M = E.T if mode == "Et" else E
    out = []
    for lab, bx in zip(row["labels"], row["bboxes_3d"]):
        x, y, z, dx, dy, dz, _, _, yaw = bx
        c, s = math.cos(yawsign * yaw), math.sin(yawsign * yaw)
        P = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    px, py, pz = sx * dx / 2, sy * dy / 2, sz * dz / 2
                    P.append((x + c * px + s * pz, y + py, z - s * px + c * pz))
        P = np.array(P + [(x, y, z)]) @ M.T
        if P[:, 2].min() < 0.15:
            continue
        us, vs = K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2], K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]
        full = [us[:8].min(), vs[:8].min(), us[:8].max(), vs[:8].max()]
        bb = [max(0, full[0]), max(0, full[1]), min(W, full[2]), min(H, full[3])]
        fa = (full[2] - full[0]) * (full[3] - full[1])
        ia = max(0, bb[2] - bb[0]) * max(0, bb[3] - bb[1])
        out.append({"label": lab, "c": (x, y, z), "bb": bb, "inside": ia / fa if fa > 0 else 0, "area": ia / (W * H), "uv": (float(us[8]), float(vs[8])),
                    "zc": float(P[8, 2]), "ext": max(dx, dy, dz)})
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    return ix / ua if ua > 0 else 0


def sun_distance_image(im, boxes):
    from PIL import ImageDraw

    im = im.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    lw = max(3, min(im.size) // 140)
    for b, col in zip(boxes, ["#ff0000", "#0000ff", "#00c000"]):
        d.rectangle([int(v) for v in b], outline=col, width=lw)
    return im


def circle_points(im, pts, labels, color="#ff0000"):
    from PIL import ImageDraw

    im = im.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    W, H = im.size
    r = max(8, min(W, H) // 48)
    f = font(int(r * 2.2), FONT)
    for (u, v), lab in zip(pts, labels):
        d.ellipse([u - r, v - r, u + r, v + r], outline=color, width=max(3, r // 3))
        tw = d.textlength(lab, font=f)
        tx = u + r + 4 if u + r + 6 + tw < W else u - r - 4 - tw
        ty = min(max(0, v - r * 1.1), H - r * 2.4)
        d.text((tx, ty), lab, fill=color, font=f, stroke_width=2, stroke_fill="white")
    return im


def depth_at(D, u, v, k=3):
    import numpy as np

    w = D[max(0, int(v) - k):int(v) + k + 1, max(0, int(u) - k):int(u) + k + 1]
    w = w[(w > 0.3) & (w < 7.9)]
    return float(np.median(w)) if w.size >= (2 * k + 1) ** 2 // 2 else None


def b_sun(debug=0):
    from PIL import Image

    svd, svp = B.Saver("pl_sun_dist"), B.Saver("pl_sun_depth")
    rng = random.Random(SEED + 9102)
    letd, letp = Letters(rng), Letters(rng)
    cal = SunCal()
    dist, dep = [], []
    want_d, want_p = int(QUOTA["pl_sun_dist"] * 1.25), int(QUOTA["pl_sun_depth"] * 1.25)
    if debug:
        want_d = want_p = debug
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(DL / "kcz358__SUNRGBD-3D/data/test-00000-of-00001.parquet")
    for rg in range(pf.num_row_groups):
        rows = pf.read_row_group(rg).to_pylist()
        rng.shuffle(rows)
        for row in rows[:: 1 if debug else 3]:  # 10k frames, many near-duplicate views of one room: thin them
            if len(dist) >= want_d and len(dep) >= want_p:
                return dist, dep
            objs = [o for o in sun_objs(row, cal) if o["label"].lower() not in SUN_SKIP]
            names = Counter(o["label"] for o in objs)
            W, H = row["image_width"], row["image_height"]
            stem = row["scene_id"]
            good = [o for o in objs if names[o["label"]] == 1 and o["inside"] >= 0.85 and 0.008 <= o["area"] <= 0.4]
            trip = []
            for r in good:
                for a in good:
                    for b in good:
                        if len({id(r), id(a), id(b)}) < 3 or id(a) > id(b):
                            continue
                        da, db = math.dist(r["c"], a["c"]), math.dist(r["c"], b["c"])
                        if min(da, db) > 0.2 and max(da, db) / min(da, db) >= 1.3 and max(iou(r["bb"], a["bb"]), iou(r["bb"], b["bb"]), iou(a["bb"], b["bb"])) < 0.4:
                            trip.append((r, a, b, da, db))
            pairs = []
            if len(dep) < want_p and len(objs) >= 2:
                D = cal.depth(row["scene_path"])
                if D is not None and D.shape == (H, W):
                    m = 0.07
                    pts = []
                    for o in objs:  # the circled point must lie on the object: measured depth at the projected centre agrees with the box
                        u, v = o["uv"]
                        if m * W <= u <= (1 - m) * W and m * H <= v <= (1 - m) * H:
                            d = depth_at(D, u, v)
                            if d is not None and o["zc"] - o["ext"] / 2 - 0.25 <= d <= o["zc"] + 0.25:
                                pts.append((o, d))
                    for a, da in pts:
                        for b, db in pts:
                            if da < db and db / da >= 1.3 and db - da >= 0.5 and math.hypot(a["uv"][0] - b["uv"][0], a["uv"][1] - b["uv"][1]) >= 0.18 * min(W, H):
                                pairs.append((a, b))  # a = near; the label comes from the measured depth map
            if trip and len(dist) < want_d and (len(dist) <= len(dep) * want_d / max(1, want_p) or not pairs):
                r, a, b, da, db = rng.choice(trip)
                pos = letd.pick(2)
                if (da < db) != (pos == 0):
                    a, b, da, db = b, a, db, da
                im = Image.open(io.BytesIO(row["image"]["bytes"]))
                nm = lambda o: o["label"].replace("_", " ")  # noqa: E731
                q = ("Estimate the real-world distances between objects in this image. Which object is closer to the "
                     f"{nm(r)} (highlighted by a red box), the {nm(a)} (highlighted by a blue box) or the {nm(b)} (highlighted by a green box)?")
                rel = svd.save(jpg(sun_distance_image(im, [r["bb"], a["bb"], b["bb"]])), stem)
                dist.append(it("pl_sun_dist", stem, rel, q, [nm(a), nm(b)], LET[pos], "Distance", svd))
            elif pairs and len(dep) < want_p:
                a, b = rng.choice(pairs)
                pos = letp.pick(2)  # A is closer / B is closer
                pts = [a["uv"], b["uv"]] if pos == 0 else [b["uv"], a["uv"]]
                im = Image.open(io.BytesIO(row["image"]["bytes"]))
                rel = svp.save(jpg(circle_points(im, pts, ["A", "B"])), stem)
                q = "Two points are circled on the image, labeled by A and B beside each circle. Which point is closer to the camera?"
                dep.append(it("pl_sun_depth", stem, rel, q, ["A is closer", "B is closer"], LET[pos], "Relative_Depth", svp))
    return dist, dep


# ---------------------------------------------------------------- BLINK Visual_Correspondence (synthetic, ADE20K shard 7)
CORR_Q = ("A point is circled on the first image, labeled with REF. We change the camera position or lighting and shoot the second image. "
          "You are given multiple red-circled points on the second image, choices of \"A, B, C, D\" are drawn beside each circle. "
          "Which point on the second image corresponds to the point in the first image? Select from the following options.\n"
          "(A) Point A\n(B) Point B\n(C) Point C\n(D) Point D")


def b_corr():
    import numpy as np
    from PIL import Image, ImageEnhance

    sv = B.Saver("pl_ade_corr")
    rng = random.Random(SEED + 9103)
    let = Letters(rng)
    out = []
    want = int(QUOTA["pl_ade_corr"] * 1.25)
    for i, row in enumerate(rgroups(DL / "1aurent__ADE20K/data/train-00007-of-00010.parquet", ["image", "filename"])):
        if len(out) >= want:
            break
        if i % 2 == 0 or rng.random() < 0.5:  # even rows belong to b_ade
            continue
        im = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        if min(im.size) < 300:
            continue
        if max(im.size) > 1024:
            s = 1024 / max(im.size)
            im = im.resize((int(im.size[0] * s), int(im.size[1] * s)), Image.LANCZOS)
        W, H = im.size
        g = np.asarray(im.convert("L"), dtype=np.float64)
        gy, gx = np.gradient(g)
        cand = []
        for _ in range(500):
            u, v = rng.randint(int(0.2 * W), int(0.8 * W)), rng.randint(int(0.2 * H), int(0.8 * H))
            px, py = gx[v - 6:v + 7, u - 6:u + 7], gy[v - 6:v + 7, u - 6:u + 7]
            cand.append((min(float((px ** 2).mean()), float((py ** 2).mean())), u, v))  # corner-like: strong gradients in both directions
        cand.sort(reverse=True)
        # second image: inner quad stretched to the full frame (no borders) + brightness / contrast change
        src = [(rng.uniform(0.02, 0.14) * W, rng.uniform(0.02, 0.14) * H), (W - rng.uniform(0.02, 0.14) * W, rng.uniform(0.02, 0.14) * H),
               (W - rng.uniform(0.02, 0.14) * W, H - rng.uniform(0.02, 0.14) * H), (rng.uniform(0.02, 0.14) * W, H - rng.uniform(0.02, 0.14) * H)]
        dst = [(0, 0), (W, 0), (W, H), (0, H)]
        A = []
        for (x, y), (u, v) in zip(dst, src):
            A += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        co = np.linalg.solve(np.array(A, dtype=np.float64), np.array(src, dtype=np.float64).reshape(8))
        Hinv = np.linalg.inv(np.array([[co[0], co[1], co[2]], [co[3], co[4], co[5]], [co[6], co[7], 1.0]]))

        def fwd(u, v):
            p = Hinv @ np.array([u, v, 1.0])
            return float(p[0] / p[2]), float(p[1] / p[2])

        picked = []
        sep = 0.13 * min(W, H)
        for _, u, v in cand[:120]:
            u2, v2 = fwd(u, v)
            if not (0.06 * W <= u2 <= 0.94 * W and 0.06 * H <= v2 <= 0.94 * H):
                continue
            if all(math.hypot(u2 - p[2], v2 - p[3]) >= sep for p in picked):
                picked.append((u, v, u2, v2))
            if len(picked) == 4:
                break
        if len(picked) < 4:
            continue
        im2 = im.transform((W, H), Image.PERSPECTIVE, tuple(co), Image.BICUBIC)
        im2 = ImageEnhance.Brightness(im2).enhance(rng.uniform(0.6, 1.4))
        im2 = ImageEnhance.Contrast(im2).enhance(rng.uniform(0.7, 1.3))
        pos = let.pick(4)
        order = picked[1:]
        rng.shuffle(order)
        order = order[:pos] + [picked[0]] + order[pos:]
        stem = row["filename"].rsplit(".", 1)[0]
        r1 = sv.save(jpg(circle_points(im, [picked[0][:2]], ["REF"])), stem + "_1")
        r2 = sv.save(jpg(circle_points(im2, [p[2:] for p in order], list("ABCD"))), stem + "_2")
        out.append(it("pl_ade_corr", stem, [r1, r2], CORR_Q, [f"Point {l}" for l in "ABCD"], LET[pos], "Visual_Correspondence", sv))
    return out


# ---------------------------------------------------------------- BLINK Counting (CountBenchQA) / Spatial_Relation (What's Up)
def b_countbench():
    sv = B.Saver("pl_cb_count")
    rng = random.Random(SEED + 9104)
    let = Letters(rng)
    rows = list(enumerate(rgroups(DL / "vikhyatk__CountBenchQA/data/test-00000-of-00001.parquet", ["image", "question", "number"])))
    rng.shuffle(rows)
    out = []
    for i, r in rows[: int(QUOTA["pl_cb_count"] * 1.3)]:
        g = str(r["number"])
        opts, gold = arrange(g, numeric_distractors(g, [], rng), let.pick(4), rng)
        out.append(it("pl_cb_count", f"cb{i:04d}", sv.save(r["image"]["bytes"], f"cb{i:04d}"), r["question"], opts, gold, "Counting", sv))
    return out


def b_whatsup():
    sv = B.Saver("pl_whatsup")
    rng = random.Random(SEED + 9105)
    let = Letters(rng)
    PH = {"left_of": "to the left of", "right_of": "to the right of", "on": "on", "under": "under"}
    files = sorted((DL / "amitakamath2__whatsup_vlms/controlled_images").glob("*.jpeg"))
    rng.shuffle(files)
    out = []
    for p in files:
        m = re.fullmatch(r"(.+)_(left_of|right_of|on|under)_(.+)", p.stem)
        if not m:
            continue
        obj, relk, furn = m.group(1).replace("-", " "), m.group(2), m.group(3).replace("-", " ")
        pos = let.pick(2)  # yes / no
        ask = relk if pos == 0 else rng.choice([k for k in PH if k != relk])
        out.append(it("pl_whatsup", p.stem, sv.save(p.read_bytes(), p.stem), f"Is the {obj} {PH[ask]} the {furn}?", ["yes", "no"], LET[pos],
                      "Spatial_Relation", sv))
        if len(out) >= int(QUOTA["pl_whatsup"] * 1.3):
            break
    return out


# ---------------------------------------------------------------- RealWorldQA (MME-RealWorld-Lite, 3 options)
def b_mme():
    sv = B.Saver("pl_mme_rw")
    rng = random.Random(SEED + 9106)
    let = Letters(rng)
    zf = zipfile.ZipFile(DL / "yifanzhang114__MME-RealWorld-Lite/data.zip")
    J = json.loads(zf.read("data/MME-RealWorld-Lite.json"))
    per = {"Autonomous_Driving": 17, "Monitoring": 14, "OCR with Complex Context": 16}
    by = {}
    for r in J:
        if r["Subtask"] in per:
            by.setdefault((r["Subtask"], r["Category"].lower()), []).append(r)
    out = []
    for (st, cat), lst in sorted(by.items()):
        rng.shuffle(lst)
        n = 0
        for r in lst:
            if n >= per[st]:
                break
            ch = {}
            for c in r["Answer choices"]:
                m = re.match(r"\(([A-F])\)\s*(.*)", c.strip(), re.S)
                if m:
                    ch[m.group(1)] = m.group(2).strip()
            g = r["Ground truth"].strip()
            if g not in "ABCD" or g not in ch or len(ch) < 5:
                continue
            ds = [ch[l] for l in "ABCD" if l != g and l in ch and ch[l] and ch[l] != ch[g]]
            ds = list(dict.fromkeys(ds))
            if len(ds) < 2:
                continue
            opts, gold = arrange(ch[g], rng.sample(ds, 2), let.pick(3), rng)
            sid = re.sub(r"[^A-Za-z0-9_.-]", "_", r["Question_id"])
            stem = hashlib.sha1(r["Image"].encode()).hexdigest()[:16]
            try:
                data = zf.read("data/imgs/" + r["Image"])
            except KeyError:
                continue
            out.append(it("pl_mme_rw", sid, sv.save(data, stem), r["Text"].strip(), opts, gold, "mc", sv))
            n += 1
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------- CharXiv (ChartMuseum numeric, ChartQAPro MCQ + numeric)
def b_charts():
    svm, svq = B.Saver("pl_cmuseum"), B.Saver("pl_cqp")
    rng = random.Random(SEED + 9107)
    letm, letq = Letters(rng), Letters(rng)
    cm = []
    root = DL / "lytang__ChartMuseum"
    rows = list(rgroups(next((root / "data").glob("test-*.parquet")), None))
    rng.shuffle(rows)
    for r in rows:
        g = str(r["answer"]).strip()
        p = root / r["image"]
        if not NUM.match(g) or not p.exists():
            continue
        ds = numeric_distractors(g, [], rng)
        if len(set(ds + [g])) != 4:
            continue
        opts, gold = arrange(g, ds, letm.pick(4), rng)
        sid = r["hash"].replace("chartmuseum_", "")[:20]
        cm.append(it("pl_cmuseum", sid, svm.save(p.read_bytes(), re.sub(r"[^A-Za-z0-9_.-]", "_", p.stem)[:60] + "_" + sid[:6]), r["question"].strip(), opts, gold,
                     "number-in-chart", svm))
        if len(cm) >= int(QUOTA["pl_cmuseum"] * 1.15):
            break
    mcq, numq = [], []
    for i, r in enumerate(rgroups(DL / "ahmed-masry__ChartQAPro/data/test-00000-of-00001.parquet", None)):
        qs, ans = list(r["Question"]), list(r["Answer"])
        if len(qs) != 1 or len(ans) != 1:
            continue
        q, a = qs[0].strip(), str(ans[0]).strip()
        if r["Question Type"] == "Multi Choice":
            parts = re.split(r"\n\s*([a-dA-D])[).]\s*", "\n" + q if re.match(r"\s*[aA][).]", q) else q)
            if len(parts) != 9 or [x.lower() for x in parts[1::2]] != list("abcd") or a.upper() not in list("ABCD"):
                continue
            opts = [x.strip() for x in parts[2::2]]
            if len(set(opts)) != 4 or not parts[0].strip():
                continue
            mcq.append((i, r, parts[0].strip(), opts, a.upper(), "text-in-chart"))
        elif r["Question Type"] == "Factoid" and NUM.match(a):
            numq.append((i, r, q, None, a, "number-in-general"))
    rng.shuffle(mcq)
    rng.shuffle(numq)
    cq = []
    for i, r, q, opts, a, sub in mcq[:150] + numq[:140]:
        if opts is None:
            ds = numeric_distractors(a, [], rng)
            if len(set(ds + [a])) != 4:
                continue
            opts, gold = arrange(a, ds, letq.pick(4), rng)
        else:
            gold = a
        img = r["image"] if isinstance(r["image"], (bytes, bytearray)) else r["image"]["bytes"]
        cq.append(it("pl_cqp", f"cqp{i:04d}", svq.save(img, f"cqp{i:04d}"), q, opts, gold, sub, svq))
    rng.shuffle(cq)
    return cm, cq


# ---------------------------------------------------------------- InfographicVQA (multi-chart infographics, kit distractor rule)
def b_info():
    d = DL / "xychen-zh__multi-chart-infographic-reasoning"
    zf = zipfile.ZipFile(d / "infographics.zip")
    names = set(zf.namelist())
    metas, paths = [], {}
    for i, r in enumerate(rgroups(d / "test.parquet", None)):
        if r["image_path"] not in names:
            continue
        key = f"mc{i:04d}"
        paths[key] = r["image_path"]
        for j in range(1, 7):
            q, a = (r.get(f"sub_question {j}") or "").strip(), (r.get(f"answer {j}") or "").strip()
            if not q or not a or a.lower() == "none" or not re.search(r"[A-Za-z0-9]", a) or re.search(r"(?i)sub-?question", q) or len(a) > 60:
                continue
            metas.append({"qid": f"{key}_{j}", "question": q, "answers": [a], "img_key": key})
    return B.infovqa_like("pl_mcinfo", metas, lambda m: zf.read(paths[m["img_key"]]), 1, 9108)


# ---------------------------------------------------------------- Mind2Web (VisualWebBench grounding, kit 4-box marks)
def b_vwb():
    from PIL import Image

    sv = B.Saver("pl_vwb")
    rng = random.Random(SEED + 9109)
    let = Letters(rng)
    out = []
    VH = 1080
    for split, cap in (("action_ground", 200), ("element_ground", 330)):
        n = 0
        for r in rgroups(DL / f"visualwebbench__VisualWebBench/{split}/test-00000-of-00001.parquet", None):
            if n >= cap:
                break
            task = r["instruction"].strip() if split == "action_ground" else f'Open "{" ".join(r["elem_desc"].split())}"'
            if len(task) > 160 or len(task) < 4:
                continue
            im = Image.open(io.BytesIO(r["raw_image"]["bytes"])).convert("RGB")
            W, H = im.size
            px = [[b[0] * W, b[1] * H, b[2] * W, b[3] * H] for b in r["options"]]
            g = px[int(r["answer"])]
            vh = min(H, VH)
            cy = (g[1] + g[3]) / 2
            top = max(0, int(min(max(0, cy - vh / 2 + rng.uniform(-200, 200)), H - vh)))
            if not (g[1] >= top and g[3] <= top + vh) or g[2] - g[0] < 4 or g[3] - g[1] < 4:
                continue
            negs = []
            for k, b in enumerate(px):
                if k == int(r["answer"]) or b[1] < top or b[3] > top + vh or b[2] - b[0] < 8 or b[3] - b[1] < 8:
                    continue
                if overlaps(b, g) or any(overlaps(b, x) for x in negs):
                    continue
                negs.append(b)
            if len(negs) < 3:
                continue
            pos = let.pick(4)
            ds = rng.sample(negs, 3)
            boxes = ds[:pos] + [g] + ds[pos:]
            crop = im.crop((0, top, W, top + vh))
            marked = draw_marks(crop, [[b[0], b[1] - top, b[2], b[3] - top] for b in boxes], FONT)
            q = f"Task: {task}\nPrevious actions:\n- (none)\nWhich marked element should the next action (CLICK) be performed on?"
            out.append(it("pl_vwb", r["id"], sv.save(jpg(marked), r["id"] + "_marked"), q, [f"box {LET[i]}" for i in range(4)], LET[pos], split, sv))
            n += 1
    return out


# ---------------------------------------------------------------- KIE (SROIE receipts, CORD-style)
def b_sroie():
    sv = B.Saver("pl_sroie")
    rng = random.Random(SEED + 9110)
    let = Letters(rng)
    PRES = {"subtotal_price": (r"\bSUB[\s-]?TOTAL\b", r"SUB"), "service_price": (r"SERVICE\s+CHARGE|SERV(\.|ICE)?\s*CHG|\bSVC\b|\bSVR?\s*CHR?G", r"SERV|SVC|SVR"),
            "changeprice": (r"\bCHANGE\b", r"CHANGE|CHG")}
    val, pres = [], {k: {"Yes": [], "No": []} for k in PRES}
    for r in rgroups(DL / "jsdnrs__ICDAR2019-SROIE/data/test-00000-of-00001.parquet", None):
        key = r["key"]
        words = [w for w in r["words"] if w]
        gv = parse_amount((r["entities"] or {}).get("total"))
        amounts = {}
        for w in words:
            for m in re.findall(r"\d[\d,]*\.\d{2}", w):
                v = parse_amount(m)
                if v:
                    amounts[round(v, 2)] = v
        if gv and round(gv, 2) in amounts:
            pool = [(v, fmt_dec(v)) for k, v in amounts.items() if abs(v - gv) > 0.005]
            ds = pick_by_rank(gv, pool, 3, rng) if len(pool) >= 2 else None
            if ds and len(ds) >= 2:
                opts, gold = arrange(fmt_dec(gv), [t for _, t in ds], let.pick(len(ds) + 1), rng)
                val.append((key, r, "What is the total amount on this receipt?", opts, gold, "field_value"))
        text = "\n".join(words).upper()
        for k, (yes, anyhint) in PRES.items():
            if re.search(yes, text):
                pres[k]["Yes"].append((key, r))
            elif not re.search(anyhint, text):
                pres[k]["No"].append((key, r))
    out = []
    for key, r, q, opts, gold, sub in val:
        out.append(it("pl_sroie", f"{key}_total", sv.save(r["image"]["bytes"], key), q, opts, gold, sub, sv))
    pr = []
    for k, d in pres.items():  # kit rule: yes/no balanced per field
        n = min(len(d["Yes"]), len(d["No"]), 30)
        for lab in ("Yes", "No"):
            rng.shuffle(d[lab])
            for key, r in d[lab][:n]:
                pr.append(it("pl_sroie", f"{key}_{k}", sv.save(r["image"]["bytes"], key), f"Does this receipt show {CORD_PRESENT_Q[k]}?", ["Yes", "No"],
                             "A" if lab == "Yes" else "B", "field_present", sv))
    rng.shuffle(out)
    rng.shuffle(pr)
    out = out[: QUOTA["pl_sroie"] - len(pr) + 30] + pr
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------- Hateful Memes (Multi3Hate English)
def b_m3h():
    sv = B.Saver("pl_m3h")
    out = []
    for r in rgroups(DL / "MinhDucBui__Multi3Hate/data/train-00000-of-00001.parquet", None):
        if r["Language"] != "en" or r["US"] is None:
            continue
        sid = f"m{int(r['Meme ID']):04d}"
        out.append(it("pl_m3h", sid, sv.save(r["image"]["bytes"], sid), "Is this meme hateful?", ["Yes", "No"], "A" if r["US"] >= 0.5 else "B", "binary", sv))
    random.Random(SEED + 9111).shuffle(out)
    return out


BUILD = {"ade": (b_ade, ["pl_ade_count", "pl_ade_rel"]), "sun": (b_sun, ["pl_sun_dist", "pl_sun_depth"]), "corr": (b_corr, ["pl_ade_corr"]),
         "countbench": (b_countbench, ["pl_cb_count"]), "whatsup": (b_whatsup, ["pl_whatsup"]), "mme": (b_mme, ["pl_mme_rw"]),
         "charts": (b_charts, ["pl_cmuseum", "pl_cqp"]), "info": (b_info, ["pl_mcinfo"]), "vwb": (b_vwb, ["pl_vwb"]), "sroie": (b_sroie, ["pl_sroie"]),
         "m3h": (b_m3h, ["pl_m3h"])}


def build(which=None):
    B.ITEMS.mkdir(parents=True, exist_ok=True)
    for name in which or list(BUILD):
        fn, keys = BUILD[name]
        try:
            res = fn()
        except Exception as e:  # one failing source must not sink the rest; the failure is printed and the family is reported short
            import traceback

            traceback.print_exc()
            print(json.dumps({"builder": name, "FAILED": repr(e)[:300]}), flush=True)
            continue
        res = res if isinstance(res, tuple) else (res,)
        for k, v in zip(keys, res):
            with open(B.ITEMS / f"{k}.jsonl", "w") as f:
                for x in v:
                    f.write(json.dumps(x) + "\n")
            print(json.dumps({"source": k, "items": len(v), "subtasks": dict(Counter(x["subtask"] for x in v)), "gold": dict(Counter(x["gold"] for x in v)),
                              "nopts": dict(Counter(len(x["options"]) for x in v))}), flush=True)
    print("BUILD_DONE", flush=True)


def rows():
    """Dedupe vs public images, every training pool and the dev set; per-source quotas; kit rows (sha-pinned) + image exclusion list."""
    import numpy as np
    from vb_manifest import PH_T, public_hashes

    pub = public_hashes()
    sha = {r["sha256"] for r in pub}
    phs = [int(r["phash"], 16) for r in pub if r["phash"]]
    pools = {"A": "/data/local/visdata/items", "B": "/data/local/visdata/items_b", "B3": "/data/local/visdata/items_b3",
             "C": "/data/shared/visdata_c/items_c", "D": "/data/shared/visdata_d/items_d", "dev": "/data/shared/visdata_dev/items_dev"}
    n_pool = Counter()
    for tag, d in pools.items():
        for f in sorted(Path(d).glob("*.jsonl")):
            for l in open(f):
                r = json.loads(l)
                sha.update(r["img_src_sha256"])
                sha.update(r["img_sha256"])
                phs += [int(p, 16) for p in r["img_phash"]]
                n_pool[tag] += 1
    PH = np.array(sorted(set(phs)), dtype=np.uint64)
    drops, kept, built = Counter(), {}, []
    for f in sorted(B.ITEMS.glob("*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            built.append(r)
            bad = any(s1 in sha or s2 in sha or int(np.bitwise_count(PH ^ np.uint64(int(p, 16))).min()) <= PH_T
                      for s1, s2, p in zip(r["img_src_sha256"], r["img_sha256"], r["img_phash"]))
            if bad:
                drops[r["source"]] += 1
                continue
            kept.setdefault(r["source"], []).append(r)
    out, excl = [], []
    for src in sorted(kept):
        for r in kept[src][: QUOTA[src]]:
            k = len(r["options"])
            crit = {LET[i]: str(o) for i, o in enumerate(r["options"])}
            assert r["gold"] in LET[:k]
            out.append({"id": r["id"], "benchmark": r["family"], "set": r["source"], "subtask": r["subtask"], "images": r["images"], "state": {},
                        "questions": {"answer": {"type": "choice", "instructions": r["question"], "criteria": crit}},
                        "prompt": r.get("prompt") or {"options_in_image": False, "readout": None}, "expected": {"answer": r["gold"]},
                        "_evaluation": {"run_id": r["id"], "benchmark": r["family"], "set": r["source"]}, "licence": r["licence"]})
    out.sort(key=lambda r: (r["benchmark"], r["set"], r["id"]))
    rows_path = PV / "priv_rows.jsonl"
    with open(rows_path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    used = {r["id"] for r in out}
    for r in built:  # every image we built, in the proxy or not (dedupe drops and over-quota spares included): all of it stays out of training
        for p, s1, s2, ph in zip(r["images"], r["img_src_sha256"], r["img_sha256"], r["img_phash"]):
            excl.append({"row_id": r["id"], "family": r["family"], "source": r["source"], "path": p, "src_sha256": s1, "sha256": s2, "phash": ph, "in_proxy": r["id"] in used})
    (PV / "priv_images.txt").write_text("".join(sorted({p + "\n" for r in out for p in r["images"]})))
    seen, ex2 = set(), []
    for e in excl:
        if e["path"] not in seen:
            seen.add(e["path"])
            ex2.append(e)
    with open(PV / "privlike_image_exclusion.jsonl", "w") as f:
        for e in ex2:
            f.write(json.dumps(e) + "\n")
    hdr = {"rows": len(out), "rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(), "per_family": dict(Counter(r["benchmark"] for r in out)),
           "per_source": dict(Counter(r["set"] for r in out)), "per_subtask": {f"{k[0]}|{k[1]}": v for k, v in Counter((r["benchmark"], r["subtask"]) for r in out).items()},
           "dedupe_drops": dict(drops), "dedupe_against": {"public_images": len(pub), **{f"pool_{k}_items": v for k, v in n_pool.items()}}, "phash_threshold": PH_T,
           "exclusion_images": len(ex2), "exclusion_sha256": hashlib.sha256((PV / "privlike_image_exclusion.jsonl").read_bytes()).hexdigest(),
           "image_root": str(B.IMG), "sources": {k: v[2] for k, v in B.LIC.items() if k.startswith("pl_")}}
    (PV / "priv_rows.header.json").write_text(json.dumps(hdr, indent=1))
    print(json.dumps(hdr))
    print("ROWS_DONE", flush=True)


def debug_sun():
    """contact sheet: every annotated box of 4 scenes under (E^T | E) x (yaw + | -), to settle the projection by eye; then 4 built samples."""
    import base64
    import pyarrow.parquet as pq
    from PIL import Image, ImageDraw

    cal = SunCal()
    pf = pq.ParquetFile(DL / "kcz358__SUNRGBD-3D/data/test-00000-of-00001.parquet")
    rows = pf.read_row_group(0).to_pylist()
    pick = [r for r in rows if 3 <= len(r["labels"]) <= 9][:60:15] + [r for r in pf.read_row_group(3).to_pylist() if 3 <= len(r["labels"]) <= 9][:2]
    tiles = []
    for mode, ys in (("Et", 1.0), ("E", 1.0), ("Et", -1.0)):
        for r in pick[:5]:
            im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            d = ImageDraw.Draw(im)
            f = font(16, FONT)
            for k, o in enumerate(sun_objs(r, cal, mode, ys)):
                col = ["#ff0000", "#0000ff", "#00c000", "#ff00ff", "#ff8000", "#00c0c0", "#ffff00", "#8000ff", "#ffffff"][k % 9]
                if o["bb"][2] <= o["bb"][0] or o["bb"][3] <= o["bb"][1]:
                    continue
                d.rectangle([int(v) for v in o["bb"]], outline=col, width=3)
                d.text((o["bb"][0] + 3, o["bb"][1] + 2), o["label"], fill=col, font=f, stroke_width=2, stroke_fill="black")
            tiles.append(im.resize((480, 360)))
        print(mode, ys, [r["scene_path"] for r in pick[:5]])
    B.IMG = PV / "dbg_img"
    d, p = b_sun(debug=3)
    for x in (d + p)[:5]:
        tiles.append(Image.open(B.IMG / x["images"][0]).convert("RGB").resize((480, 360)))
        print(x["id"], x["question"][-130:], x["options"], x["gold"])
    sheet = Image.new("RGB", (480 * 5, 360 * 4))
    for i, t in enumerate(tiles[:20]):
        sheet.paste(t, (480 * (i % 5), 360 * (i // 5)))
    b = io.BytesIO()
    sheet.save(b, "JPEG", quality=60)
    (PV / "dbg_sun_sheet.jpg").write_bytes(b.getvalue())
    print("SHEET_B64 " + base64.b64encode(b.getvalue()).decode())


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(sys.argv[2:] or None)
    elif cmd == "rows":
        rows()
    elif cmd == "debug_sun":
        debug_sun()
