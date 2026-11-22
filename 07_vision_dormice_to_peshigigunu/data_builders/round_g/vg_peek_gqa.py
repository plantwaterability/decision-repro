"""peek at GQA balanced train questions (streamed) + scene graph format"""
import collections, json, re, sys, zipfile
D = "/data/shared/visdata_g/dl/gqa"
SP = re.compile(r"\b(left|right|above|below|behind|in front of|on top of|under|underneath|beneath|next to|beside|near|top|bottom)\b")
def iter_json_dict(s):
    dec = json.JSONDecoder(); i = s.index("{") + 1; n = len(s)
    while True:
        while s[i] in " \n\r\t,": i += 1
        if s[i] == "}": return
        k, i = dec.raw_decode(s, i)
        while s[i] in " \n\r\t:": i += 1
        v, i = dec.raw_decode(s, i)
        yield k, v
z = zipfile.ZipFile(f"{D}/questions1.2.zip")
print([(x.filename, x.file_size) for x in z.infolist()])
s = z.read("train_balanced_questions.json").decode("utf-8")
print("chars", len(s))
tot, sp, det, struct, ex = 0, 0, collections.Counter(), collections.Counter(), collections.defaultdict(list)
imgs, nimg = set(), collections.Counter()
for k, v in iter_json_dict(s):
    tot += 1
    imgs.add(v["imageId"])
    nimg["n" if v["imageId"].startswith("n") else "digit"] += 1
    if not SP.search(v["question"]):
        continue
    sp += 1
    t = v["types"]; det[t["detailed"]] += 1; struct[(t["structural"], t["semantic"])] += 1
    if len(ex[t["detailed"]]) < 3:
        ex[t["detailed"]].append((v["question"], v["answer"], v["semantic"][-1]["operation"], v["semantic"][-1]["argument"], v["imageId"]))
print("total", tot, "images", len(imgs), nimg, "spatial-word questions", sp)
print("structural/semantic", struct.most_common())
print("detailed", det.most_common())
for d, e in sorted(ex.items(), key=lambda kv: -det[kv[0]])[:40]:
    print(d, e)
del s
zs = zipfile.ZipFile(f"{D}/sceneGraphs.zip"); print([(x.filename, x.file_size) for x in zs.infolist()])
zi = zipfile.ZipFile(f"{D}/vg_image_data.json.zip"); print([(x.filename, x.file_size) for x in zi.infolist()])
im = json.loads(zi.read(zi.namelist()[0])); print(len(im), im[0], im[-1], sum(1 for r in im if r.get("coco_id")))
