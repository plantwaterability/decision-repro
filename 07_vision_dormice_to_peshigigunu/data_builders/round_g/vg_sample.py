"""contact sheet + text of k random rows of a round G source (round F's vf_sample.py). usage: vg_sample.py <source> [k] [seed] [items|lists] [filter key=value]"""
import base64, io, json, random, sys
from PIL import Image
G = "/data/shared/visdata_g/"
src, k, seed = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6, int(sys.argv[3]) if len(sys.argv) > 3 else 1
where = sys.argv[4] if len(sys.argv) > 4 else "items"
flt = sys.argv[5] if len(sys.argv) > 5 else None
rows = [json.loads(l) for l in open(G + (f"items_g/{src}.jsonl" if where == "items" else f"lists/{src}.jsonl"))]
if flt:
    fk, fv = flt.split("=", 1)
    rows = [r for r in rows if str((r.get("extra") or {}).get(fk, r.get(fk))) == fv]
random.Random(seed).shuffle(rows)
rows = rows[:k]
print("SRC", src, len(rows))
ims = []
for i, r in enumerate(rows):
    q = r["question"] if len(r["question"]) < 330 else r["question"][:150] + " ... " + r["question"][-150:]
    print(i, q, "|", r["options"], "| gold", r["gold"], "|", r["subtask"], {x: v for x, v in (r.get("extra") or {}).items() if x in ("count", "class", "kind", "gqa_type", "construct", "text", "group", "method", "set")})
    for p in r["images"]:
        im = Image.open(G + "img_g/" + p).convert("RGB"); im.thumbnail((440, 330)); ims.append(im)
sheet = Image.new("RGB", (4 * 440, ((len(ims) + 3) // 4) * 330), (40, 40, 40))
for j, im in enumerate(ims):
    sheet.paste(im, ((j % 4) * 440, (j // 4) * 330))
o = io.BytesIO(); sheet.save(o, "JPEG", quality=62)
print("B64START" + base64.b64encode(o.getvalue()).decode() + "B64END")
