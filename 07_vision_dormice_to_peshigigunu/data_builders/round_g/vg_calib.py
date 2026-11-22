"""PixMo flat-illustration screen calibration: stats on a seeded sample of fetched images + two contact sheets (rejected / accepted)"""
import base64, io, json, random, sys
from pathlib import Path
from PIL import Image, ImageDraw
sys.path.insert(0, "/data/shared/visdata_g/scripts")
sys.argv = [sys.argv[0]] + sys.argv[1:]
import numpy as np
P = Path("/data/shared/visdata_g/dl/pixmo")
def photo_stats(im):
    a = np.asarray(im.convert("RGB").resize((128, 128)), dtype=np.uint8) >> 3
    k = (a[..., 0].astype(np.int32) << 10) | (a[..., 1].astype(np.int32) << 5) | a[..., 2]
    u, cnt = np.unique(k, return_counts=True)
    return int(len(u)), round(float(np.sort(cnt)[::-1][:10].sum() / k.size), 3)
plan = {json.loads(l)["sha"]: json.loads(l) for l in open(P / "plan.jsonl")}
files = sorted(p.name for p in (P / "img").iterdir())
random.Random(5).shuffle(files)
stats = []
for f in files[:400]:
    try:
        im = Image.open(P / "img" / f); im.load()
    except Exception:
        continue
    u, t = photo_stats(im)
    stats.append((f, u, t, im.size))
us = sorted(s[1] for s in stats); ts = sorted(s[2] for s in stats)
q = lambda a, x: a[int(x * (len(a) - 1))]
print("n", len(stats), "colours q10/q25/q50", q(us, .1), q(us, .25), q(us, .5), "top10 q50/q75/q90", q(ts, .5), q(ts, .75), q(ts, .9))
rej = [s for s in stats if s[1] < 900 or s[2] > 0.45]; acc = [s for s in stats if not (s[1] < 900 or s[2] > 0.45)]
print("rejected", len(rej), "accepted", len(acc), "small(<224)", sum(1 for s in stats if min(s[3]) < 224))
bord = sorted(acc, key=lambda s: s[1])[:12]
for name, sel in (("rej", rej[:16]), ("acc_border", bord)):
    ims = []
    for f, u, t, sz in sel:
        im = Image.open(P / "img" / f).convert("RGB"); im.thumbnail((330, 250))
        c = Image.new("RGB", (330, 270), (30, 30, 30)); c.paste(im, (0, 0))
        ImageDraw.Draw(c).text((4, 252), f"{u} {t} {plan[f]['plural']}={plan[f]['count']}", fill=(255, 255, 0))
        ims.append(c)
    sheet = Image.new("RGB", (4 * 330, ((len(ims) + 3) // 4) * 270), (40, 40, 40))
    for j, im in enumerate(ims):
        sheet.paste(im, ((j % 4) * 330, (j // 4) * 270))
    o = io.BytesIO(); sheet.save(o, "JPEG", quality=60)
    print(f"B64{name}START" + base64.b64encode(o.getvalue()).decode() + f"B64{name}END")
