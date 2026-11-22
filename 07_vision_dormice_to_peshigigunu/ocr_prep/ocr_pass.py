"""OCR pass for the OCR-trained arm: PP-OCRv5 (mobile det + en mobile rec; Arm 0's exact settings) over one shard of an image
list -> jsonl (per image: lines with box/score, wall ms). Resumable: images already in the output are skipped.
usage: python ocr_pass.py <image_list.txt> <image_root> <out_shard.jsonl> <shard k> <n shards> <device>"""
import json
import os
import sys
import time

list_f, root, out_f, k, n, device = sys.argv[1:7]
k, n = int(k), int(n)
from paddleocr import PaddleOCR  # noqa: E402

imgs = [l.rstrip("\n") for l in open(list_f) if l.strip()][k::n]
done = set()
if os.path.exists(out_f):
    good = []
    for l in open(out_f):
        try:
            done.add(json.loads(l)["image"])
            good.append(l)
        except Exception:  # a kill mid-write leaves a partial last line
            break
    open(out_f, "w").writelines(good)
todo = [p for p in imgs if p not in done]
if not todo:
    print(json.dumps({"shard": k, "images": len(imgs), "todo": 0}))
    sys.exit(0)
ocr = PaddleOCR(text_detection_model_name="PP-OCRv5_mobile_det", text_recognition_model_name="en_PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
                text_det_limit_side_len=960, text_det_limit_type="max", text_recognition_batch_size=8, device=device)
ocr.predict(f"{root}/{todo[0]}")  # first call builds/caches kernels; excluded from timing like model load
errs = 0
with open(out_f, "a") as f:
    for i, p in enumerate(todo):
        t0 = time.perf_counter()
        rec = {"image": p}
        try:
            res = ocr.predict(f"{root}/{p}")
            lines = []
            for r in res:
                j = r.json["res"] if hasattr(r, "json") else r
                for t, s, b in zip(j.get("rec_texts", []), j.get("rec_scores", []), j.get("rec_boxes", [])):
                    b = [float(x) for x in (b.tolist() if hasattr(b, "tolist") else b)]
                    lines.append({"text": t, "score": float(s), "box": b})
            rec["lines"] = lines
        except Exception as e:  # recorded, counted, never silently dropped
            rec["lines"], rec["err"] = [], repr(e)[:300]
            errs += 1
        rec["ms"] = (time.perf_counter() - t0) * 1000
        f.write(json.dumps(rec) + "\n")
        if i % 500 == 0:
            f.flush()
print(json.dumps({"shard": k, "images": len(imgs), "todo": len(todo), "errors": errs}))
