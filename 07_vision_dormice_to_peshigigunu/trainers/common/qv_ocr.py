"""OCR text in the request state (Arm 0's fixed format), shared by the trainer and the eval wrapper.

Before the question: "Text found in the image by OCR (may contain errors):", one line per image "[image k] t1 | t2 | ..."
(rec score >= 0.5; top-to-bottom in 12 px bands, then left-to-right; <= 60 lines and <= 1,200 characters per image;
"[image k] (no text)" for a textless image), a blank line, then the unchanged question. The block is placed only on rows
where OCR found text in at least one image; a row whose images are all textless is left exactly as it was."""
import json

HEADER = "Text found in the image by OCR (may contain errors):"


def load_ocr(path):
    """OCR cache jsonl -> ({image: "t1 | t2 | ..." or ""}, {image: OCR wall ms})."""
    text, ms = {}, {}
    for l in open(path):
        r = json.loads(l)
        lines = [x for x in r["lines"] if x["score"] >= 0.5 and x["text"].strip()]
        lines.sort(key=lambda x: (round(x["box"][1] / 12), x["box"][0]))
        txt, n = [], 0
        for x in lines[:60]:
            if n + len(x["text"]) > 1200:
                break
            txt.append(x["text"])
            n += len(x["text"]) + 3
        text[r["image"]] = " | ".join(txt)
        ms[r["image"]] = float(r.get("ms", 0.0))
    return text, ms


def ocr_prefix(text, images):
    """The block to put before the question, or "" (no cache given, or no text in any image). A missing image fails closed."""
    if text is None or not images:
        return ""
    t = [text[p] for p in images]
    if not any(t):
        return ""
    return "\n".join([HEADER] + [f"[image {k}] {x or '(no text)'}" for k, x in enumerate(t, 1)]) + "\n\n"
