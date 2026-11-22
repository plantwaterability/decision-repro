"""Check the installed environment on the GPU: OpenCV, PyTorch and PaddlePaddle on one GPU in one process, the engine module,
the repository's files, and one OCR call. Exits with a message if any of that fails.  uv run --frozen python selfcheck.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def fail(msg):
    print("selfcheck: FAILED: " + msg, file=sys.stderr)
    sys.exit(1)


try:
    import cv2  # noqa: F401
except ImportError as e:
    fail(f"OpenCV does not import ({e}); install the system packages libgl1 and libglib2.0-0")
try:
    import torch
except ImportError as e:
    fail(f"torch does not import ({e}); PyTorch's CUDA libraries are not the ones installed; run `uv sync --frozen` again")
if not torch.cuda.is_available():
    fail("torch sees no GPU")
x = torch.randn(4, 3, 64, 64, device="cuda:0", dtype=torch.bfloat16)
y = torch.nn.functional.conv2d(x, torch.randn(8, 3, 3, 3, device="cuda:0", dtype=torch.bfloat16))
torch.cuda.synchronize()
if not bool(torch.isfinite(y.float()).all()):
    fail("a convolution on the GPU gave non-finite values")
import paddle

if not paddle.device.is_compiled_with_cuda():
    fail("this PaddlePaddle build has no GPU support")
import numpy as np
import peshtur as engine_module  # the engine, without loading the model
from huggingface_hub import snapshot_download
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont

here = Path(engine_module.__file__).parent
cfg = json.loads((here / "peshtur.json").read_text())
missing = [f for f in cfg["files"] if not (here / f).is_file()]
if missing:
    fail(f"{len(missing)} of the repository's files are missing here, e.g. {missing[:3]}; download the whole repository")
small = [f for f in cfg["files"] if (here / f).stat().st_size < 50_000_000]
bad = [f for f in small if engine_module.sha256(here / f) != cfg["files"][f]]
if bad:
    fail(f"sha256 of {bad[:3]} does not match peshtur.json")
oc = cfg["ocr"]
det = snapshot_download(oc["detection"]["repo"], revision=oc["detection"]["revision"])
rec = snapshot_download(oc["recognition"]["repo"], revision=oc["recognition"]["revision"])
try:
    ocr = PaddleOCR(
        text_detection_model_name=oc["detection"]["name"],
        text_detection_model_dir=str(det),
        text_recognition_model_name=oc["recognition"]["name"],
        text_recognition_model_dir=str(rec),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_side_len=oc["det_limit_side_len"],
        text_det_limit_type="max",
        text_recognition_batch_size=oc["rec_batch_size"],
        device="gpu:0",
    )
    im = Image.new("RGB", (640, 160), (255, 255, 255))
    try:
        font = ImageFont.load_default(size=48)
    except TypeError:
        font = ImageFont.load_default()
    ImageDraw.Draw(im).text((30, 50), "INSTALL 2468", fill=(0, 0, 0), font=font)
    texts = []
    for r in ocr.predict(np.ascontiguousarray(np.asarray(im)[:, :, ::-1])):
        j = r.json["res"] if hasattr(r, "json") else r
        texts += list(j.get("rec_texts", []))
except Exception as e:
    fail(f"the OCR model did not run on the GPU ({type(e).__name__}: {e})")
if "2468" not in " ".join(texts):
    fail(f"the OCR model read {texts!r} instead of 'INSTALL 2468'")
y = torch.nn.functional.conv2d(x, torch.randn(8, 3, 3, 3, device="cuda:0", dtype=torch.bfloat16))
torch.cuda.synchronize()
print(
    f"selfcheck: passed (torch {torch.__version__}, paddle {paddle.__version__}, {torch.cuda.get_device_name(0)}, "
    f"{len(cfg['files'])} files present, {len(small)} small files' sha256 ok, OCR read {texts})"
)
