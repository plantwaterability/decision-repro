"""Check the installed environment on the GPU: PyTorch CUDA, flash-linear-attention, transformers, eshbartur module,
and the repository's files. Exits with a message if any of that fails.  uv run --frozen python selfcheck.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def fail(msg):
    print("selfcheck: FAILED: " + msg, file=sys.stderr)
    sys.exit(1)


try:
    import torch
except ImportError as e:
    fail(f"torch does not import ({e}); run `uv sync --frozen` again")
if not torch.cuda.is_available():
    fail("torch sees no GPU")
x = torch.randn(256, 256, device="cuda:0", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
if not bool(torch.isfinite(y.float()).all()):
    fail("bf16 matmul on the GPU gave non-finite values")
try:
    import fla  # noqa: F401
    import transformers
    from transformers import Qwen3_5ForCausalLM  # noqa: F401
except ImportError as e:
    fail(f"required package missing ({e})")

import eshbartur as engine_module

here = Path(engine_module.__file__).parent
cfg = json.loads((here / "eshbartur.json").read_text())
missing = [f for f in cfg["files"] if not (here / f).is_file()]
if missing:
    fail(f"{len(missing)} of the repository's files are missing here, e.g. {missing[:3]}; download the whole repository")
small = [f for f in cfg["files"] if (here / f).stat().st_size < 50_000_000]
bad = [f for f in small if engine_module.sha256(here / f) != cfg["files"][f]]
if bad:
    fail(f"sha256 of {bad[:3]} does not match eshbartur.json")

print(
    f"selfcheck: passed (torch {torch.__version__}, transformers {transformers.__version__}, "
    f"{torch.cuda.get_device_name(0)}, {len(cfg['files'])} files present, {len(small)} small files' sha256 ok)"
)
