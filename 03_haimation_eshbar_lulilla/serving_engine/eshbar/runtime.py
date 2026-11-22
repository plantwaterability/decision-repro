import os
import sys
RUNTIME_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xr_runtime')
if RUNTIME_DIR in sys.path:
    sys.path.remove(RUNTIME_DIR)
sys.path.insert(0, RUNTIME_DIR)
import xr
if os.path.dirname(os.path.dirname(os.path.abspath(xr.__file__))) != RUNTIME_DIR:
    raise ImportError(f'the executor resolved to {xr.__file__}, not the bundled runtime in {RUNTIME_DIR}')
from xr import VERSION, parse_option_number, score_python_batch
WARM_PROGRAM = 'import math\ndef solution():\n    x = 3\n    return math.sqrt(16) + x ** 2 + 7 // 2\n'
STATE = {}

def warm(backend):
    import time
    t0 = time.perf_counter()
    r = score_python_batch([{'src': WARM_PROGRAM, 'inputs': {}, 'option_texts': {'a': '16', 'b': '14'}}], N=64, numbers='rational', max_steps=150000, backend=backend)[0]
    if r.get('status') != 'ok' or r.get('value') != 16 or (not (r.get('match') or {}).get('a')):
        raise RuntimeError(f"executor warm-up gave {r.get('status')} / {r.get('value')}, not a unique match on 16")
    STATE.update(version=VERSION, requested=backend, backend=r.get('backend'), fallback_reason=r.get('fallback_reason'), runtime_diff_sha256=r.get('runtime_diff_sha256'), warm_s=round(time.perf_counter() - t0, 3))
    return dict(STATE)
