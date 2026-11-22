import hashlib
import json
import os

from .route import RouteEngine, MEM_TAU
from . import runtime as _runtime

PINS_FILE = "PINS.json"
FROZEN = ("route_tau", "route_tau_truth", "mem_block", "mem_graph", "exec_backend")


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_pins(model_dir):
    path = os.path.join(model_dir, PINS_FILE)
    if not os.path.exists(path):
        raise SystemExit(f"{model_dir}: no {PINS_FILE}")
    with open(path) as f:
        pins = json.load(f)
    for k in ("arm", "engine_class", "route_thresholds", "package_files", "noul_head", "checkpoint"):
        if k not in pins:
            raise SystemExit(f"{PINS_FILE}: missing {k!r}")
    if pins.get("choice_head"):
        raise SystemExit(f"{PINS_FILE}: this engine serves the model's own head; a choice head is not part of it")
    return path, pins


def verify_files(model_dir, pins):
    want = dict(pins["package_files"])
    want.update(pins.get("memory_files") or {})
    want[pins["noul_head"]["file"]] = pins["noul_head"]["json_md5"]
    bad = []
    for rel, m in sorted(want.items()):
        p = os.path.join(model_dir, rel)
        if not os.path.exists(p):
            bad.append(f"missing {rel}")
        elif _md5(p) != m:
            bad.append(f"md5 {rel}")
    if bad:
        raise SystemExit(f"{model_dir}: files differ from {PINS_FILE}: " + "; ".join(bad[:8]))
    return len(want)


class Engine(RouteEngine):
    name = "eshbar-lulilla"
    latency = ("CUDA-synchronized in-process request wall time: conversion, one forward per prompt (a request's prompts packed), the "
               "head's readout, and on executed rows the program generation and the executor run; excludes model loading.")

    def __init__(self, model="", **options):
        for k in FROZEN:
            if k in options:
                raise SystemExit(f"{k} is fixed by {PINS_FILE}")
        mdir = model
        if not os.path.isdir(mdir):
            from huggingface_hub import snapshot_download
            mdir = snapshot_download(model)
        pins_path, pins = load_pins(mdir)
        n_files = verify_files(mdir, pins)
        rt = pins["route_thresholds"]
        tau_num, tau_truth = float(rt["tau_num"]), float(rt["tau_truth"])
        stated = getattr(type(self), "route_tau_pinned", None)
        if stated is not None and (float(stated[0]), float(stated[1])) != (tau_num, tau_truth):
            raise SystemExit(f"thresholds {stated} of {type(self).__module__} differ from {PINS_FILE} {(tau_num, tau_truth)}")
        backend = pins.get("executor_backend", "auto")
        self._exec_warm = _runtime.warm(backend)
        super().__init__(model=mdir, route_tau=tau_num, route_tau_truth=tau_truth, exec_backend=backend, **options)
        prov = {"engine": self.name, "arm": pins["arm"], "pins_md5": _md5(pins_path), "files_verified": n_files,
                "export_identity_sha256": pins["checkpoint"].get("export_identity_sha256"),
                "files_md5": dict(sorted(pins["package_files"].items())),
                "head": {"file": pins["noul_head"]["file"], "json_md5": pins["noul_head"]["json_md5"],
                         "tensor_sha256": pins["noul_head"].get("tensor_sha256"), "device": getattr(self, "head_device", None)},
                "route": {"on": self.route_on, "tau_num": tau_num, "tau_truth": tau_truth, "max_new_tokens": self.route_max_new,
                          "executor": {"requested": self.exec_backend, **self._exec_warm}, "warm_s": self._warm_s},
                "store": {"on": self.mem_on, "tau": MEM_TAU, "files": dict(sorted((pins.get("memory_files") or {}).items())),
                          "warm": getattr(self, "_mem_warm", None)}}
        try:
            import torch
            import transformers
            prov.update(torch=torch.__version__, transformers=transformers.__version__)
        except ImportError:
            pass
        self.provenance = prov
