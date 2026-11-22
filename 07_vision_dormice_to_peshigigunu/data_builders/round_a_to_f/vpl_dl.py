#!/workspace/anaconda3/bin/python3
"""Private-like proxy (fork A): tokenless, revision-pinned downloads of FRESH eval-only sources. python vpl_dl.py"""
import json
import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.pop("HF_TOKEN", None)
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

DL = Path("/data/shared/visdata_priv/dl")
DL.mkdir(parents=True, exist_ok=True)
SRC = {  # repo: (expected sha prefix, files or None for snapshot)
    "yifanzhang114/MME-RealWorld-Lite": ("0e378f64ae", ["data.zip", "README.md"]),
    "lytang/ChartMuseum": ("462d46deb1", None),
    "ahmed-masry/ChartQAPro": ("e27c287482", ["data/test-00000-of-00001.parquet", "README.md"]),
    "xychen-zh/multi-chart-infographic-reasoning": ("50e542ff92", None),
    "amitakamath2/whatsup_vlms": ("373b38706e", None),
    "MinhDucBui/Multi3Hate": ("e3692a908a", None),
    "konfuzio/funsd_plus": ("54c4746705", ["LICENSE", "README.md", "data/test-00000-of-00001-f819e99768aa3843.parquet"]),
    "jsdnrs/ICDAR2019-SROIE": ("bffe40c267", ["LICENSE", "NOTICE", "README.md", "data/test-00000-of-00001.parquet"]),
    "kcz358/SUNRGBD-3D": ("a5baef8e7f", None),
    "1aurent/ADE20K": ("bbf21e2662", ["README.md", "data/train-00003-of-00010.parquet", "data/train-00007-of-00010.parquet"]),
    "vikhyatk/CountBenchQA": ("76d600309e", None),
    "visualwebbench/VisualWebBench": ("9dace8aceb", ["README.md", "action_ground/test-00000-of-00001.parquet",
                                                    "element_ground/test-00000-of-00001.parquet"]),
    "Theivaprakasham/wildreceipt": ("05f68a2dbe", None),
    "FrancophonIA/XFUND": ("66a95aca09", None),
}
api = HfApi(token=False)
pins = {}
for repo, (pre, files) in SRC.items():
    t = time.time()
    try:
        info = api.dataset_info(repo)
        assert info.sha.startswith(pre), (repo, info.sha, pre)
        assert not info.gated, repo
        out = DL / repo.replace("/", "__")
        if files is None:
            snapshot_download(repo, repo_type="dataset", revision=info.sha, local_dir=out, token=False, max_workers=8)
        else:
            for f in files:
                hf_hub_download(repo, f, repo_type="dataset", revision=info.sha, local_dir=out, token=False)
        cd = info.card_data.to_dict() if info.card_data else {}
        pins[repo] = {"sha": info.sha, "licence_tag": cd.get("license"), "files": files or "snapshot"}
        print("OK", repo, info.sha[:10], round(time.time() - t), "s", flush=True)
    except Exception as e:
        print("FAIL", repo, repr(e)[:300], flush=True)
(DL / "pins.json").write_text(json.dumps(pins, indent=1))
print("DL_DONE", len(pins), "of", len(SRC), flush=True)
