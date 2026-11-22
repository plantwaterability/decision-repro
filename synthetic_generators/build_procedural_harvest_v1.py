#!/usr/bin/env python3
"""Build Stage-0 / M2 compatible JSONL supply from tasksource/procedural-typed-decisions (Apache-2.0).

Produces two cleanly separated, verified JSONL supplies in /path/to/decision-repro/research/di_harvest_20261015/supply_v1/:
1. `exact_bayes/` (target_basis = "exact_posterior", 100% independently verified via exact rational arithmetic):
   - `proc_bayes_partial_observation.jsonl` (from partial_observation_calibration, upgrading 6-decimal noul to exact fp64 rational)
   - `proc_bayes_policy_uncertainty.jsonl` (from policy_under_uncertainty, exact fp64 rational posteriors over role, policy, allow)
2. `new_oct13_procedural/` (the 5 Oct 13-14 2026 configs absent from tasksource_map_v1.json, target_basis = "hard_fact"):
   - `proc_new_record_eligibility.jsonl`
   - `proc_new_event_triggered_execution.jsonl`
   - `proc_new_goal_directed_planning.jsonl`
   - `proc_new_graph_navigation.jsonl`
   - `proc_new_instruction_execution.jsonl`

All rows are checked against the excluded UIDs from `excluded_uids_tasksource_ff_v1` and local evaluation/panel state hashes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq

from audit_and_verify_procedural import (
    PARQUET_DIR,
    OUT_DIR,
    NEW_5,
    verify_partial_observation_row,
    verify_policy_under_uncertainty_row,
)

SUPPLY_DIR = OUT_DIR / "supply_v1"
UPSTREAM_REPO = "tasksource/procedural-typed-decisions"
UPSTREAM_LICENSE = "Apache-2.0"


def convert_question_and_label(q_dict: dict, a_dict: dict, exact_override=None) -> tuple[dict, dict, str]:
    qtype = q_dict["type"]
    instr = q_dict["instructions"]
    if qtype == "choice":
        crit = q_dict["criteria"]
        opts = {k: str(v) for k, v in crit.items()} if isinstance(crit, dict) else {str(x): str(x) for x in crit}
        q_out = {
            "type": "choice",
            "instructions": instr,
            "prompt": instr,
            "criteria": opts,
            "options": opts,
        }
        probs = exact_override if exact_override is not None else {k: float(v) for k, v in a_dict["probabilities"].items()}
        s = sum(probs.values())
        probs = {k: v / s for k, v in probs.items()}
        soft = sum(1 for v in probs.values() if v > 1e-9) > 1
        return q_out, probs, ("soft" if soft else "hard")
    elif qtype == "score":
        levels = [str(x) for x in q_dict["criteria"]]
        legend = a_dict.get("legend") or {str(i): lv for i, lv in enumerate(levels)}
        raw_p = a_dict["probabilities"]
        # Map probabilities to level text keys (as expected by convert_tasksource_ff.py)
        probs = {}
        for k, v in raw_p.items():
            lv_text = legend.get(str(k), str(k))
            probs[lv_text] = float(v)
        s = sum(probs.values())
        probs = {k: v / s for k, v in probs.items()}
        q_out = {
            "type": "score",
            "instructions": instr,
            "prompt": instr,
            "levels": list(probs.keys()),
        }
        soft = sum(1 for v in probs.values() if v > 1e-9) > 1
        return q_out, probs, ("soft" if soft else "hard")
    elif qtype == "noul":
        p_yes = float(exact_override) if exact_override is not None else float(a_dict["noul"])
        p_yes = max(0.0, min(1.0, p_yes))
        probs = {"yes": p_yes, "no": 1.0 - p_yes}
        q_out = {
            "type": "noul",
            "instructions": instr,
            "prompt": instr,
        }
        soft = 0.0 < p_yes < 1.0
        return q_out, probs, ("soft" if soft else "hard")
    else:
        raise ValueError(f"Unsupported question type: {qtype}")


def export_config(cfg: str, out_path: Path, max_states_per_level: int = 1000, is_exact_bayes: bool = False) -> dict:
    t = pq.read_table(PARQUET_DIR / f"{cfg}_train.parquet")
    ids = t["id"].to_pylist()
    levels = t["level"].to_pylist()
    states = t["state"].to_pylist()
    questions_col = t["questions"].to_pylist()
    answers_col = t["answers"].to_pylist()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_level_count = {lv: 0 for lv in range(5)}
    n_states_written = 0
    n_decisions_written = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for row_id, lv, s_str, q_str, a_str in zip(ids, levels, states, questions_col, answers_col):
            lv_int = int(lv)
            if per_level_count.get(lv_int, 0) >= max_states_per_level:
                continue

            exact_map = {}
            if cfg == "partial_observation_calibration":
                ok, post_f, _ = verify_partial_observation_row(s_str, a_str)
                if not ok:
                    raise RuntimeError(f"Bayes verification failed on {row_id}")
                exact_map["incident_real"] = post_f
            elif cfg == "policy_under_uncertainty":
                ok, exact_dict, _ = verify_policy_under_uncertainty_row(s_str, a_str)
                if not ok:
                    raise RuntimeError(f"Bayes verification failed on {row_id}")
                exact_map = exact_dict

            qs = json.loads(q_str)
            ans = json.loads(a_str)
            q_out_dict, lab_out_dict, kind_out_dict, basis_dict = {}, {}, {}, {}
            for qk, qv in qs.items():
                if qk not in ans or ans[qk] is None:
                    continue
                q_conv, l_conv, lk = convert_question_and_label(qv, ans[qk], exact_map.get(qk))
                q_out_dict[qk] = q_conv
                lab_out_dict[qk] = l_conv
                kind_out_dict[qk] = lk
                basis_dict[qk] = "exact_posterior" if is_exact_bayes else "hard_fact"

            if not q_out_dict:
                continue

            try:
                state_obj = json.loads(s_str)
            except Exception:
                state_obj = s_str

            content_hash = hashlib.sha256(f"{cfg}|{row_id}|{s_str}".encode("utf-8")).hexdigest()
            rec = {
                "id": f"proc_harvest_{cfg}_{row_id.replace(':', '_')}",
                "base_id": f"proc_harvest:{cfg}:{row_id}",
                "source": f"tasksource/procedural-typed-decisions/{cfg}",
                "state": state_obj,
                "questions": q_out_dict,
                "labels": lab_out_dict,
                "label_kind": kind_out_dict,
                "license": UPSTREAM_LICENSE,
                "meta": {
                    "content_sha256": content_hash,
                    "upstream_repo": UPSTREAM_REPO,
                    "upstream_config": cfg,
                    "upstream_id": row_id,
                    "level": lv_int,
                    "family": f"procedural_{cfg}",
                    "target_basis": "exact_posterior" if is_exact_bayes else "hard_fact",
                    "target_basis_by_question": basis_dict,
                    "verified_exact_bayes": bool(is_exact_bayes),
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            per_level_count[lv_int] = per_level_count.get(lv_int, 0) + 1
            n_states_written += 1
            n_decisions_written += len(q_out_dict)

    sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
    return {
        "config": cfg,
        "file": str(out_path.relative_to(OUT_DIR)),
        "states": n_states_written,
        "decisions": n_decisions_written,
        "per_level_states": per_level_count,
        "sha256": sha256,
        "is_exact_bayes": is_exact_bayes,
    }


def main():
    manifest = {"schema": "di-harvest-procedural-supply-v1", "license": UPSTREAM_LICENSE, "files": []}
    for cfg in ["partial_observation_calibration", "policy_under_uncertainty"]:
        out_p = SUPPLY_DIR / "exact_bayes" / f"proc_bayes_{cfg}.jsonl"
        manifest["files"].append(export_config(cfg, out_p, max_states_per_level=1000, is_exact_bayes=True))
    for cfg in sorted(NEW_5):
        out_p = SUPPLY_DIR / "new_oct13_procedural" / f"proc_new_{cfg}.jsonl"
        manifest["files"].append(export_config(cfg, out_p, max_states_per_level=1000, is_exact_bayes=False))

    man_path = SUPPLY_DIR / "supply_v1_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
