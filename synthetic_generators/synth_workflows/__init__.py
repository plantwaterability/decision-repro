"""synth_workflows: latent-factor case generators for NEW workflow families in the typed-decisions style.

Registry: every module in this package that is not infrastructure defines a module-level `WORKFLOW`
(a base.Workflow). `load_all()` imports them all, runs the contract and exclusion checks, and returns
{name: Workflow}. Adding a workflow is adding one file; nothing here needs editing (see README.md).
"""
from __future__ import annotations

import importlib
import pkgutil

from .base import GENERATOR_VERSION, Schema, Workflow, case_rng, make_case  # noqa: F401

INFRA = {"base", "exclusion", "contract"}


def workflow_modules() -> list:
    return sorted(m.name for m in pkgutil.iter_modules(__path__)
                  if m.name not in INFRA and not m.name.startswith("_"))


def load_all(names=None, strict: bool = True) -> dict:
    """{workflow name: Workflow} for every workflow module (or just `names`), checked.

    strict: raise on any contract or exclusion error (the build always runs strict). With strict=False the
    errors are returned in `load_all.errors` instead, so a self-test can report every problem at once.
    """
    from .exclusion import exclusion_errors

    reg, per_module = {}, {}
    wanted = set(names) if names else None
    for mod_name in workflow_modules():
        if wanted is not None and mod_name not in wanted:
            continue                      # a broken module elsewhere must not block building a subset
        errs = per_module.setdefault(mod_name, [])
        try:
            mod = importlib.import_module(f"{__name__}.{mod_name}")
        except Exception as ex:  # noqa: BLE001
            errs.append(f"{mod_name}: import failed: {type(ex).__name__}: {ex}")
            continue
        wf = getattr(mod, "WORKFLOW", None)
        if not isinstance(wf, Workflow):
            errs.append(f"{mod_name}: no module-level WORKFLOW = base.Workflow(...)")
            continue
        if wf.name != mod_name:
            errs.append(f"{mod_name}: WORKFLOW.name is {wf.name!r}; it must equal the module name")
        errs.extend(wf.workflow_errors())
        errs.extend(exclusion_errors(wf))
        reg[wf.name] = wf
    errors = [e for errs in per_module.values() for e in errs]
    if names:
        missing = [n for n in names if n not in per_module]
        if missing:
            errors.append(f"unknown workflows {missing}; known: {workflow_modules()}")
        reg = {n: reg[n] for n in names if n in reg}
    load_all.errors = errors
    if strict and errors:
        raise SystemExit("synth_workflows contract/exclusion errors:\n  " + "\n  ".join(errors))
    return reg


load_all.errors = []
