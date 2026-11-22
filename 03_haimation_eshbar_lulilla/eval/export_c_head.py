#!/usr/bin/env python3
"""BAYES-SERVE track C: export readout C of eval/bayes_ae.py (77c662b2) as the served choice head, and prove parity.

Head = v2 head/vbll_refit.pt (sdv1_head refit 'sdh': AuxVBLL, full covariance, dim 128, seed 0). Its letter codes are extended to every
jev label exactly as bayes_ae --label-rows does: the saved 52 codes are kept bit-exact and the labels past them are appended from
make_head on label_rows E. T = bayes_ae's fit_T for C at the reference grid (64 draws, seed 0) on cal.pt: golden section over log T in
[0.1, 10], tol 1e-4, source-balanced NLL, probabilities chunked by 512 as bayes_ae. Export = fullft_head.export_head (schema
jev-residual-vbll-head-v1, samples 64, draw_seed 0, grid 1) into OUT/choice_head/.

Parity, on every --eval feature file: p_ref = bayes_ae's C (sh.draw_logits + sh.probs_T('vbll'), chunk 512) and p_srv =
vbll_core.ResidualVBLL.from_dir(OUT/choice_head).predict (chunk 2,048: a different batch shape). Reports max |p_ref - p_srv| over the
shown options and each readout's source-balanced NLL beside the A-E report's C_S64_seed0.

  python3 export_c_head.py --code trainers --engine ENGINE_DIR --heads HEADDIR --feat FEATDIR --report AE.json --out OUT \
      --eval fresh=eval_fresh.pt dev=dev.pt w2c=when2call_4opt_clean_dev.pt
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True); ap.add_argument("--engine", required=True); ap.add_argument("--heads", required=True)
    ap.add_argument("--feat", required=True); ap.add_argument("--report", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--cal-manifest", default=None, help="(v3 plan) export the T fit on the served VBLL cal rows (non-noul); report the all-rows T too")
    ap.add_argument("--export-identity", default="b69e7d2d (v2)")
    ap.add_argument("--eval", nargs="*", default=[])
    a = ap.parse_args()
    sys.path[:0] = [a.code, a.engine]
    import sdv1_head as sh
    import vbll_core as vc
    t0 = time.time()
    lt = sh.fh.pinned_trainer(a.code)
    cal = sh.load_features(f"{a.feat}/cal.pt")
    lr = torch.load(f"{a.feat}/label_rows.pt", map_location="cpu", weights_only=False)
    if lr["label_ids"] != list((cal.get("meta") or {}).get("label_ids") or []):
        raise SystemExit("label rows: label ids differ from the cal features'")
    st = torch.load(f"{a.heads}/vbll_refit.pt", map_location="cpu", weights_only=False)
    hd = sh.make_head(lt, lr["E"].float(), "vbll", st["dim"], st["seed"])
    saved, n0 = st["state"]["codes"], st["state"]["codes"].shape[1]
    dmax = float((hd.codes[:, :n0] - saved).abs().max())
    if dmax > 1e-5:
        raise SystemExit(f"rebuilt codes differ from the saved {n0} (max |diff| {dmax:.3g})")
    hd.load_state_dict({**st["state"], "codes": torch.cat([saved, hd.codes[:, n0:]], 1)})
    hd.eval()
    for p_ in hd.parameters():
        p_.requires_grad_(False)

    def p_ref(f, T, chunk=512):
        h, ell, ks = f["h"].float(), f["ell"].float(), f["k"].long()
        out = []
        with torch.no_grad():
            for i in range(0, len(ks), chunk):
                sl = slice(i, i + chunk)
                z, m = sh.draw_logits(hd, h[sl], ell[sl], ks[sl], samples=64, seed=0)
                out.append(sh.probs_T("vbll", z, m, T))
        return torch.cat(out)

    def sources(f):
        return [s if s is not None else "unknown" for s in (f.get("source") or ["all"] * len(f["k"]))]

    def fitT(c):
        t_, src_ = c["t"].float(), sources(c)
        obj = lambda lT: sh.source_balanced_nll(p_ref(c, math.exp(lT)), t_, src_)  # noqa: E731
        x_, fx_ = sh.golden(obj, math.log(sh.T_BOUNDS[0]), math.log(sh.T_BOUNDS[1]), 1e-4)
        return math.exp(x_), fx_
    T_all, nll_all = fitT(cal)
    T, fx, n_fit = T_all, nll_all, int(len(cal["k"]))
    if a.cal_manifest:                                  # the served hybrid: noul rows go to the Latent01 head, so T is fit on the rest
        man = [r for r in (json.loads(l) for l in open(a.cal_manifest) if l.strip()) if not r.get("manifest_header")]
        assert [r["decision_uid"] for r in man] == list(cal["decision_uid"]), "cal manifest order != features"
        keep = [i for i, r in enumerate(man) if (r.get("kind") or r.get("qtype")) != "noul"]
        n, ix = len(man), torch.tensor(keep)
        calc = {k: (v[ix] if torch.is_tensor(v) and v.shape[:1] == (n,) else [v[i] for i in keep] if isinstance(v, list) and len(v) == n else v)
                for k, v in cal.items()}
        T, fx = fitT(calc)
        n_fit = len(keep)
    rep = json.load(open(a.report)) if a.report else None
    T_rep = float(rep["temperatures"]["C_S64_seed0"]["T"]) if rep else None
    print("T", json.dumps({"T_exported": T, "cal_nll": fx, "cal_rows_fit": n_fit, "T_all_rows": T_all, "cal_nll_all_rows": nll_all,
                           "T_report": T_rep}), flush=True)
    W = float(sh.load_features(f"{a.feat}/headfit.pt")["w"].double().sum())
    od = os.path.join(a.out, "choice_head")
    src_info = {"readout": "C of eval/bayes_ae.py 77c662b2 at the reference grid (64 draws, seed 0)",
                "head": "esbar_final_v2 head/vbll_refit.pt (sdv1_head refit sdh)", "vbll_refit_md5": md5(f"{a.heads}/vbll_refit.pt"),
                "label_rows_md5": md5(f"{a.feat}/label_rows.pt"), "cal_md5": md5(f"{a.feat}/cal.pt"), "codes_saved": int(n0),
                "codes_total": int(hd.codes.shape[1]), "codes_rebuild_max_diff": dmax, "T_fit": {"cal_nll": fx, "tol": 1e-4, "bounds": list(sh.T_BOUNDS),
                "objective": "source-balanced NLL on cal.pt", "cal_rows": n_fit, "contract": "non-noul cal rows (served VBLL rows)" if a.cal_manifest else "all cal rows",
                "T_all_rows": T_all, "T_report_C_S64_seed0": T_rep}, "export_identity": a.export_identity}
    meta = sh.fh.export_head(od, hd, lr["label_ids"], 0, 64, W,
                             {"temperature": float(T), "temperature_semantics": "mean_s softmax(z_s / T)", "refit": sh.VERSION,
                              "bayes_serve": src_info})
    print("EXPORT", json.dumps({"dir": od, "tensor_sha256": meta["tensor_sha256"], "json_md5": md5(os.path.join(od, "vbll_head.json")),
                                "safetensors_md5": md5(os.path.join(od, "vbll_head.safetensors"))}), flush=True)
    srv, smeta = vc.ResidualVBLL.from_dir(od)
    srv.eval()
    res = {"T": T, "T_all_rows": T_all, "cal_rows_fit": n_fit, "T_report": T_rep, "cal_nll": fx, "tensor_sha256": meta["tensor_sha256"], "eval": {}}
    for spec in [f"cal={a.feat}/cal.pt"] + a.eval:
        tag, path = spec.split("=", 1)
        f = cal if tag == "cal" else sh.load_features(path if os.path.isabs(path) else f"{a.feat}/{path}")
        pr = p_ref(f, T)
        h, ell, ks = f["h"].float(), f["ell"].float(), f["k"].long()
        ps = []
        with torch.no_grad():
            for i in range(0, len(ks), 2048):
                sl = slice(i, i + 2048)
                p, _, _ = srv.predict(h[sl], ell[sl], ks[sl], int(smeta["samples"]), int(smeta["draw_seed"]), smeta.get("draw_grid_size"),
                                      int(smeta.get("draw_slot", 0)), temperature=float(smeta["temperature"]))
                ps.append(p[0])
        pv = torch.cat(ps)
        K = pr.shape[1]
        mask = torch.arange(K)[None] < ks[:, None]
        d = (pr.double() - pv[:, :K].double()).abs().masked_fill(~mask, 0)
        tt, s = f["t"].float(), sources(f)
        nll_r, nll_s = sh.source_balanced_nll(pr, tt, s), sh.source_balanced_nll(pv[:, :K], tt, s)
        rep_nll = ((rep.get("eval") or {}).get(tag) or {}).get("nll_sb", {}).get("C_S64_seed0") if rep else None
        res["eval"][tag] = {"n": int(len(ks)), "k_max": int(ks.max()), "rows_k_over_52": int((ks > 52).sum()), "max_abs_diff": float(d.max()),
                            "mean_abs_diff": float(d.sum() / mask.sum()), "argmax_disagree": int((pr.argmax(1) != pv[:, :K].argmax(1)).sum()),
                            "nll_sb_ref": nll_r, "nll_sb_served": nll_s, "nll_sb_report": rep_nll}
        print("PARITY", tag, json.dumps(res["eval"][tag]), flush=True)
    res["wall_s"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(a.out, "parity_c_head.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
