# decision-repro

End-to-end training, data-building, head-fitting, distillation, weight-averaging, merging, and export scripts to reproduce all 17 `plantwaterer` Decision Index models (11 text models and 6 vision models).

## Repository Layout

| Directory | Models covered | Key contents |
|---|---|---|
| `01_garas_garum/` | `garas`, `garum` | `trainers/claude_lmtune.py`, `vbll.py`, `task_ddp.sh`, `task_screen4.sh`, `task_screen4f.sh`, `build_screen4*.py`, 25+ real-data and benchmark builders (`build_jev_*.py`, `build_wikidata_jev.py`, `build_gdelt_jev.py`) |
| `02_allec_muria_oenogarum_liquamen/` | `allec`, `muria`, `oenogarum`, `liquamen` | `trainers/fullft_train.py`, `fullft_mem.py` (Policy B attention), `fullft_head.py`, `sdv1_head.py`, `stock_decision_v1/` configs, Stage-0 builders, clean head-fit/calibration manifests, `serving_engines/` (`LenAvgBayesBalancedK5Engine`, `DikitEngine`) |
| `03_haimation_eshbar_lulilla/` | `haimation`, `eshbar-lulilla` | `stages/{final_v1,v2}/`, `training_ov/`, `data/` (chess Stockfish labelling, manifests, 18 synthetic pool generators), `serving_engine/` (`eshbar_lulilla`, LOOM executor, store reader) |
| `04_munku/` | `munku` | `breadth_v1/`, `v8build/`, `ref/` (`stage1_fs5.sh`, `soup75_build.sh`, `soup_pair.py`), `licence/` (`fs1_source_counts.json`, `fs1_upstream_map.tsv`, `fs1_vs_e1_lineage.json`) |
| `05_kudhab/` | `kudhab` | `trainer/` (`kl_train.py`, `kl_data.py`, `kl_run.sh`, `kl_hbuild24.sh`, `kl_pick24.py`), `data_round_h/` (`fs5h_build.py`, `dh_h0.py`), `data_fs5/` (`fs5_build.py`, `fs5_validate.py`) |
| `06_eshbartur/` | `eshbartur` | `trainer/` (`t4b_train.py`, `t4b_prep_data.py`, `t4b_layout.py`, `t4b_chain.sh`), `export/` (`eshbartur_export.py`, `eshbartur_check_merge.py`), `serving/` (`eshbartur.py`, `install.sh`, `pyproject.toml`) |
| `07_vision_dormice_to_peshigigunu/` | `dormice`, `glires`, `peshgi`, `peshashaga`, `peshigigunu` | `trainers/` (`qv_train.py`, `qv_train_vl.py`, `qv_ocr.py`, `qv_layout.py`, `vl_gen_code.py`), `data_builders/` (Rounds A-G builders `vb_*.py`, `vg_*.py`, `vg_gqa.py`), `ocr_prep/`, `soups_and_merges/` (`build_f5.py`, `peshashaga_blend.env`, `m2_build.py`, `wx_engine.py`, `peshigigunu_merge.env`), `manifests/manifest_sources.json` |
| `08_peshtur/` | `peshtur` | `trainer/` (`vx_train_vl.py`, `v4b_chain.sh`, `wx_engine.py`), `export/` (`peshtur_export.py`, `peshtur_check_merge.py`), `serving/` (`peshtur.py`, `install.sh`, `pyproject.toml`) |
| `synthetic_generators/` | All text models | All 80 `synth_workflows/` schemas, all 18 `pools_generators/` modules, Bayesian/numeric/procedural generators (`build_bayes_*.py`, `build_numeric_*.py`, `build_pcd*.py`, `build_prior_cond.py`, `build_procedural_harvest_v1.py`) |

## Pinned External Dependencies and Base Models

- **Decision Index kit:** `apolinario/decision-index`
  - Text runs: `9eb2dbe2a358004c8782c66e40a83ac07b953fec`
  - Vision runs: `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d`
- **Base models (Apache-2.0):**
  - `google/gemma-4-31B-it` @ `842da3794eaa0b77d5f08bae87a17459d91ff475` (`garas` through `kudhab`)
  - `google/gemma-4-12B-it` @ `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` (`muria`)
  - `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (`dormice` through `peshigigunu`)
  - `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (`eshbartur`, `peshtur`)
- **External parents, teachers, and engines (Apache-2.0 unless noted):**
  - `tattrongvu/Jeffan-31b-1710` @ `cebfa86c2bed2b5c83b12e44de4374b20c87fd79` (`kudhab` starting merge)
  - `MachinePulseAI/karpo-decision-31B` @ `d69cdd57` / `06b966b9e154` (bit-identical shards; Apache-2.0 since 2026-11-16) and `machinepulse-ai/karpo-decision` @ `a6fa38bf4e8de9dbdb9491fb461b9b320a2e2118` (`kudhab`, `eshbartur`)
  - `Rover00/WeMM-Decision` @ `1543fbfcb171c7990607369e4c17329d3a7a6385` (`peshigigunu` merge parent, `peshtur` teacher 1 of 3)
  - `XiaomiMiMo/MiMo-V2.6-Pro-RL` @ `a066ba0057192126bd5d255f37138265fc1a435e` (MIT; 5,354 soft targets in `munku`)
  - `Qwen/Qwen3-32B` @ `9216db57` (SuperGPQA key checks in `munku`)
  - `Qwen/Qwen3.5-27B` (cross-architecture co-teaching in `liquamen`'s intermediate checkpoint)
  - `PaddlePaddle/PP-OCRv5_mobile_det` + `PaddlePaddle/en_PP-OCRv5_mobile_rec` (live OCR in `glires` through `peshtur`)

## Companion Datasets (CC BY-SA 4.0)

- Synthetic text corpora + generators: `plantwaterer/decision-synthetic-text` (and `decision-synthetic-extra`, `decision-bayes-worlds`, `decision-numeric-full`)
- Open GDELT event-forecasting corpus: `plantwaterer/decision-gdelt-v1`
- Open/public-record real-data corpus (`rd3`, 828,967 units): `plantwaterer/decision-realdata-rd3`
- Multi-source text manifests, licence tables, and non-redistributed source builders: `plantwaterer/decision-text-recipes`
- Vision manifests, deduplication/OCR specs, and non-redistributed source builders: `plantwaterer/decision-vision-recipes`

## Model-by-Model Reproduction Summary

### Text Line (`google/gemma-4-31B-it` / `12B-it` / `Qwen/Qwen3.5-4B`)

1. **`garas`**: Build `rd3` (or load `decision-realdata-rd3`), `gdelt_v1` (`decision-gdelt-v1`), `flights` (`decision-synthetic-text`), and `mix2` (`build_jev_data.py` + `build_jev_ext*.py` excluding `ms_marco`). Run `01_garas_garum/fleet/task_ddp.sh` (`trainers/claude_lmtune.py`, LoRA `r=16`, `alpha=32`, 410 LM linears, 18,000 updates, batch 64, peak LR `1e-4`, warmup 40, cosine, `max_len=2048`, seed 0).
2. **`garum`**: Build `s4_mixed_v1` (`build_screen4.py` / `build_screen4f.py`: 8,000 replay + 4,000 evidence-grounded units). Run `task_screen4f.sh` for 2 seeds (`s0`, `s1`) initialized from `garas`'s LoRA (`1,500` updates, batch 8, LR `1.5e-5`, `rps_weight=1.0`, parent soft-label weight 0.3). Stack `0.50 * garas + 0.25 * s0 + 0.25 * s1` into a rank-48 adapter and fit `vbll_head.safetensors`.
3. **`allec`**: Assemble `M2.1sP` (`38,400` rows, `<= 8,192` tokens) and clean head pools (`16,384` fit / `4,086` cal) via `02_allec_muria_oenogarum_liquamen/stock_decision_v1/stage0/code/`. Train 2 full fine-tune seeds (seeds 0, 1) from stock Gemma 4 31B with `fullft_train.py` (Policy B via `fullft_mem.py`, fused fp32 AdamW `(0.9, 0.95)`, `wd=0`, clip `1.0`, peak LR `2.5e-6`, warmup 50, cosine floor `0.1`, `600` updates x `64`, `alpha=0.15`), average 0.5/0.5 in fp32, and refit the head (`T = 1.031`) via `fullft_head.py`.
4. **`muria`**: Run the exact `allec` recipe on `google/gemma-4-12B-it @ 707f0a3b` (2 seeds averaged 0.5/0.5, head `T = 0.970`).
5. **`oenogarum`**: Sample `U277pA` (`38,400` rows from the 276k pool, parent soft labels at 0.15 from stock Gemma on 17,334 rows and `garas` on 7,384 rows). Train 3 seeds (0, 1, 2) with `allec`'s recipe, average 1/3 each in fp32, and refit the head (`T = 1.080`).
6. **`liquamen`**: Average 5-seed `allec`/`oenogarum` pools with cross-architecture `Qwen/Qwen3.5-27B` co-teaching waves into the shared intermediate checkpoint `C0`, branch into `C5` (`400` updates x `64` over the 164-family `<= 4,096`-token mixture) and `C10` (same plus Prentice-Pyke case-control offsets on ACOS/MultiRC and balanced ESCI), average `0.50 * C5 + 0.50 * C10` in fp32, and refit the head (`T = 1.085`). Serve with `LenAvgBayesBalancedK5Engine`.
7. **`haimation`**: From `liquamen`, run `03_haimation_eshbar_lulilla/stages/final_v1/stage1_final.sh` (`1,369` steps, `87,601` rows) followed by `stages/v2/stage1_v2.sh` (`1,320` steps, `84,455` rows, LR `7.5e-7`, warmup 30, `alpha=0.35`, `pack_tokens=8192`, seed 0), then refit head (`T = 1.2984`).
8. **`eshbar-lulilla`**: Build the `199,803`-row `v5_main_a4` mixture using `03_haimation_eshbar_lulilla/data/` (`build_chess_pool.py`, `sf_label_chess.py`, `pools_generators/*`, DI kit builders). Continue from `haimation` for `3,122` updates x `64`, then refit head (`T = 1.4004`).
9. **`munku`**: Build `v8` (`04_munku/v8build/`), `breadth_v1` (`04_munku/breadth_v1/`), and `pools_v4` (`384,663` rows total; 5,354 `MiMo-V2.6-Pro-RL` targets, `Qwen3-32B` SuperGPQA checks, `eshbar-lulilla` soft targets at 0.15 on 281,765 rows). Train `FS1F` from stock Gemma 4 31B (`6,011` steps x `64`, peak LR `2.5e-6`, warmup 100, cosine floor `0.1`, seed 0), average `0.50 * eshbar-lulilla + 0.50 * FS1F` via `soup75_build.sh` / `soup_pair.py`, and refit head (`T = 1.2786`).
10. **`kudhab`**: Average `0.50 * munku + 0.50 * Jeffan-31b-1710` in fp32. Label `FS5` (`420,731` rows via `05_kudhab/data_fs5/fs5_build.py`) with `karpo-decision-31B` (`view=joint`, `T=1`). Train 4 Stage-1 students with `05_kudhab/trainer/kl_train.py` (linear layers of 60 blocks, LR `1e-6`, warmup 100, batch 64: `K7`/`K8` pure KL stopped at step 2,811; `K12` pure KL at step 5,622; `K3` `0.8 KL + 0.2 gold` at step 6,574), average `0.25` each, then run Stage 2 (`kl_hbuild24.sh` + `fs5h_build.py`, `2,152` steps over `137,708` rows excluding `droplist_roundH`, `cfcolor`, `HellaSwag`, pure KL, LR `5e-7`, seed 1).
11. **`eshbartur`**: Assemble the `445,594`-question pool via `06_eshbartur/trainer/t4b_prep_data.py` with `karpo-decision-31B` targets. Train `Qwen/Qwen3.5-4B` pure-text backbone (`t4b_train.py` / `t4b_chain.sh`: LoRA `r=32`, `alpha=64`, 248 LM linears, bidirectional full attention + RE2 read-twice for `<= 2,048` tokens, `3,516` steps x batch 128, LR `4e-5` LoRA / `2e-4` readout, `wd=0.02`, pure KL), fit temperature `T = 1.40`, and merge/export via `06_eshbartur/export/eshbartur_export.py`.

### Vision Line (`Qwen/Qwen3.8-27B` / `Qwen/Qwen3.5-4B`)

12. **`dormice`**: Build `arm_a_nc` (`60,632`), `a2_mix` (`64,000`), `a3_mix` (`64,000`), and `u_union` (`192,000`) with `07_vision_dormice_to_peshigigunu/data_builders/round_a_to_f/` (exact sha256 + pHash `<= 6` deduplication against Decision Index Vision 0.3 non-Winoground images). Train `arm_a_nc` (1,000 steps), `a2_mix`/`a3_mix` from `arm_a_nc` (1,000 steps each), and `u_union` from stock Qwen3.8-27B (3,000 steps) via `qv_train.py` (`r=16`, `alpha=32`, 496 LM linears, vision tower frozen, LoRA LR `1e-4`, head LR `1e-3`, warmup 50, cosine floor `0.1`, `label_smoothing=0.05`, batch 64). Package `0.50 * u_union + 0.25 * a2_mix + 0.25 * a3_mix` via `dormice_build_bundle.py`.
13. **`glires`**: Run PP-OCRv5 over `u_union` (`ocr_prep/`), train `uo_union` (`3,000` steps, OCR) and `ue2_noade` (`3,700` steps on `236,800` rows, no OCR) with `qv_train.py`, and blend `[0.375, 0.125, 0.0625, 0.0625, 0.375]` (`uo_union, u_union, a2_mix, a3_mix, ue2_noade`).
14. **`peshgi`**: Build `uf2_nc_ade` (`289,200` rows), train `uf2` (no OCR) and `uof2` (with OCR) for `4,519` steps each, and blend all 7 members `[0.1, 0.1, 0.05, 0.05, 0.1, 0.3, 0.3]` via `soups_and_merges/build_f5.py`.
15. **`peshashaga`**: Build `ug1_nc` (`323,264` rows via `data_builders/round_g/vg_build.py` + `vg_gqa.py`), train `uoq1` (`5,051` steps, OCR + `sandwich` layout via `trainers/sandwich/qv_train_vl.py`), and blend all 8 members per `soups_and_merges/peshashaga_blend.env` (`[0.1, 0.1, 0.05, 0.05, 0.1, 0.3, 0.3, 0.3] / 1.3`).
16. **`peshigigunu`**: Zero-training merge `1.0 * WeMM-Decision + 0.5 * delta_peshashaga` across the 496 LM linear matrices in fp32 (single bf16 rounding) using `soups_and_merges/m2_build.py` and `peshigigunu_merge.env`; verify bit-identity with `m2_check_merge.py`.
17. **`peshtur`**: Generate 3-teacher probabilities over all `323,264` `ug1_nc` rows (`WeMM-Decision`, `peshashaga`, and `vxg2` vision LoRA of `eshbar-lulilla`), mix `0.5 * gold + 0.5 * mean(teachers)` with `0.05` label smoothing, train `Qwen/Qwen3.5-4B` via `08_peshtur/trainer/v4b_chain.sh` (`vx_train_vl.py`, `r=16`, `alpha=32`, 248 LM linears, vision encoder and merger frozen, `5,051` steps x `64`, LR `5e-5` LoRA / `3e-4` head, `wd=0.02`, `d3` + PP-OCRv5 prompt), and merge/export via `08_peshtur/export/peshtur_export.py`.

## Licence

Author-written scripts in this repository are released under Apache-2.0. Companion datasets and data recipes are released under CC BY-SA 4.0. Upstream models, engines, and third-party datasets remain under their respective original licences.
