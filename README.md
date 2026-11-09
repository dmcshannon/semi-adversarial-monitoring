# semi-adversarial-monitoring

Code and results for

> **What Does a Mahalanobis Monitor Detect? Controlled Experiments on Semiconductor Defect Classifiers**
> David McShannon, Abhimanyu Kanwar, Nicholas Dietrich
> Submitted to *Machine Learning with Applications*, 2026.

The paper asks what an input-space Mahalanobis monitor actually responds to when it is placed in front of a semiconductor defect classifier. On the SECOM wafer pass/fail dataset, the monitor's score turns out to be decoupled from classifier harm in both directions: random perturbations that leave the classifier unchanged are flagged on 99.7% of defective wafers, while a detector-aware PGD attack that raises the false acceptance rate to 0.765 is flagged on 10.3%, less often than clean defective wafers.

## What is here

| File | Purpose |
|---|---|
| `secom_main.py` | The full experiment in one file: preprocessing, MLP / 1D-CNN / XGBoost training, PGD (L∞ and L2), random sign control, detector-aware PGD, HopSkipJump, geometric sweeps, transfer to XGBoost, and the Mahalanobis detector. Writes one JSON per run to `results/`. |
| `fig1_sweep.py` | Detector-only replay of the two geometric controls in Fig. 1, including the two points at HopSkipJump's mean norms (L2 = 0.54 and 1.20). Runs on CPU in about a minute. |
| `detector_geometry_controls.py` | Covariance spectrum and raw-feature-scale diagnostics quoted in Sections 4.4 and 4.6. |
| `make_tables.py` | Regenerates every table and every number quoted in the paper from the files in `results/`. |
| `results/` | The result files the submitted manuscript was built from (see below). |
| `paper/` | LaTeX source, bibliography, and highlights of the submitted manuscript. |

## Data

SECOM is available from the UCI Machine Learning Repository (https://doi.org/10.24432/C54305). Download `secom.data` and `secom_labels.data` into a folder called `data/` at the root of this repository. The scripts also accept the Kaggle CSV mirror (`uci-secom.csv`) at the Kaggle input paths, or set `SECOM_DATA=/path/to/folder`.

## Reproducing the paper

```bash
pip install -r requirements.txt

# 1. Main pipeline (class weights only, HopSkipJump on all 50 folds)  ->  results/secom_results_none.json
#    About 40 minutes on a T4; HSJA runs after all other experiments and saves as it goes.
HSJA_FOLD_STEP=1 python secom_main.py

# 2. Comparison pipeline with raw-feature Gaussian oversampling        ->  results/secom_results_raw.json
OVERSAMPLE=raw python secom_main.py

# 3. PGD convergence check and CNN feature-order ablation (HSJA skipped) -> results/secom_results_none_extras.json
EXTRAS=1 python secom_main.py

# 4. Fig. 1 sweeps and covariance diagnostics (CPU, about a minute each)
python fig1_sweep.py
python detector_geometry_controls.py

# 5. Tables and all quoted values
python make_tables.py --check paper/mlwa_elsarticle.tex
```

`HSJA_FOLD_STEP` is read from inside the script (default 10, which runs HopSkipJump on every tenth fold); edit the constant at the top of `secom_main.py` or set it as shown. `QUICK=1` runs a reduced smoke test. `MAX_FOLDS=n` stops after n folds.

Neural network training on a GPU is not bit-reproducible. Repeated runs differ by about 0.01 in clean FAR, which is why the paper reports the PGD convergence check against the 20-step attack from the same run rather than against the main run.

## Result files

| File | Produced by | Used for |
|---|---|---|
| `results/secom_results_none.json` | `secom_main.py` (defaults, `HSJA_FOLD_STEP = 1`) | Tables 1 to 3, Fig. 1 attack markers, all HopSkipJump numbers |
| `results/secom_results_raw.json` | `secom_main.py` with `OVERSAMPLE=raw` | Table 3 right-hand columns, Section 4.6 |
| `results/secom_results_none_extras.json` | `secom_main.py` with `EXTRAS=1` | PGD convergence check, CNN feature-order ablation |
| `results/fig1_sweep_results.json` | `fig1_sweep.py` | Fig. 1 curves |
| `results/detector_geometry_results.json` | `detector_geometry_controls.py` | Eigenvalue counts, raw feature scales, synthetic-row fraction |

Each JSON maps a metric name to `{"mean", "std", "n", "values"}` with one value per fold.

## Pipeline summary

10×5 repeated stratified cross-validation. Inside each fold: median imputation, ANOVA F-test selection of 80 features, standardization, all fitted on the training portion only, with a stratified 90/10 split of the training portion for early stopping and threshold tuning. Perturbations are applied in standardized space and clamped to the training range; the clean test set is clamped the same way, so reported L∞ norms never exceed the budget. The detector is a single Gaussian fitted to the original training rows in input space, flagging inputs above the 95th percentile of validation scores. Full details are in Section 3 of the paper.

## Citation

```bibtex
@article{mcshannon2026mahalanobis,
  title   = {What Does a {Mahalanobis} Monitor Detect? Controlled Experiments on Semiconductor Defect Classifiers},
  author  = {McShannon, David and Kanwar, Abhimanyu and Dietrich, Nicholas},
  journal = {Machine Learning with Applications},
  year    = {2026},
  note    = {Submitted}
}
```

## License

MIT. See `LICENSE`.
