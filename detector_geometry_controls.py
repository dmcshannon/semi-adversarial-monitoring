# Detector geometry controls (standalone, no model training, runs in about a minute)
# 1. Clamp fix: clean and perturbed sets are clamped identically
# 2. AUROC vs L2 sweep for isotropic random noise
# 3. On-manifold control: same L2, but pointed at a real passing wafer (HSJA-like geometry)
# 4. Mahalanobis norm of delta, covariance spectrum, raw feature scales
# 5. Everything twice: detector fit on oversampled training data (as in the paper) vs original data
import numpy as np, pandas as pd, json, os
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings("ignore")

SEED = 42
FOLD_STEP = 1                      # 1 = all 50 folds, 10 = the 5 folds used for HSJA
L2_GRID = [0.1, 0.2, 0.31, 0.5, 0.76, 1.0, 1.35, 1.8]

DATA_DIR = os.environ.get("SECOM_DATA", "data")
if os.path.exists(os.path.join(DATA_DIR, "secom.data")):
    X_raw = np.genfromtxt(os.path.join(DATA_DIR, "secom.data")).astype(np.float32)
    y_raw = ((np.genfromtxt(os.path.join(DATA_DIR, "secom_labels.data"), usecols=0) + 1) / 2).astype(int)
else:
    path = "/kaggle/input/uci-semcom/uci-secom.csv"
    for p in ["/kaggle/input/datasets/paresh2047/uci-semcom/uci-secom.csv", "/kaggle/input/uci-secom/secom.csv"]:
        if not os.path.exists(path) and os.path.exists(p): path = p
    df = pd.read_csv(path)
    y_raw = df["Pass/Fail"].values
    X_raw = df.drop("Pass/Fail", axis=1).select_dtypes(include=[np.number]).values.astype(np.float32)
    if set(np.unique(y_raw)) == {-1, 1}: y_raw = ((y_raw + 1) / 2).astype(int)
X_full = X_raw[:, np.isnan(X_raw).mean(axis=0) < 0.5]
X_full = X_full[:, np.nanstd(X_full, axis=0) > 1e-8]
print(f"SECOM: {X_full.shape}, defect rate: {y_raw.mean():.4f}")

def mdist(X, mu, Sinv):            # Mahalanobis distance of each row to mu
    D = X - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", D, Sinv, D))

def mnorm(delta, Sinv):            # Mahalanobis norm of the perturbation itself
    return np.sqrt(np.einsum("ij,jk,ik->i", delta, Sinv, delta))

def auroc(clean, pert):
    return float(roc_auc_score(np.r_[np.zeros(len(clean)), np.ones(len(pert))], np.r_[clean, pert]))

res = {}
def log(key, val): res.setdefault(key, []).append(float(val))

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
for fold_i, (tr_idx, te_idx) in enumerate(rskf.split(X_full, y_raw)):
    if fold_i % FOLD_STEP != 0: continue
    y_tr, y_te = y_raw[tr_idx], y_raw[te_idx]
    imp = SimpleImputer(strategy="median")
    X_tr_imp = imp.fit_transform(X_full[tr_idx]).astype(np.float32)
    X_te_imp = imp.transform(X_full[te_idx]).astype(np.float32)
    X_tr_imp, _, y_tr, _ = train_test_split(X_tr_imp, y_tr, test_size=0.1,
                                            random_state=SEED + fold_i, stratify=y_tr)
    sel = SelectKBest(f_classif, k=80)
    X_tr = sel.fit_transform(X_tr_imp, y_tr).astype(np.float32)
    X_te = sel.transform(X_te_imp).astype(np.float32)

    # Raw scale check: sigma=0.05 noise is large for any feature with raw std below 0.05
    raw_std = X_tr.std(axis=0)
    log("n_features_raw_std_below_0.05", (raw_std < 0.05).sum())

    # Oversampling exactly as in the paper pipeline
    fm = y_tr == 1
    nc = max(1, (~fm).sum() // fm.sum())
    np.random.seed(SEED + fold_i)
    X_over = np.tile(X_tr[fm], (nc, 1)) + np.random.randn(nc * fm.sum(), 80).astype(np.float32) * 0.05
    X_tr_os = np.vstack([X_tr, X_over])
    log("oversampled_fraction_of_training_set", len(X_over) / len(X_tr_os))

    # Two detector variants: "os" = paper pipeline, "orig" = no synthetic rows anywhere
    for tag, X_fit in [("os", X_tr_os), ("orig", X_tr)]:
        sc = StandardScaler().fit(X_fit)
        Xf = sc.transform(X_fit).astype(np.float64)
        lo, hi = Xf.min(axis=0), Xf.max(axis=0)
        mu = Xf.mean(axis=0)
        S = np.cov(Xf.T) + np.eye(80) * 1e-4
        Sinv = np.linalg.inv(S)
        eig = np.linalg.eigvalsh(S)
        log(f"{tag}/cov_min_eig", eig.min()); log(f"{tag}/cov_median_eig", np.median(eig))
        log(f"{tag}/n_eig_below_0.01", (eig < 0.01).sum())

        # CLAMP FIX: the clean reference is clamped the same way as every perturbed set
        Xc = np.clip(sc.transform(X_te).astype(np.float64), lo, hi)
        clean_d = mdist(Xc, mu, Sinv)
        pass_pool = Xc[y_te == 0]
        rng = np.random.RandomState(SEED + fold_i)

        for t in L2_GRID:
            # (a) isotropic random direction
            z = rng.randn(*Xc.shape); z = z / np.linalg.norm(z, axis=1, keepdims=True) * t
            Xn = np.clip(Xc + z, lo, hi); d = Xn - Xc
            log(f"{tag}/iso/L2={t}/auroc", auroc(clean_d, mdist(Xn, mu, Sinv)))
            log(f"{tag}/iso/L2={t}/actual_l2", np.linalg.norm(d, axis=1).mean())
            log(f"{tag}/iso/L2={t}/mahal_norm_delta", mnorm(d, Sinv).mean())

            # (b) on-manifold direction: step of length t toward a random real passing wafer
            tgt = pass_pool[rng.randint(len(pass_pool), size=len(Xc))]
            v = tgt - Xc; n = np.linalg.norm(v, axis=1, keepdims=True); n[n < 1e-9] = 1.0
            step = v / n * np.minimum(t, n)          # never overshoot the target wafer
            Xm = np.clip(Xc + step, lo, hi); d = Xm - Xc
            log(f"{tag}/toward_pass/L2={t}/auroc", auroc(clean_d, mdist(Xm, mu, Sinv)))
            log(f"{tag}/toward_pass/L2={t}/actual_l2", np.linalg.norm(d, axis=1).mean())
            log(f"{tag}/toward_pass/L2={t}/mahal_norm_delta", mnorm(d, Sinv).mean())
    print(f"fold {fold_i+1}/50 done")

m = lambda k: np.mean(res[k]); s = lambda k: np.std(res[k])
print("\n" + "=" * 78)
print(f"Features with raw std < 0.05 (of 80): {m('n_features_raw_std_below_0.05'):.1f}")
print(f"Synthetic rows as fraction of training set: {m('oversampled_fraction_of_training_set'):.2f}")
for tag, name in [("os", "DETECTOR FIT ON OVERSAMPLED DATA (paper pipeline)"), ("orig", "DETECTOR FIT ON ORIGINAL DATA")]:
    print("\n" + name)
    print(f"  cov eigenvalues: min {m(tag+'/cov_min_eig'):.2e}, median {m(tag+'/cov_median_eig'):.3f}, "
          f"count below 0.01: {m(tag+'/n_eig_below_0.01'):.1f}")
    print(f"  {'L2':>5} | {'isotropic AUROC':>17} {'||d||_M':>8} | {'toward-pass AUROC':>18} {'||d||_M':>8} {'L2 used':>8}")
    for t in L2_GRID:
        a, b = f"{tag}/iso/L2={t}", f"{tag}/toward_pass/L2={t}"
        print(f"  {t:>5} | {m(a+'/auroc'):>9.3f} +/- {s(a+'/auroc'):.3f} {m(a+'/mahal_norm_delta'):>8.2f} | "
              f"{m(b+'/auroc'):>10.3f} +/- {s(b+'/auroc'):.3f} {m(b+'/mahal_norm_delta'):>8.2f} {m(b+'/actual_l2'):>8.2f}")

with open("results/detector_geometry_results.json", "w") as f:
    json.dump({k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v), "values": v}
               for k, v in res.items()}, f, indent=2)
print("\nSaved results/detector_geometry_results.json")
