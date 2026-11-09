# Detector-only replay of the Fig. 1 sweeps from secom_main.py (OVERSAMPLE=none). No models needed:
# Usage: python fig1_sweep.py [path/to/folder containing secom.data and secom_labels.data]
# the sweep AUROCs depend only on preprocessing and the per-fold random stream, so this is CPU-deterministic.
import numpy as np, json, sys, warnings
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
SEED, K = 42, 80
GRID = [0.1, 0.2, 0.31, 0.5, 0.76, 1.0, 1.35, 1.8]      # original grid, replayed first so old numbers are unchanged
EXTRA = [0.54, 1.20]                                     # HSJA mean L2: XGBoost target, MLP target
d = sys.argv[1] if len(sys.argv) > 1 else "data"
X_raw = np.genfromtxt(d + "/secom.data").astype(np.float32)
y_raw = ((np.genfromtxt(d + "/secom_labels.data", usecols=0) + 1) / 2).astype(int)
X_full = X_raw[:, np.isnan(X_raw).mean(axis=0) < 0.5]
X_full = X_full[:, np.nanstd(X_full, axis=0) > 1e-8]
res = {}
def log(k, v): res.setdefault(k, []).append(float(v))
def auroc(c, p): return roc_auc_score(np.r_[np.zeros(len(c)), np.ones(len(p))], np.r_[c, p])
for fold_i, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED).split(X_full, y_raw)):
    rng = np.random.RandomState(SEED + fold_i)
    y_tr, y_te = y_raw[tr], y_raw[te]
    imp = SimpleImputer(strategy="median"); X_tr_imp = imp.fit_transform(X_full[tr]); X_te_imp = imp.transform(X_full[te])
    X_tr_imp, _, y_tr, _ = train_test_split(X_tr_imp, y_tr, test_size=0.1, random_state=SEED + fold_i, stratify=y_tr)
    sel = SelectKBest(f_classif, k=K).fit(X_tr_imp, y_tr)
    cols = np.where(sel.get_support())[0]; cols = cols[np.argsort(-np.nan_to_num(sel.scores_[cols]))]
    X_tr, X_te = X_tr_imp[:, cols], X_te_imp[:, cols]
    sc = StandardScaler().fit(X_tr); Xo = sc.transform(X_tr)
    lo, hi = Xo.min(axis=0).astype(np.float32), Xo.max(axis=0).astype(np.float32)
    mu = Xo.mean(axis=0); Sinv = np.linalg.inv(np.cov(Xo.T) + np.eye(K) * 1e-4)
    md = lambda X: (lambda D: np.sqrt(np.einsum("ij,jk,ik->i", D, Sinv, D)))(np.asarray(X, dtype=np.float64) - mu)
    mn = lambda D: (lambda D: np.sqrt(np.einsum("ij,jk,ik->i", D, Sinv, D)))(np.asarray(D, dtype=np.float64))
    Xc = np.clip(sc.transform(X_te), lo, hi).astype(np.float32); cd = md(Xc)
    rng.choice([-1.0, 1.0], size=Xc.shape)               # random-sign draw happens here in secom_main.py
    Xd, dd, pool = Xc[y_te == 1], cd[y_te == 1], Xc[y_te == 0]
    for t in GRID + EXTRA:
        z = rng.randn(*Xd.shape); z = z / np.linalg.norm(z, axis=1, keepdims=True) * t
        Pi = np.clip(Xd + z, lo, hi)
        v = pool[rng.randint(len(pool), size=len(Xd))] - Xd; n = np.linalg.norm(v, axis=1, keepdims=True); n[n < 1e-9] = 1.0
        Pt = np.clip(Xd + v / n * np.minimum(t, n), lo, hi)
        for nm, P in [("sweep_isotropic", Pi), ("sweep_toward_pass", Pt)]:
            log(f"{nm}/L2={t}/auroc_defectives_only", auroc(dd, md(P)))
            log(f"{nm}/L2={t}/actual_L2", np.linalg.norm(P - Xd, axis=1).mean())
            log(f"{nm}/L2={t}/mahal_norm_delta", mn(P - Xd).mean())
json.dump({k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v), "values": v} for k, v in res.items()},
          open("results/fig1_sweep_results.json", "w"), indent=1)
