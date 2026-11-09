# ============================================================================
# SECOM adversarial detection study: full pipeline, single file
# Every number in the paper comes from runs of this file (see README).
#
# Changes from the original pipeline:
#   * OVERSAMPLE defaults to "none" (class weights only). "raw" reproduces the
#     original sigma=0.05 noise on unstandardized features for comparison.
#   * Detector, scaler ranges and clamp are fit on ORIGINAL training rows only.
#   * Clean test data is clamped exactly like perturbed data, so L_inf <= eps.
#   * Detector-aware PGD logs FAR, norms and a lambda sweep.
#   * Every attack logs L2, L_inf, Mahalanobis norm of delta, AUROC (all
#     samples and defectives only), and detector flag rate at 5% FPR.
#   * Isotropic and toward-passing-wafer sweeps are run on defectives.
#   * HSJA metrics no longer include the by-construction "success rate" alone;
#     norm and detector metrics are reported per fold.
# Kaggle: !pip install adversarial-robustness-toolbox --quiet
# ============================================================================
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import xgboost as xgb, time, json, os, warnings, copy
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from art.estimators.classification import BlackBoxClassifier
from art.attacks.evasion import HopSkipJump
warnings.filterwarnings("ignore")

# ------------------------------- configuration ------------------------------
SEED = 42
QUICK = os.environ.get("QUICK", "0") == "1"   # smoke test only
FOLD_STEP = 10 if QUICK else 1          # 1 = all 50 folds
HSJA_FOLD_STEP = int(os.environ.get("HSJA_FOLD_STEP", "10"))   # 1 = HSJA on all 50 folds (used for the paper)
HSJA_MAX_ITER = 2 if QUICK else 50
HSJA_MAX_EVAL = 200 if QUICK else 10000
HSJA_MAX_SAMPLES = 2 if QUICK else None # None = every correctly detected defective
HSJA_QUERY_CAP = 100000                 # per wafer; a normal run uses about 25,000
HSJA_TIME_CAP = 300                     # seconds per wafer; a stalled wafer is skipped and counted
MAX_FOLDS = int(os.environ.get("MAX_FOLDS", "0"))   # 0 = no limit (testing aid)
EXTRAS = os.environ.get("EXTRAS", "0") == "1"       # CNN feature-order ablation + stronger PGD; HSJA is skipped
RUN_HSJA = not EXTRAS
N_PERMS = 2 if QUICK else 5                         # random feature orders per fold for the CNN ablation
STRONG_STEPS, STRONG_RESTARTS = (10, 2) if QUICK else (100, 5)
MAX_EPOCHS = 8 if QUICK else 200
OVERSAMPLE = os.environ.get("OVERSAMPLE", "none")   # "none" | "post_std" | "raw"
POS_WEIGHT = 3.0
K = 80
EPS = 0.2
L2_BUDGET = 1.3
LAMBDAS = [0.01, 0.1, 1.0]              # detector-aware PGD penalty sweep
L2_GRID = [0.1, 0.2, 0.31, 0.5, 0.76, 1.0, 1.35, 1.8]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE} | OVERSAMPLE={OVERSAMPLE} | QUICK={QUICK}")

# ---------------------------------- data ------------------------------------
# Data: either the UCI raw files (data/secom.data + data/secom_labels.data) or the Kaggle CSV mirror.
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

# --------------------------------- models -----------------------------------
class MLP(nn.Module):
    def __init__(self, d=K):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(1)

class CNN(nn.Module):
    def __init__(self, d=K):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                                  nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2))
        self.fc = nn.Sequential(nn.Linear(64 * (d // 4), 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, x): return self.fc(self.conv(x.unsqueeze(1)).flatten(1)).squeeze(1)

def T(a): return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=DEVICE)

def train_nn(model, Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pw = torch.tensor(POS_WEIGHT, device=DEVICE)
    Xt, yt, Xv, yv = T(Xtr), T(ytr), T(Xva), T(yva)
    best, best_state, bad = np.inf, None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(Xt), device=DEVICE)
        for i in range(0, len(perm) - 1, 64):
            b = perm[i:i + 64]
            if len(b) < 2: continue
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(model(Xt[b]), yt[b], pos_weight=pw).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = F.binary_cross_entropy_with_logits(model(Xv), yv, pos_weight=pw).item()
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 25: break
    model.load_state_dict(best_state); model.eval()
    return model

def nn_prob(model, X):
    with torch.no_grad(): return torch.sigmoid(model(T(X))).cpu().numpy()

def tune_threshold(probs, y):
    best_t, best_b = 0.5, 0
    for t in np.arange(0.05, 0.95, 0.01):
        b = balanced_accuracy_score(y, (probs >= t).astype(int))
        if b > best_b: best_b, best_t = b, t
    return float(best_t)

def far_frr(y, pred):
    d, p = y == 1, y == 0
    return float((pred[d] == 0).mean()), float((pred[p] == 1).mean())   # FAR: defective accepted; FRR: passing rejected

# --------------------------------- attacks ----------------------------------
def pgd_linf(model, X, y, lo, hi, eps=EPS, steps=20, alpha=None, lam=0.0, mu=None, Sinv=None, seed=0):
    """L_inf PGD on the classifier loss. lam > 0 adds the detector penalty: loss = BCE - lam * D_M(x)."""
    alpha = eps / 5 if alpha is None else alpha
    g = torch.Generator(device="cpu").manual_seed(seed)
    x0, yt, lo_t, hi_t = T(X), T(y), T(lo), T(hi)
    x = x0 + (torch.rand(x0.shape, generator=g).to(DEVICE) * 2 - 1) * eps
    x = torch.min(torch.max(x, lo_t), hi_t)
    for _ in range(steps):
        x = x.detach().requires_grad_(True)
        loss = F.binary_cross_entropy_with_logits(model(x), yt, reduction="sum")
        if lam > 0:
            dlt = x - mu
            loss = loss - lam * torch.sqrt(((dlt @ Sinv) * dlt).sum(1) + 1e-12).sum()
        grad, = torch.autograd.grad(loss, x)
        x = x.detach() + alpha * grad.sign()
        x = torch.min(torch.max(x, x0 - eps), x0 + eps)
        x = torch.min(torch.max(x, lo_t), hi_t)
    return x.detach().cpu().numpy()

def pgd_strong(model, X, y, lo, hi, seed=0):
    """Convergence check for PGD: more steps, smaller step, several random restarts, worst case kept per wafer."""
    yt = T(y); best, best_m = None, None
    for r in range(STRONG_RESTARTS):
        P = pgd_linf(model, X, y, lo, hi, steps=STRONG_STEPS, alpha=EPS / 10, seed=10007 * (r + 1) + seed)
        with torch.no_grad(): z = model(T(P))
        mgn = torch.where(yt > 0.5, -z, z).cpu().numpy()      # larger = further toward the wrong side
        if best is None: best, best_m = P, mgn
        else:
            k = mgn > best_m; best[k] = P[k]; best_m[k] = mgn[k]
    return best

def pgd_l2(model, X, y, lo, hi, budget=L2_BUDGET, steps=20):
    x0, yt, lo_t, hi_t = T(X), T(y), T(lo), T(hi)
    alpha = 2.5 * budget / steps
    x = x0.clone()
    for _ in range(steps):
        x = x.detach().requires_grad_(True)
        grad, = torch.autograd.grad(F.binary_cross_entropy_with_logits(model(x), yt, reduction="sum"), x)
        d = x.detach() - x0 + alpha * grad / (grad.norm(dim=1, keepdim=True) + 1e-12)
        d = d * torch.clamp(budget / (d.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
        x = torch.min(torch.max(x0 + d, lo_t), hi_t)
    return x.detach().cpu().numpy()

class HSJABudget(Exception): pass

def run_hsja(pred_fn, x, x_init, lo, hi):
    """HSJA on ONE wafer with a hard query and time budget. Returns (adv or None, queries used)."""
    st = {"n": 0, "t": time.time()}
    def predict(Xq):
        st["n"] += len(Xq)
        if st["n"] > HSJA_QUERY_CAP or time.time() - st["t"] > HSJA_TIME_CAP: raise HSJABudget()
        return np.eye(2, dtype=np.float32)[pred_fn(np.asarray(Xq, dtype=np.float32))]
    clf = BlackBoxClassifier(predict, input_shape=(K,), nb_classes=2, clip_values=(lo, hi))
    atk = HopSkipJump(clf, targeted=False, norm=2, max_iter=HSJA_MAX_ITER, max_eval=HSJA_MAX_EVAL,
                      init_eval=min(100, HSJA_MAX_EVAL), verbose=False)
    try:
        adv = atk.generate(x=x[None].astype(np.float32), x_adv_init=x_init[None].astype(np.float32))
    except HSJABudget:
        return None, st["n"]
    return np.clip(adv[0], lo, hi).astype(np.float32), st["n"]

def make_hsja_ctx(fold_i, mlp, xg, thr, Xc, y_te, clean_d, tau, lo, hi, mu, Sinv):
    """Freeze everything HSJA needs for this fold. The MLP copy is CPU float64 so that a wafer sitting on
    the decision boundary gets the same label whether it is queried alone or in a batch."""
    mlp64 = copy.deepcopy(mlp).cpu().double().eval(); t_mlp, t_xgb = thr["MLP"], thr["XGB"]
    def p_mlp(X):
        with torch.no_grad():
            return (torch.sigmoid(mlp64(torch.as_tensor(np.asarray(X), dtype=torch.float64))).numpy() >= t_mlp).astype(int)
    def p_xgb(X): return (xg.predict_proba(np.asarray(X, dtype=np.float32))[:, 1] >= t_xgb).astype(int)
    def md(X): D = np.asarray(X, dtype=np.float64) - mu; return np.sqrt(np.einsum("ij,jk,ik->i", D, Sinv, D))
    def mn(D): D = np.asarray(D, dtype=np.float64); return np.sqrt(np.einsum("ij,jk,ik->i", D, Sinv, D))
    return dict(fold=fold_i, pred={"MLP": p_mlp, "XGB": p_xgb}, Xc=Xc, y=y_te, clean_d=clean_d, tau=tau,
                lo=lo, hi=hi, md=md, mn=mn)

# ------------------------------- bookkeeping --------------------------------
res = {}
hsja_queue = []
def say(msg): print(f"[{(time.time()-t0)/60:6.1f} min] {msg}", flush=True)
def save():
    with open(OUT, "w") as f:
        json.dump({k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v), "values": v} for k, v in res.items()}, f, indent=1)
def log(key, val):
    if val is not None and np.isfinite(val): res.setdefault(key, []).append(float(val))

def auroc(clean, pert):
    if len(clean) == 0 or len(pert) == 0: return None
    return float(roc_auc_score(np.r_[np.zeros(len(clean)), np.ones(len(pert))], np.r_[clean, pert]))

# --------------------------------- main loop --------------------------------
t0 = time.time(); os.makedirs("results", exist_ok=True); OUT = f"results/secom_results_{OVERSAMPLE}{'_extras' if EXTRAS else ''}.json"; n_done = 0
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
for fold_i, (tr_idx, te_idx) in enumerate(rskf.split(X_full, y_raw)):
    if fold_i % FOLD_STEP != 0: continue
    if MAX_FOLDS and n_done >= MAX_FOLDS: break
    n_done += 1
    rng = np.random.RandomState(SEED + fold_i)
    y_tr, y_te = y_raw[tr_idx], y_raw[te_idx]

    # fold-internal preprocessing
    imp = SimpleImputer(strategy="median")
    X_tr_imp = imp.fit_transform(X_full[tr_idx]); X_te_imp = imp.transform(X_full[te_idx])
    X_tr_imp, X_va_imp, y_tr, y_va = train_test_split(X_tr_imp, y_tr, test_size=0.1,
                                                      random_state=SEED + fold_i, stratify=y_tr)
    sel = SelectKBest(f_classif, k=K).fit(X_tr_imp, y_tr)
    cols = np.where(sel.get_support())[0]
    cols = cols[np.argsort(-np.nan_to_num(sel.scores_[cols]))]      # ANOVA rank order (used by the CNN)
    X_tr, X_va, X_te = X_tr_imp[:, cols], X_va_imp[:, cols], X_te_imp[:, cols]

    fm = y_tr == 1; nc = max(1, (~fm).sum() // fm.sum())
    def synth(X_def, sigma): return np.tile(X_def, (nc, 1)) + rng.randn(nc * len(X_def), K) * sigma
    if OVERSAMPLE == "raw":            # original behaviour: noise on unstandardized features
        scaler = StandardScaler().fit(np.vstack([X_tr, synth(X_tr[fm], 0.05)]))
        X_fit = np.vstack([scaler.transform(X_tr), scaler.transform(synth(X_tr[fm], 0.05))])
        y_fit = np.r_[y_tr, np.ones(nc * fm.sum(), dtype=int)]
    else:
        scaler = StandardScaler().fit(X_tr)
        X_fit, y_fit = scaler.transform(X_tr), y_tr
        if OVERSAMPLE == "post_std":   # noise added after standardization
            X_fit = np.vstack([X_fit, synth(X_fit[fm], 0.05)]); y_fit = np.r_[y_fit, np.ones(nc * fm.sum(), dtype=int)]
    X_fit = X_fit.astype(np.float32)

    # ranges and detector from ORIGINAL training rows only
    X_orig = scaler.transform(X_tr)
    lo, hi = X_orig.min(axis=0).astype(np.float32), X_orig.max(axis=0).astype(np.float32)
    mu = X_orig.mean(axis=0); Sinv = np.linalg.inv(np.cov(X_orig.T) + np.eye(K) * 1e-4)
    mu_t, Sinv_t = T(mu), T(Sinv)
    def mdist(X): D = np.asarray(X, dtype=np.float64) - mu; return np.sqrt(np.einsum("ij,jk,ik->i", D, Sinv, D))
    def mnorm(D): D = np.asarray(D, dtype=np.float64); return np.sqrt(np.einsum("ij,jk,ik->i", D, Sinv, D))

    X_va_sc = np.clip(scaler.transform(X_va), lo, hi).astype(np.float32)
    Xc = np.clip(scaler.transform(X_te), lo, hi).astype(np.float32)       # clamped clean reference
    clean_d = mdist(Xc); tau = np.percentile(mdist(X_va_sc), 95)          # detector threshold at 5% FPR on validation
    DEF, PAS = y_te == 1, y_te == 0

    say(f"fold {fold_i+1}/50: training")
    # train
    mlp = train_nn(MLP(), X_fit, y_fit, X_va_sc, y_va, SEED + fold_i)
    cnn = train_nn(CNN(), X_fit, y_fit, X_va_sc, y_va, SEED + fold_i)
    xg = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, scale_pos_weight=POS_WEIGHT,
                           eval_metric="logloss", random_state=SEED + fold_i, early_stopping_rounds=20)
    xg.fit(X_fit, y_fit, eval_set=[(X_va_sc, y_va)], verbose=False)
    thr = {"MLP": tune_threshold(nn_prob(mlp, X_va_sc), y_va), "CNN": tune_threshold(nn_prob(cnn, X_va_sc), y_va),
           "XGB": tune_threshold(xg.predict_proba(X_va_sc)[:, 1], y_va)}
    pred = {"MLP": lambda X: (nn_prob(mlp, X) >= thr["MLP"]).astype(int),
            "CNN": lambda X: (nn_prob(cnn, X) >= thr["CNN"]).astype(int),
            "XGB": lambda X: (xg.predict_proba(np.asarray(X, dtype=np.float32))[:, 1] >= thr["XGB"]).astype(int)}

    def classify(name, P, models):
        dist = mdist(P)
        for m in models:
            pr = pred[m](P); far, frr = far_frr(y_te, pr)
            log(f"{name}/{m}/FAR", far); log(f"{name}/{m}/FRR", frr)
            # defectives that are accepted by the classifier AND not flagged by the detector
            log(f"{name}/{m}/defectives_accepted_and_unflagged", ((pr[DEF] == 0) & (dist[DEF] <= tau)).mean())

    def geometry(name, P, ref=None, ref_d=None, mask=None):
        ref = Xc if ref is None else ref; ref_d = clean_d if ref_d is None else ref_d
        d = P - ref; dist = mdist(P)
        log(f"{name}/L2", np.linalg.norm(d, axis=1).mean()); log(f"{name}/Linf", np.abs(d).max(axis=1).mean())
        log(f"{name}/mahal_norm_delta", mnorm(d).mean())
        log(f"{name}/auroc_all_clean_vs_perturbed", auroc(clean_d, dist))
        if mask is not None:
            log(f"{name}/auroc_defectives_only", auroc(ref_d[mask], dist[mask]))
            log(f"{name}/detector_flag_rate_defectives", (dist[mask] > tau).mean())
        else:
            log(f"{name}/auroc_paired", auroc(ref_d, dist)); log(f"{name}/detector_flag_rate", (dist > tau).mean())

    # clean baselines
    for m in pred:
        log(f"clean/{m}/bal_acc", balanced_accuracy_score(y_te, pred[m](Xc))); log(f"clean/{m}/threshold", thr[m])
    classify("clean", Xc, ["MLP", "CNN", "XGB"])
    log("clean/detector_flag_rate_all", (clean_d > tau).mean()); log("clean/detector_flag_rate_defectives", (clean_d[DEF] > tau).mean())
    log("clean/auroc_defective_vs_passing", auroc(clean_d[PAS], clean_d[DEF]))

    say(f"fold {fold_i+1}/50: attacks")
    # gradient attacks on all test wafers, plus transfer to XGB
    P = pgd_linf(mlp, Xc, y_te, lo, hi, seed=fold_i); geometry("pgd_linf_mlp", P, mask=DEF); classify("pgd_linf_mlp", P, ["MLP", "XGB"])
    P = pgd_linf(cnn, Xc, y_te, lo, hi, seed=fold_i); geometry("pgd_linf_cnn", P, mask=DEF); classify("pgd_linf_cnn", P, ["CNN", "XGB"])
    P = pgd_l2(mlp, Xc, y_te, lo, hi);                geometry("pgd_l2_mlp", P, mask=DEF);   classify("pgd_l2_mlp", P, ["MLP", "XGB"])
    P = np.clip(Xc + rng.choice([-1.0, 1.0], size=Xc.shape).astype(np.float32) * EPS, lo, hi)
    geometry("random_sign", P, mask=DEF); classify("random_sign", P, ["MLP", "CNN", "XGB"])
    for lam in LAMBDAS:
        P = pgd_linf(mlp, Xc, y_te, lo, hi, steps=40, alpha=0.01, lam=lam, mu=mu_t, Sinv=Sinv_t, seed=fold_i)
        geometry(f"detector_aware_pgd_lam{lam}", P, mask=DEF); classify(f"detector_aware_pgd_lam{lam}", P, ["MLP", "XGB"])

    # geometry sweeps on defective wafers: isotropic vs toward a real passing wafer
    Xd, dd = Xc[DEF], clean_d[DEF]; pool = Xc[PAS]
    for t in L2_GRID:
        z = rng.randn(*Xd.shape); z = z / np.linalg.norm(z, axis=1, keepdims=True) * t
        Pi = np.clip(Xd + z, lo, hi)
        v = pool[rng.randint(len(pool), size=len(Xd))] - Xd; n = np.linalg.norm(v, axis=1, keepdims=True); n[n < 1e-9] = 1.0
        Pt = np.clip(Xd + v / n * np.minimum(t, n), lo, hi)
        for nm, Pp in [("sweep_isotropic", Pi), ("sweep_toward_pass", Pt)]:
            log(f"{nm}/L2={t}/auroc_defectives_only", auroc(dd, mdist(Pp)))
            log(f"{nm}/L2={t}/actual_L2", np.linalg.norm(Pp - Xd, axis=1).mean())
            log(f"{nm}/L2={t}/mahal_norm_delta", mnorm(Pp - Xd).mean())
            for m in ["MLP", "XGB"]: log(f"{nm}/L2={t}/{m}/FAR", (pred[m](Pp) == 0).mean())

    if EXTRAS:
        say(f"fold {fold_i+1}/50: extras")
        for nm, mdl, mm in [("pgd_strong_mlp", mlp, "MLP"), ("pgd_strong_cnn", cnn, "CNN")]:
            P = pgd_strong(mdl, Xc, y_te, lo, hi, seed=fold_i); geometry(nm, P, mask=DEF); classify(nm, P, [mm])
        # CNN feature-order ablation: retrain with random column orders, same data and seed otherwise
        prng = np.random.RandomState(SEED + 5000 + fold_i); accs, fars, frrs = [], [], []
        for _ in range(N_PERMS):
            perm = prng.permutation(K)
            c2 = train_nn(CNN(), X_fit[:, perm], y_fit, X_va_sc[:, perm], y_va, SEED + fold_i)
            t2 = tune_threshold(nn_prob(c2, X_va_sc[:, perm]), y_va)
            p2 = (nn_prob(c2, Xc[:, perm]) >= t2).astype(int); fa, fr = far_frr(y_te, p2)
            accs.append(balanced_accuracy_score(y_te, p2)); fars.append(fa); frrs.append(fr)
        ordered = balanced_accuracy_score(y_te, pred["CNN"](Xc))
        log("cnn_perm/bal_acc_mean_of_perms", np.mean(accs)); log("cnn_perm/FAR_mean_of_perms", np.mean(fars)); log("cnn_perm/FRR_mean_of_perms", np.mean(frrs))
        log("cnn_perm/bal_acc_sd_across_perms", np.std(accs)); log("cnn_perm/bal_acc_anova_order", ordered)
        log("cnn_perm/bal_acc_perm_minus_anova", np.mean(accs) - ordered)
    if RUN_HSJA and fold_i % HSJA_FOLD_STEP == 0:        # HSJA runs after all folds, see phase 2
        hsja_queue.append(make_hsja_ctx(fold_i, mlp, xg, thr, Xc, y_te, clean_d, tau, lo, hi, mu, Sinv))
    save()
    say(f"fold {fold_i+1}/50 done, results saved")

# --------------------------------- summary ----------------------------------
def summary():
    def g(k): return f"{np.mean(res[k]):.3f}+/-{np.std(res[k]):.3f}" if k in res else "   n/a   "
    print("\n" + "=" * 100 + f"\nSUMMARY (OVERSAMPLE={OVERSAMPLE}, mean +/- sd over folds)\n" + "=" * 100)
    print("\nTable I: baselines")
    for m in ["MLP", "CNN", "XGB"]:
        print(f"  {m}: bal_acc {g(f'clean/{m}/bal_acc')}  FAR {g(f'clean/{m}/FAR')}  FRR {g(f'clean/{m}/FRR')}  thr {g(f'clean/{m}/threshold')}")
    print("\nTable II: effect on XGB (transfer and random control)")
    for a in ["clean", "pgd_linf_mlp", "pgd_linf_cnn", "random_sign"]:
        print(f"  {a:<16} XGB FAR {g(f'{a}/XGB/FAR')}  FRR {g(f'{a}/XGB/FRR')}")
    print("\nTable III/IV: detection and geometry (AUROC on defectives only is the evasion-relevant one)")
    print(f"  {'attack':<28}{'target FAR':>17}{'AUROC all':>17}{'AUROC def':>17}{'L2':>17}{'Linf':>17}{'||d||_M':>17}{'acc+unflag':>17}")
    for a, m in [("pgd_linf_mlp", "MLP"), ("pgd_linf_cnn", "CNN"), ("pgd_l2_mlp", "MLP"), ("random_sign", "MLP")] + \
                [(f"detector_aware_pgd_lam{l}", "MLP") for l in LAMBDAS]:
        print(f"  {a:<28}{g(f'{a}/{m}/FAR'):>17}{g(f'{a}/auroc_all_clean_vs_perturbed'):>17}{g(f'{a}/auroc_defectives_only'):>17}"
              f"{g(f'{a}/L2'):>17}{g(f'{a}/Linf'):>17}{g(f'{a}/mahal_norm_delta'):>17}{g(f'{a}/{m}/defectives_accepted_and_unflagged'):>17}")
    for a in ["hsja_mlp", "hsja_xgb"]:
        print(f"  {a:<28}{'(by constr.)':>17}{g(f'{a}/auroc_all_clean_vs_perturbed'):>17}{g(f'{a}/auroc_paired'):>17}"
              f"{g(f'{a}/L2'):>17}{g(f'{a}/Linf'):>17}{g(f'{a}/mahal_norm_delta'):>17}{g(f'{a}/accepted_and_unflagged'):>17}"
              f"   queries {g(f'{a}/queries_per_sample')}  n/fold {g(f'{a}/n_attacked')}")
    print(f"  clean baseline: MLP defectives accepted and unflagged {g('clean/MLP/defectives_accepted_and_unflagged')}, "
          f"detector flag rate on clean defectives {g('clean/detector_flag_rate_defectives')}, "
          f"AUROC clean defective vs passing {g('clean/auroc_defective_vs_passing')}")
    print("\nFigure: AUROC vs L2 on defectives (isotropic vs toward a passing wafer)")
    print(f"  {'L2':>5}{'isotropic':>16}{'||d||_M':>17}{'MLP FAR':>17}{'toward pass':>16}{'||d||_M':>17}{'MLP FAR':>17}")
    for t in L2_GRID:
        a, b = f"sweep_isotropic/L2={t}", f"sweep_toward_pass/L2={t}"
        print(f"  {t:>5}{g(a+'/auroc_defectives_only'):>16}{g(a+'/mahal_norm_delta'):>17}{g(a+'/MLP/FAR'):>17}"
              f"{g(b+'/auroc_defectives_only'):>16}{g(b+'/mahal_norm_delta'):>17}{g(b+'/MLP/FAR'):>17}")

save(); summary()
if EXTRAS:
    print("\nExtras")
    def gg(k): return f"{np.mean(res[k]):.3f}+/-{np.std(res[k]):.3f}" if k in res else "n/a"
    for mm in ["MLP", "CNN"]:
        a, b = f"pgd_linf_{mm.lower()}", f"pgd_strong_{mm.lower()}"
        print(f"  {mm}: PGD-20 FAR {gg(f'{a}/{mm}/FAR')}  vs  PGD-{STRONG_STEPS} x {STRONG_RESTARTS} restarts FAR {gg(f'{b}/{mm}/FAR')}"
              f"   flagged {gg(b+'/detector_flag_rate_defectives')}  joint evasion {gg(f'{b}/{mm}/defectives_accepted_and_unflagged')}")
    print(f"  CNN bal. acc.: ANOVA order {gg('cnn_perm/bal_acc_anova_order')}, random orders {gg('cnn_perm/bal_acc_mean_of_perms')}, "
          f"difference {gg('cnn_perm/bal_acc_perm_minus_anova')}, SD across orders within a fold {gg('cnn_perm/bal_acc_sd_across_perms')}")
say("PHASE 1 COMPLETE. Everything except HSJA is saved." + (" Starting HSJA." if RUN_HSJA else " HSJA skipped (EXTRAS run)."))

# ------------------------- phase 2: HSJA, one wafer at a time ----------------
for c in hsja_queue:
    rng = np.random.RandomState(SEED + 1000 + c["fold"])
    Xc, y, cd = c["Xc"], c["y"], c["clean_d"]
    for m in ["MLP", "XGB"]:
        name = f"hsja_{m.lower()}"; pc = c["pred"][m](Xc)
        idx = np.where((y == 1) & (pc == 1))[0][:HSJA_MAX_SAMPLES]
        pool = Xc[(y == 0) & (pc == 0)]
        if len(idx) == 0 or len(pool) == 0: continue
        advs, kept, stalled = [], [], 0
        for j, i in enumerate(idx):
            adv, q = run_hsja(c["pred"][m], Xc[i], pool[rng.randint(len(pool))], c["lo"], c["hi"])
            if adv is None: stalled += 1; say(f"  fold {c['fold']+1} {m} wafer {j+1}/{len(idx)}: hit budget after {q} queries, skipped")
            else: advs.append(adv); kept.append(i); log(f"{name}/queries_per_sample", q)
        log(f"{name}/n_attacked", len(idx)); log(f"{name}/n_stalled", stalled)
        if advs:
            A, kept = np.array(advs), np.array(kept); d = A - Xc[kept]; dist = c["md"](A); pa = c["pred"][m](A)
            log(f"{name}/L2", np.linalg.norm(d, axis=1).mean()); log(f"{name}/Linf", np.abs(d).max(axis=1).mean())
            log(f"{name}/mahal_norm_delta", c["mn"](d).mean())
            log(f"{name}/auroc_all_clean_vs_perturbed", auroc(cd, dist)); log(f"{name}/auroc_paired", auroc(cd[kept], dist))
            log(f"{name}/detector_flag_rate", (dist > c["tau"]).mean()); log(f"{name}/success_rate", (pa == 0).mean())
            log(f"{name}/accepted_and_unflagged", ((pa == 0) & (dist <= c["tau"])).mean())
        save(); say(f"fold {c['fold']+1} HSJA on {m}: {len(advs)}/{len(idx)} wafers done, saved")

summary()
print(f"  HSJA wafers skipped for hitting the budget: MLP {int(np.sum(res.get('hsja_mlp/n_stalled', [0])))}, XGB {int(np.sum(res.get('hsja_xgb/n_stalled', [0])))}")
say(f"ALL DONE. Saved {OUT}")
