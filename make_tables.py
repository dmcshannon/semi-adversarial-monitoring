"""Regenerate every number in the paper from the result files in results/.

Usage:  python make_tables.py            (prints the tables as Markdown)
        python make_tables.py --check paper/mlwa_elsarticle.tex
                                          (also checks that each value appears in the manuscript)
"""
import json, re, sys, os

R = "results"
N = json.load(open(f"{R}/secom_results_none.json"))          # main pipeline, class weights only, HSJA on all folds
RW = json.load(open(f"{R}/secom_results_raw.json"))          # comparison pipeline, raw-feature oversampling
E = json.load(open(f"{R}/secom_results_none_extras.json"))   # PGD convergence check, CNN feature-order ablation
G = json.load(open(f"{R}/detector_geometry_results.json"))   # covariance spectrum, raw feature scales
S = json.load(open(f"{R}/fig1_sweep_results.json"))          # Fig. 1 sweeps incl. the two HSJA-matched norms

def m(d, k, p=3): return f"{d[k]['mean']:.{p}f}"
def ms(d, k, p=3): return f"{d[k]['mean']:.{p}f} ± {d[k]['std']:.{p}f}"
def pc(d, k): return f"{100*d[k]['mean']:.1f}"
def pcs(d, k): return f"{100*d[k]['std']:.1f}"
tot = lambda k: int(sum(N[k]["values"]))

out = []
def P(*a): out.append(" ".join(str(x) for x in a))

P("## Table 1: clean baselines (mean ± SD over 50 folds)\n")
P("| Model | Bal. Acc. | FAR | FRR |\n|---|---|---|---|")
for mm in ["MLP", "CNN", "XGB"]:
    P(f"| {mm} | {ms(N,f'clean/{mm}/bal_acc')} | {ms(N,f'clean/{mm}/FAR')} | {ms(N,f'clean/{mm}/FRR')} |")

P("\n## Table 2: classifier evasion and detector response (MLP target unless noted)\n")
P("| Perturbation | FAR | AUROC (all) | AUROC (defective) | Flagged (%) | Joint evasion | L2 | Linf | ||d||_M |\n|---|---|---|---|---|---|---|---|---|")
P(f"| None (clean) | {ms(N,'clean/MLP/FAR')} | -- | -- | {pc(N,'clean/detector_flag_rate_defectives')} ± {pcs(N,'clean/detector_flag_rate_defectives')} | {ms(N,'clean/MLP/defectives_accepted_and_unflagged')} | 0 | 0 | 0 |")
rows = [("Random sign","random_sign"),("PGD Linf","pgd_linf_mlp"),("PGD L2","pgd_l2_mlp"),
        ("Det.-aware PGD, lambda=0.01","detector_aware_pgd_lam0.01"),("Det.-aware PGD, lambda=0.1","detector_aware_pgd_lam0.1"),("Det.-aware PGD, lambda=1","detector_aware_pgd_lam1.0")]
for lab, a in rows:
    P(f"| {lab} | {ms(N,f'{a}/MLP/FAR')} | {m(N,a+'/auroc_all_clean_vs_perturbed')} | {ms(N,a+'/auroc_defectives_only')} | {pc(N,a+'/detector_flag_rate_defectives')} ± {pcs(N,a+'/detector_flag_rate_defectives')} | {ms(N,f'{a}/MLP/defectives_accepted_and_unflagged')} | {m(N,a+'/L2',2)} | {m(N,a+'/Linf',2)} | {m(N,a+'/mahal_norm_delta',1)} |")
for lab, a in [("HopSkipJump (MLP target)","hsja_mlp"),("HopSkipJump (XGB target)","hsja_xgb")]:
    P(f"| {lab} | 1.000* | {m(N,a+'/auroc_all_clean_vs_perturbed')} | {ms(N,a+'/auroc_paired')} | {pc(N,a+'/detector_flag_rate')} ± {pcs(N,a+'/detector_flag_rate')} | {ms(N,a+'/accepted_and_unflagged')}* | {m(N,a+'/L2',2)} | {m(N,a+'/Linf',2)} | {m(N,a+'/mahal_norm_delta',1)} |")
P("\n\\* by construction; see the paper's table note.")
P(f"\nHopSkipJump wafers: {tot('hsja_mlp/n_attacked')} against the MLP; {tot('hsja_xgb/n_attacked')-tot('hsja_xgb/n_stalled')} of {tot('hsja_xgb/n_attacked')} completed against XGBoost ({tot('hsja_xgb/n_stalled')} hit the budget). Queries per completed wafer: {N['hsja_mlp/queries_per_sample']['mean']:,.0f}.")

P("\n## Table 3: XGBoost under perturbations not crafted against it\n")
P("| Source | FAR (no oversampling) | FRR | FAR (raw-feature oversampling) | FRR |\n|---|---|---|---|---|")
for lab, a in [("None (clean)","clean"),("MLP PGD Linf","pgd_linf_mlp"),("CNN PGD Linf","pgd_linf_cnn"),("Random sign","random_sign")]:
    P(f"| {lab} | {ms(N,f'{a}/XGB/FAR')} | {ms(N,f'{a}/XGB/FRR')} | {ms(RW,f'{a}/XGB/FAR')} | {ms(RW,f'{a}/XGB/FRR')} |")
P(f"\nDetector-aware PGD (lambda=0.1) in the oversampling pipeline: FAR {m(RW,'detector_aware_pgd_lam0.1/MLP/FAR')}, flagged {pc(RW,'detector_aware_pgd_lam0.1/detector_flag_rate_defectives')}%, joint evasion {m(RW,'detector_aware_pgd_lam0.1/MLP/defectives_accepted_and_unflagged')}.")

P("\n## Figure 1 data: detector AUROC on defective wafers vs L2\n")
P("| L2 | Isotropic AUROC | ||d||_M | Toward-pass AUROC | ||d||_M | MLP FAR (toward-pass) |\n|---|---|---|---|---|---|")
for t in [0.1, 0.2, 0.31, 0.5, 0.54, 0.76, 1.0, 1.2, 1.35, 1.8]:
    a, b = f"sweep_isotropic/L2={t}", f"sweep_toward_pass/L2={t}"
    far = ms(N, b+"/MLP/FAR") if b+"/MLP/FAR" in N else "--"
    P(f"| {t} | {ms(S,a+'/auroc_defectives_only')} | {m(S,a+'/mahal_norm_delta',1)} | {ms(S,b+'/auroc_defectives_only')} | {m(S,b+'/mahal_norm_delta',2)} | {far} |")

P("\n## Other quantities quoted in the text\n")
P(f"- PGD convergence check (separate run): MLP FAR {m(E,'pgd_strong_mlp/MLP/FAR')} (100 steps, 5 restarts) vs {m(E,'pgd_linf_mlp/MLP/FAR')} (20 steps); CNN {m(E,'pgd_strong_cnn/CNN/FAR')} vs {m(E,'pgd_linf_cnn/CNN/FAR')}; flagged {pc(E,'pgd_strong_mlp/detector_flag_rate_defectives')}%")
P(f"- CNN feature order: random orders {ms(E,'cnn_perm/bal_acc_mean_of_perms')} vs ANOVA order {ms(E,'cnn_perm/bal_acc_anova_order')}, difference {ms(E,'cnn_perm/bal_acc_perm_minus_anova')}")
P(f"- Covariance spectrum: {G['orig/n_eig_below_0.01']['mean']:.1f} of 80 eigenvalues below 0.01, median eigenvalue {G['orig/cov_median_eig']['mean']:.2f}")
P(f"- Features with raw SD below 0.05: {G['n_features_raw_std_below_0.05']['mean']:.1f} of 80; synthetic rows {100*G['oversampled_fraction_of_training_set']['mean']:.0f}% of the oversampled training set")
P(f"- Clean MLP FAR in the extras run: {m(E,'clean/MLP/FAR')} (main run {m(N,'clean/MLP/FAR')}), the run-to-run variation quoted in Limitations")

text = "\n".join(out)
print(text)
if len(sys.argv) > 2 and sys.argv[1] == "--check":
    tex = open(sys.argv[2]).read().replace("$\\pm$", "±")
    nums = sorted(set(re.findall(r"\d+\.\d+", text)))
    missing = [n for n in nums if n not in tex]
    print(f"\n[check] {len(nums)} distinct values in the tables; {len(missing)} not found in {sys.argv[2]}: {missing}")
