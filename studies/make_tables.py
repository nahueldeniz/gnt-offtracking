"""Generate the per-path result tables (S2 estimator, S4 scaling, S5 robustness)."""
import json
import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")
PAPER = os.path.join(ROOT, "paper")
os.makedirs(PAPER, exist_ok=True)
def load(n):
    p = os.path.join(OUT, n + ".json")
    return json.load(open(p)) if os.path.exists(p) else None
def f(x, d):
    try:
        return "n.c." if x != x else f"{x:.{d}f}"
    except TypeError:
        return "n.c."

# ----------------------------------------------------------------- S2
s2 = load("s2_estimator")
if s2:
    rows = [("Lemniscate, bounded noise", "clean"), ("Lemniscate, encoder outliers", "outliers"),
            ("Rhombus, bounded noise", "limit")]
    L = ["\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
         "& \\multicolumn{2}{c}{Estimator \\eqref{eq:ocp-mhe}} & \\multicolumn{2}{c}{Extended Kalman filter} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
         "Regime & Tractor & Trailers & Tractor & Trailers \\\\",
         "\\midrule"]
    nan = float("nan")
    for lab, k in rows:
        m = s2[f"{k}/nmhe"].get("rmse_per_seg", [nan, nan, nan])
        e = s2[f"{k}/ekf"].get("rmse_per_seg", [nan, nan, nan])
        L.append(f"{lab} & {f(m[0],4)} & {f(max(m[1:]),4)} & {f(e[0],4)} & {f(max(e[1:]),4)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(PAPER, "tab_s2.tex"), "w").write("\n".join(L) + "\n")

# ----------------------------------------------------------------- S4
s4 = load("s4_scaling")
if s4:
    L = ["\\begin{tabular}{@{}cccccc@{}}", "\\toprule",
         "$N$ & Variables & Generator & Tracking & Estimator & Total \\\\",
         " & (tracking) & (ms) & (ms) & (ms) & (ms) \\\\", "\\midrule"]
    for r in s4["N_sweep"]:                       # every N measured, none left out
        if True:
            L.append(f"{r['N']} & {r['n_var_mpc']} & {f(r['t_ref'],1)} & {f(r['t_mpc'],1)} & "
                     f"{f(r['t_mhe'],1)} & {f(r['t_tot'],1)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(PAPER, "tab_s4.tex"), "w").write("\n".join(L) + "\n")

# ----------------------------------------------------------------- S5
s5 = load("s5_robustness")
if s5 and "value" in s5["noise"][0]:
    L = ["\\begin{tabular}{@{}llcc@{}}", "\\toprule",
         "Perturbation & Value & Worst segment (m) & Estimation error (m) \\\\", "\\midrule"]
    def blk(key, label, fmtv):
        out = []
        for i, r in enumerate(s5[key]):
            name = label if i == 0 else ""
            out.append(f"{name} & {fmtv(r['value'])} & {f(r['off'],3)} & {f(r['est'],4)} \\\\")
        return out
    L += blk("noise", "Measurement noise", lambda v: f"$\\times${v:g}")
    L += ["\\addlinespace"]
    L += blk("hitch", "Geometry error", lambda v: f"{100*v:g}\\,\\%")
    L += ["\\addlinespace"]
    L += blk("slip", "Actuator gain", lambda v: f"{v:g}")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(PAPER, "tab_s5.tex"), "w").write("\n".join(L) + "\n")
print("tables written")
