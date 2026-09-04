"""Emit the manuscript tables from the result JSONs, so no number is transcribed by hand."""
import os, json
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "..", "manuscript")
RHO = 0.85          # conservative anchor-exactness factor (measured min 0.855)

T = json.load(open(os.path.join(RES, "design_v2_table.json")))["rows"]
D = {round(d["lam"], 5): d for d in json.load(open(os.path.join(RES, "probeD_endtoend.json")))}
B = {round(b["lam"], 5): b for b in json.load(open(os.path.join(RES, "probeB2_burst_v2.json")))}
keys = [k for k in sorted(T, key=float) if T[k].get("feasible")]

# ---- one wide table: design (left) + validation (right) ----
t = [r"\begin{table*}[t]", r"\centering\footnotesize",
     r"\setlength{\tabcolsep}{4.2pt}",
     r"\caption{Design ($\pbit{=}5\%$, $L_c{=}10$, $\alpha{=}10^{-9}$, "
     r"$\Lobs{=}2000$ packets; every row meets \eqref{eq:fa}, largest "
     r"$P_{\mathrm{FA}}=7.7{\times}10^{-10}$, and \eqref{eq:disjoint}) and "
     r"Monte-Carlo validation under actual insertions and deletions. "
     r"$Y_{\mathrm{hit}}$ counts frames called aligned, $Y_{\mathrm{exact}}$ only "
     r"those on the true position \emph{and} frame index; the last two columns are "
     r"under bursty edits of mean length $20$ at the same mean $\lambda$.}",
     r"\label{tab:main}",
     r"\begin{tabular}{@{}c cccccccc c ccccc@{}}", r"\toprule",
     r"& \multicolumn{8}{c}{design} & & \multicolumn{5}{c}{validation} \\",
     r"\cmidrule(r){2-9}\cmidrule(l){11-15}",
     r"$\lambda$(\%) & $\Lh$ & $\tau$ & $R$ & $\Ns$ & $\ks$ & $q/p_W$ & $\Nmax$ & "
     r"$\muY^{\mathrm{dec}}$ & & $Y_{\mathrm{hit}}$ & $Y_{\mathrm{exact}}$ & "
     r"$Y_{\mathrm{exact}}/Y_{\mathrm{hit}}$ & blackout & "
     r"$Y_{\mathrm{exact}}$ burst \\", r"\midrule"]
for k in keys:
    r = T[k]; lam = round(r["lam"], 5)
    d = D[lam]["no_recapture"]; b = B[lam]["by_burst"]
    t.append("%g & %d & %d & %d & %d & %d & %.0f & %d & %.0f & & %.1f & %.1f & %.3f & "
             "%.2f & %.1f \\\\"
             % (r["lam"] * 100, r["L_h"], r["tau"], r["R"], r["N_scr"], r["k_scr"],
                r["sep"], r["N_max"], RHO * r["muY"], d["Y_hit_mean"],
                d["Y_exact_mean"], d["Y_exact_over_Y_hit"],
                b["20"]["blackout_frac"], b["20"]["Y_exact"]))
t += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
open(os.path.join(OUT, "tab_main.tex"), "w").write("\n".join(t) + "\n")
print("\n".join(t))
print("\nsanity: min Y_exact / muY_dec = %.2f (must exceed 1)"
      % min(D[round(T[k]['lam'],5)]["no_recapture"]["Y_exact_mean"] / (RHO * T[k]["muY"])
            for k in keys))
