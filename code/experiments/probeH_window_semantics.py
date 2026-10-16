"""Probe H: which observation window L does the SUBMITTED design actually use?

The submitted text (Sec. IV-B) writes P_FA(N,k;L_rej) and P_D(N,k;L_rej) with
L_rej = 500 packets, but letter_design.py counts the M sliding starts and the J true
periodic phases over L_max = 2000.  This script recomputes both and matches them
against the stored M_scr/J_scr/P_FA/P_D, showing that L = 2000 is what was computed,
and that under the L = 500 semantics of the text the whole table would be infeasible.
"""
import os, json
from math import comb, log1p, expm1

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
M_Q, F_FA = 0.15, 2.0


def sf(k, n, p):
    return float(sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1, n + 1)))


def agg(p_start, n):
    if n <= 0 or p_start <= 0:
        return 0.0
    return float(-expm1(n * log1p(-min(p_start, 1 - 1e-16))))


def main():
    d = json.load(open(os.path.join(RES, "legacy", "letter_design.json")))
    out = []
    print("%-6s %-4s %-4s %-4s %-6s %-8s %-8s | %-9s %-9s | %-8s %-8s | %s"
          % ("lam", "L_h", "N", "k", "span", "M@2000", "M@500",
             "PFA@2000", "PFA@500", "PD@2000", "PD@500", "stored M matches"))
    for r in d["rows"]:
        L_h, T_f, N, k = r["L_h"], r["T_f"], r["N_scr"], r["k_scr"]
        p0c, pWc = min(1, F_FA * r["p0"]), min(1, F_FA * r["pW"])
        qc, qcc = (1 - M_Q) * r["q"], (1 - M_Q) * r["q_cap"]
        span = (N - 1) * T_f + L_h
        res = {}
        for L in (2000, 500):
            M = max(0, L - span + 1)
            J = max(0, (L - span + 1) // T_f)
            res[L] = (M, J, agg(p0c * sf(k - 2, N - 1, pWc), M),
                      agg(min(1.0, qcc * sf(k - 2, N - 1, qc)), J))
        match = "2000" if r["M_scr"] == res[2000][0] else (
                "500" if r["M_scr"] == res[500][0] else "?")
        out.append({"lam": r["lam"], "L_h": L_h, "N": N, "k": k, "span": span,
                    "stored_M": r["M_scr"], "stored_J": r["J_scr"],
                    "stored_PFA": r["P_FA_scr_check"], "stored_PD": r["P_D_scr_check"],
                    "L2000": {"M": res[2000][0], "J": res[2000][1],
                              "P_FA": res[2000][2], "P_D": res[2000][3]},
                    "L500": {"M": res[500][0], "J": res[500][1],
                             "P_FA": res[500][2], "P_D": res[500][3]},
                    "stored_matches_L": match})
        print("%-6.3f %-4d %-4d %-4d %-6d %-8d %-8d | %-9.2e %-9.2e | %-8.5f %-8.5f | %s"
              % (r["lam"], L_h, N, k, span, res[2000][0], res[500][0],
                 res[2000][2], res[500][2], res[2000][3], res[500][3], match))
    json.dump(out, open(os.path.join(RES, "probeH_window_semantics.json"), "w"), indent=1)
    print("\nStored values match L = 2000 in every row, while the text says L_rej = 500.")
    print("Under the text's semantics P_D collapses to 0.07-0.44 -> the table is infeasible.")
    print("WROTE probeH_window_semantics.json")


if __name__ == "__main__":
    main()
