"""Probe F: the submitted product-form P_D is PROVABLY optimistic; the scan statistic
is the correct aggregation.

The J candidate confirmation tracks of a marked flow all read the SAME periodic hit
sequence H_1..H_Nmax ~ Bern(q).  Each confirmation event is an increasing event in that
Bernoulli vector, so by Harris/FKG the complements satisfy
      P(no track confirms) >= prod_f P(track f does not confirm),
i.e.  P_D <= 1 - (1 - p_track)^J .
The product form is therefore a valid SEARCH RELAXATION but not a guarantee.

Outputs, per row of the SUBMITTED design table:
  PD_prod      the submitted formula
  PD_scan      scan statistic at the analytic q
  PD_scan_mcq  scan statistic at the q measured by the submitted channel Monte-Carlo
  MC H1 conf.  the submitted end-to-end blind confirmation ratio (ground truth)
"""
import os, json
import numpy as np
from math import comb, log1p, expm1

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
NS = 200000


def sf(k, n, p):                      # P(Bin(n,p) > k)
    return float(sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1, n + 1)))


def scan(q, q_cap, N, k, J, N_max, rng, ns=NS):
    H = rng.random((ns, N_max)) < q
    cap = rng.random((ns, N_max)) < q_cap
    cs = np.concatenate([np.zeros((ns, 1), dtype=np.int32),
                         np.cumsum(H, axis=1, dtype=np.int32)], axis=1)
    conf = np.zeros(ns, dtype=bool)
    for f in range(min(J, N_max - N + 1)):
        conf |= (cap[:, f] & ((cs[:, f + N] - cs[:, f + 1]) >= k - 1))
    return float(conf.mean())


def main():
    d = json.load(open(os.path.join(RES, "legacy", "letter_design.json")))
    mc = json.load(open(os.path.join(RES, "legacy", "letter_mc.json")))
    mcmap = {round(r["lam"], 4): r for r in mc["rows"]}
    rng = np.random.default_rng(5)
    out = []
    print("%-6s %-4s %-4s %-6s | %-9s %-9s %-12s | %-9s"
          % ("lam", "N", "k", "q", "PD_prod", "PD_scan", "PD_scan_mcq", "MC H1conf"))
    for r in d["rows"]:
        L_h, T_f, N, k = r["L_h"], r["T_f"], r["N_scr"], r["k_scr"]
        q, qc = r["q"], r["q_cap"]
        span = (N - 1) * T_f + L_h
        L = 2000                          # the window the submitted code actually used
        J = max(0, (L - span + 1) // T_f)
        N_max = L // T_f
        pd_prod = -expm1(J * log1p(-min(qc * sf(k - 2, N - 1, q), 1 - 1e-16)))
        s1 = scan(q, qc, N, k, J, N_max, rng)
        m = mcmap.get(round(r["lam"], 4))
        qmc = m["q_mc"] if m else None
        s2 = scan(qmc, qmc, N, k, J, N_max, rng) if qmc else float("nan")
        gt = m["h1_screen"]["p_confirm_mc"] if m else None
        out.append({"lam": r["lam"], "N": N, "k": k, "J": J, "N_max": N_max,
                    "q_analytic": q, "q_mc": qmc, "PD_product": pd_prod,
                    "PD_scan": s1, "PD_scan_at_q_mc": s2, "MC_h1_confirm": gt})
        print("%-6.3f %-4d %-4d %-6.3f | %-9.5f %-9.5f %-12.5f | %s"
              % (r["lam"], N, k, q, pd_prod, s1, s2,
                 ("%.4f" % gt) if gt is not None else "-"))
    json.dump(out, open(os.path.join(RES, "probeF_pd_aggregation.json"), "w"), indent=1)
    print("\nWROTE probeF_pd_aggregation.json")


if __name__ == "__main__":
    main()
