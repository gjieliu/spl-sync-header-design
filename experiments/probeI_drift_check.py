"""Probe I: validate the per-frame drift model that sizes R, using the CHANNEL ONLY.

The letter sizes R from  Var(drift per frame) = lambda * T_f  (symmetric baseline).
Measuring drift through the detector is misleading: conditioning on "the previous slot
was a hit" mixes in frame-index slips and accumulated extrapolation (variance inflated
5-12x), while conditioning on a clean anchor selects small drifts (variance deflated).
The model must therefore be checked on the pure channel map: the difference of the true
receive positions of consecutive headers, minus T_f.
"""
import os, sys, json, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from probeB_burst_channel import channel_ge

RES = os.path.join(HERE, "..", "results")


def main():
    rng = np.random.default_rng(1)
    out = []
    for lam, T_f in [(0.005, 20), (0.01, 20), (0.02, 21), (0.03, 22), (0.05, 21)]:
        N, F = 200, 250
        d = []
        for _ in range(F):
            sent = rng.integers(0, 2, size=N * T_f).astype(np.int8)
            _, s2r = channel_ge(sent, lam, 1, 0.5, 0.0, rng, want_map=True)
            pos = s2r[np.arange(N) * T_f].astype(float)
            pos[pos < 0] = np.nan
            dd = np.diff(pos) - T_f
            d.append(dd[~np.isnan(dd)])
        d = np.concatenate(d)
        rec = {"lam": lam, "T_f": T_f, "n": int(d.size), "mean": float(d.mean()),
               "var_measured": float(d.var()), "var_theory_lamTf": lam * T_f,
               "ratio": float(d.var() / (lam * T_f)),
               "P_abs_gt": {str(R): float((np.abs(d) > R).mean()) for R in (2, 3, 4, 5, 6)}}
        out.append(rec)
        print("lam=%.3f T_f=%2d  n=%6d  mean=%+.4f  Var=%.4f  (theory lam*T_f=%.4f, "
              "ratio %.3f)" % (lam, T_f, d.size, d.mean(), d.var(), lam * T_f, rec["ratio"]))
        print("        P(|drift|>R):", "  ".join("R=%s:%.5f" % (k, v)
                                                 for k, v in rec["P_abs_gt"].items()))
    json.dump(out, open(os.path.join(RES, "probeI_drift_check.json"), "w"), indent=1)
    print("\nWROTE probeI_drift_check.json")


if __name__ == "__main__":
    main()
