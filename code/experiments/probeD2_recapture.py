"""Probe D2 (answers R1-C3): paired comparison of re-capture rules.
All modes are run on the SAME channel realisations (common random numbers), and the
per-frame +-R scan is a slice of a precomputed correlation profile, so 800+ flows fit.

modes: none              -- the submitted detector
       widen             -- after m consecutive misses, scan +-(mult*R)
       widen+fullmatch   -- same, but require a FULL match (tau=L_h) to re-anchor,
                            so a widened window cannot buy hits at wrong offsets
"""
import os, sys, json
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from probeB_burst_channel import channel_ge

OUT = os.path.join(HERE, "..", "results")
EPS = 0.05
N_FLOWS = 800
M_STALE = 4
MULT = 3


def profile(recv, s):
    """agreement count of the header at every start position of recv."""
    if recv.size < s.size:
        return np.zeros(0, dtype=np.int16)
    w = sliding_window_view(recv, s.size)
    return (w == s[None, :]).sum(axis=1).astype(np.int16)


def track(prof, tp, T_f, L_h, tau, R, N_max, mode):
    anchor = int(tp[0]) if tp[0] >= 0 else 0
    yh = ye = 0
    rm = 0
    P = prof.size
    for f in range(1, N_max):
        pred = anchor + T_f
        rad, thr = R, tau
        if mode != "none" and rm >= M_STALE:
            rad = MULT * R
            if mode == "strict":
                thr = L_h
        lo, hi = max(0, pred - rad), min(P, pred + rad + 1)
        if lo >= hi:
            rm += 1; anchor = pred; continue
        seg = prof[lo:hi]
        j = int(np.argmax(seg))
        best, bp = int(seg[j]), lo + j
        if best >= thr:
            yh += 1
            if tp[f] >= 0 and bp == tp[f]:
                ye += 1
            rm = 0; anchor = bp
        else:
            rm += 1; anchor = pred
    return yh, ye


def main():
    rows = json.load(open(os.path.join(OUT, "design_v2_table.json")))["rows"]
    rng = np.random.default_rng(777)
    res = []
    modes = ["none", "widen", "strict"]
    print("paired, n_flows=%d, m_stale=%d, widen=x%d" % (N_FLOWS, M_STALE, MULT))
    print("%-6s %-9s | %s" % ("lam%", "cfg",
          "  ".join("%-15s" % m for m in modes)))
    for key in sorted(rows, key=lambda k: float(k)):
        r = rows[key]
        if not r.get("feasible"):
            continue
        L_h, tau, R, L_c, N_max, lam = (r["L_h"], r["tau"], r["R"], r["L_c"],
                                        r["N_max"], r["lam"])
        T_f = L_h + L_c
        s = rng.integers(0, 2, size=L_h).astype(np.int8)
        acc = {m: [[], []] for m in modes}
        for _ in range(N_FLOWS):
            sent = np.empty(N_max * T_f, dtype=np.int8)
            for f in range(N_max):
                sent[f * T_f:f * T_f + L_h] = s
                sent[f * T_f + L_h:(f + 1) * T_f] = rng.integers(0, 2, size=L_c)
            recv, s2r = channel_ge(sent, lam, 1, 0.5, EPS, rng, want_map=True)
            prof = profile(recv, s)
            tp = s2r[np.arange(N_max) * T_f]
            for m in modes:                       # SAME realisation for every mode
                yh, ye = track(prof, tp, T_f, L_h, tau, R, N_max, m)
                acc[m][0].append(yh); acc[m][1].append(ye)
        line = []
        base = float(np.mean(acc["none"][1]))
        rec = {"lam": lam, "L_h": L_h, "tau": tau, "R": R, "N_max": N_max,
               "muY_design": r["muY"], "n_flows": N_FLOWS}
        for m in modes:
            yh = np.array(acc[m][0], dtype=float); ye = np.array(acc[m][1], dtype=float)
            d = ye - np.array(acc["none"][1], dtype=float)
            rec[m] = {"Y_hit": float(yh.mean()), "Y_exact": float(ye.mean()),
                      "Y_exact_se": float(ye.std(ddof=1) / np.sqrt(ye.size)),
                      "paired_gain": float(d.mean()),
                      "paired_gain_se": float(d.std(ddof=1) / np.sqrt(d.size)),
                      "exact_over_hit": float(ye.mean() / max(yh.mean(), 1e-9))}
            line.append("Yh%5.1f Ye%5.1f%s" % (yh.mean(), ye.mean(),
                        "" if m == "none" else " (%+.2f+-%.2f)" %
                        (rec[m]["paired_gain"], 1.96 * rec[m]["paired_gain_se"])))
        print("%-6.1f %-9s | %s" % (lam * 100, "%d/%d/%d" % (L_h, tau, R),
                                    "  ".join(line)), flush=True)
        res.append(rec)
    json.dump(res, open(os.path.join(OUT, "probeD2_recapture.json"), "w"), indent=1)
    print("\nWROTE probeD2_recapture.json")


if __name__ == "__main__":
    main()
