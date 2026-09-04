"""Within the BALANCED class, does low aperiodic autocorrelation sidelobe predict a
low real-trace false-hit rate?  Theory: for a balanced header g in {+-1}^L,
   Var(2M-L) = sum_{i,j} g_i g_j gamma_{|i-j|} = gamma_0*L + 2*sum_{k>=1} gamma_k c_k(g),
   c_k(g) = sum_i g_i g_{i+k}   (aperiodic autocorrelation sidelobes).
Balance fixes E[M]=L/2 for any background marginal; low sidelobes cap the variance
inflation caused by background DEPENDENCE."""
import os, json, itertools
import numpy as np
from math import comb
from numpy.lib.stride_tricks import sliding_window_view

TRACE = os.environ.get("NFW_TRACE_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..",
    "NFW2_Simulation", "results", "real_traffic"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
CASES = [("iscx_tor_file_transfer_ipd", 1.0, 10, 10, 11),     # the failing cell
         ("iscx_tor_file_transfer_ipd", 1.0, 12, 11, 11),
         ("iscx_tor_chat_ipd", 1.0, 10, 10, 11),
         ("iscx_tor_ipd", 1.0, 10, 9, 9)]
MAX_BITS = 200_000


def balanced(L):
    w = L // 2
    return np.array([np.bincount(list(c), minlength=L).astype(np.int8)
                     for c in itertools.combinations(range(L), w)])


def max_sidelobe(g):
    L = g.size
    return max(abs(int(np.dot(g[:L - k], g[k:]))) for k in range(1, L))


out = []
for (pool, d, L_h, tau, W) in CASES:
    x = np.load(os.path.join(TRACE, pool + ".npy"))
    x = x[np.isfinite(x) & (x >= 0)][:MAX_BITS]
    b = (np.rint(2.0 * x / d).astype(np.int64) & 1).astype(np.int8)
    s = (2 * b.astype(np.float32) - 1.0)
    win = sliding_window_view(s, L_h)
    H = balanced(L_h)
    G = 2 * H.astype(np.int16) - 1
    sl = np.array([max_sidelobe(g) for g in G])
    p0 = np.empty(H.shape[0])
    for i in range(0, H.shape[0], 128):
        Hb = H[i:i + 128]
        ag = (L_h + win @ (2 * Hb.T.astype(np.float32) - 1.0)) * 0.5
        p0[i:i + Hb.shape[0]] = (ag >= tau - 0.25).mean(axis=0)
    p0t = float(sum(comb(L_h, i) * 0.5 ** L_h for i in range(tau, L_h + 1)))
    print("%s  Delta=%g  L_h=%d tau=%d   bound p0=%.6f   (%d balanced headers)"
          % (pool, d, L_h, tau, p0t, H.shape[0]))
    print("   %-14s %-8s %-10s %-10s %-8s" % ("max|c_k|", "n_hdrs", "p0_mean", "p0_worst", "ratio"))
    grp = []
    for v in sorted(set(sl.tolist())):
        m = sl == v
        grp.append({"sidelobe": int(v), "n": int(m.sum()),
                    "p0_mean": float(p0[m].mean()), "p0_worst": float(p0[m].max()),
                    "ratio_worst": float(p0[m].max() / p0t)})
        print("   %-14d %-8d %-10.6f %-10.6f %-8.2f"
              % (v, m.sum(), p0[m].mean(), p0[m].max(), p0[m].max() / p0t))
    lo = sl <= np.percentile(sl, 10)
    print("   -> lowest-sidelobe decile (%d hdrs): worst p0=%.6f (%.2fx bound)\n"
          % (lo.sum(), p0[lo].max(), p0[lo].max() / p0t))
    out.append({"pool": pool, "delta": d, "L_h": L_h, "tau": tau, "p0_bound": p0t,
                "by_sidelobe": grp,
                "low_decile_worst_ratio": float(p0[lo].max() / p0t),
                "all_worst_ratio": float(p0.max() / p0t)})
json.dump(out, open(os.path.join(OUT, "probeA6_sidelobe.json"), "w"), indent=1)
