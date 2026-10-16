"""Correct predictor: the SIGNED alignment V(g) = gamma_0*L + 2*sum_{k>=1} gamma_k c_k(g)
between the header's aperiodic autocorrelation and the background autocovariance."""
import os, json, itertools
import numpy as np
from math import comb
from numpy.lib.stride_tricks import sliding_window_view

TRACE = os.environ.get("NFW_TRACE_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..",
    "NFW2_Simulation", "results", "real_traffic"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
CASES = [("iscx_tor_file_transfer_ipd", 1.0, 10, 10),
         ("iscx_tor_file_transfer_ipd", 1.0, 12, 11),
         ("iscx_tor_chat_ipd", 1.0, 10, 10)]
MAX_BITS = 200_000


def balanced(L):
    w = L // 2
    return np.array([np.bincount(list(c), minlength=L).astype(np.int8)
                     for c in itertools.combinations(range(L), w)])


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float(np.dot(ra, rb) / np.sqrt(np.dot(ra, ra) * np.dot(rb, rb)))


res = []
for (pool, d, L_h, tau) in CASES:
    x = np.load(os.path.join(TRACE, pool + ".npy"))
    x = x[np.isfinite(x) & (x >= 0)][:MAX_BITS]
    b = (np.rint(2.0 * x / d).astype(np.int64) & 1).astype(np.int8)
    sf = 2.0 * b - 1.0
    mu = sf.mean(); c = sf - mu
    gam = np.array([float(np.dot(c[:len(c) - k], c[k:]) / len(c)) if k else float(np.dot(c, c) / len(c))
                    for k in range(L_h)])
    s32 = (2 * b.astype(np.float32) - 1.0)
    win = sliding_window_view(s32, L_h)
    H = balanced(L_h); G = 2 * H.astype(np.int16) - 1
    V = np.empty(H.shape[0]); p0 = np.empty(H.shape[0])
    for i, g in enumerate(G):
        ck = np.array([int(np.dot(g[:L_h - k], g[k:])) for k in range(L_h)], dtype=float)
        V[i] = gam[0] * L_h + 2.0 * float(np.dot(gam[1:], ck[1:]))
    for i in range(0, H.shape[0], 128):
        Hb = H[i:i + 128]
        ag = (L_h + win @ (2 * Hb.T.astype(np.float32) - 1.0)) * 0.5
        p0[i:i + Hb.shape[0]] = (ag >= tau - 0.25).mean(axis=0)
    p0t = float(sum(comb(L_h, j) * 0.5 ** L_h for j in range(tau, L_h + 1)))
    sl = np.array([max(abs(int(np.dot(g[:L_h - k], g[k:]))) for k in range(1, L_h)) for g in G])
    rho_V = spearman(V, p0); rho_sl = spearman(sl.astype(float), p0)
    lo = V <= np.percentile(V, 10)
    print("%s D=%g L_h=%d tau=%d | Spearman(p0, V)=%+.3f   Spearman(p0, max|c_k|)=%+.3f"
          % (pool, d, L_h, tau, rho_V, rho_sl))
    print("    worst ratio: all balanced = %.2fx | lowest-V decile (%d hdrs) = %.2fx | "
          "lowest-max|c_k| decile = %.2fx"
          % (p0.max() / p0t, int(lo.sum()), p0[lo].max() / p0t,
             p0[sl <= np.percentile(sl, 10)].max() / p0t))
    res.append({"pool": pool, "delta": d, "L_h": L_h, "tau": tau, "p0_bound": p0t,
                "spearman_V": rho_V, "spearman_sidelobe": rho_sl,
                "worst_all": float(p0.max() / p0t),
                "worst_lowV_decile": float(p0[lo].max() / p0t),
                "gamma": gam.tolist()})
json.dump(res, open(os.path.join(OUT, "probeA7_alignment.json"), "w"), indent=1)
