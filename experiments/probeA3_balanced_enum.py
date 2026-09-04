"""Probe A3: enumerate ALL balanced headers (not a sample) and report the true
worst case over that class, on the real ISCX pools at the paper's nominal Delta."""
import os, json, itertools
import numpy as np
from math import comb
from numpy.lib.stride_tricks import sliding_window_view

TRACE = os.environ.get("NFW_TRACE_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..",
    "NFW2_Simulation", "results", "real_traffic"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
POOLS = ["iscx_tor_ipd", "iscx_vpn_ipd", "iscx_vpn_audio_ipd", "iscx_tor_video_ipd"]
DELTAS = [1.0, 19.0, 50.0]
CFGS = [(10, 9, 7), (11, 11, 13), (11, 10, 9), (12, 11, 11)]
MAX_BITS = 150_000
CHUNK = 96


def binom_tail_ge(n, p, k):
    return float(sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1)))


def all_balanced(L_h):
    w = L_h // 2                      # odd L_h: both w and L_h-w are "as balanced as possible"
    ws = {w, L_h - w}
    out = []
    for wt in sorted(ws):
        for idx in itertools.combinations(range(L_h), wt):
            h = np.zeros(L_h, dtype=np.int8); h[list(idx)] = 1
            out.append(h)
    return np.array(out)


def worst_over(bits, H, tau, W, chunk=CHUNK):
    s = (2 * bits.astype(np.float32) - 1.0)
    win = sliding_window_view(s, H.shape[1])
    p0max = pWmax = 0.0
    p0sum = 0.0
    for i in range(0, H.shape[0], chunk):
        Hb = H[i:i + chunk]
        agree = (H.shape[1] + win @ (2 * Hb.T.astype(np.float32) - 1.0)) * 0.5
        hit = agree >= tau - 0.25
        p0 = hit.mean(axis=0)
        cs = np.concatenate([np.zeros((1, Hb.shape[0]), dtype=np.int64),
                             np.cumsum(hit.astype(np.int64), axis=0)], axis=0)
        pW = ((cs[W:] - cs[:-W]) > 0).mean(axis=0)
        p0max = max(p0max, float(p0.max())); pWmax = max(pWmax, float(pW.max()))
        p0sum += float(p0.sum())
    return p0max, pWmax, p0sum / H.shape[0]


res = {"note": "worst case over the FULL balanced-header class, real ISCX IPDs", "rows": []}
for name in POOLS:
    p = os.path.join(TRACE, name + ".npy")
    if not os.path.exists(p):
        continue
    x = np.load(p); x = x[np.isfinite(x) & (x >= 0)][:MAX_BITS]
    for d in DELTAS:
        b = (np.rint(2.0 * x / d).astype(np.int64) & 1).astype(np.int8)
        pi = float(b.mean())
        for (L_h, tau, W) in CFGS:
            H = all_balanced(L_h)
            p0m, pWm, p0mean = worst_over(b, H, tau, W)
            p0t = binom_tail_ge(L_h, 0.5, tau); pWt = 1 - (1 - p0t)**W
            res["rows"].append({"pool": name, "delta": d, "bias": pi, "L_h": L_h,
                                "tau": tau, "W": W, "n_headers": int(H.shape[0]),
                                "p0_theory": p0t, "p0_worst_balanced": p0m,
                                "p0_mean_balanced": p0mean,
                                "pW_theory": pWt, "pW_worst_balanced": pWm,
                                "p0_ratio_worst": p0m / p0t, "pW_ratio_worst": pWm / pWt})
            print("%-19s d=%-4g pi=%.3f  %2d/%2d (%d hdrs): p0 th=%.6f worstBAL=%.6f (%.2fx) | "
                  "pW th=%.4f worstBAL=%.4f (%.2fx)"
                  % (name, d, pi, L_h, tau, H.shape[0], p0t, p0m, p0m / p0t,
                     pWt, pWm, pWm / pWt), flush=True)
with open(os.path.join(OUT, "probeA3_balanced_enum.json"), "w") as f:
    json.dump(res, f, indent=1)
print("WROTE probeA3_balanced_enum.json")
