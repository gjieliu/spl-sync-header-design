"""Validate the balanced-header false-hit bound on real ISCX IPDs for the
CONFIGURATIONS OF THE v2 DESIGN TABLE (short headers included)."""
import os, json, itertools
import numpy as np
from math import comb
from numpy.lib.stride_tricks import sliding_window_view

TRACE = os.environ.get("NFW_TRACE_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..",
    "NFW2_Simulation", "results", "real_traffic"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
POOLS = ["iscx_tor_ipd", "iscx_vpn_ipd", "iscx_vpn_audio_ipd", "iscx_tor_video_ipd",
         "iscx_tor_chat_ipd", "iscx_tor_file_transfer_ipd"]
DELTAS = [1.0, 5.0, 19.0, 50.0]
# (L_h, tau, W) taken from the current v2 design table
_T = json.load(open(os.path.join(OUT, "design_v2_table.json")))["rows"]
CFGS = sorted({(r["L_h"], r["tau"], r["W"]) for r in _T.values() if r.get("feasible")})
MAX_BITS = 200_000
CHUNK = 128


def binom_sf_ge(n, p, k):
    return float(sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1)))


def balanced(L_h):
    w = L_h // 2
    return np.array([np.bincount(list(c), minlength=L_h).astype(np.int8)
                     for c in itertools.combinations(range(L_h), w)])


def worst(bits, H, tau, W):
    s = (2 * bits.astype(np.float32) - 1.0)
    win = sliding_window_view(s, H.shape[1])
    p0m = pWm = 0.0
    for i in range(0, H.shape[0], CHUNK):
        Hb = H[i:i + CHUNK]
        ag = (H.shape[1] + win @ (2 * Hb.T.astype(np.float32) - 1.0)) * 0.5
        hit = ag >= tau - 0.25
        p0m = max(p0m, float(hit.mean(axis=0).max()))
        cs = np.concatenate([np.zeros((1, Hb.shape[0]), dtype=np.int64),
                             np.cumsum(hit.astype(np.int64), axis=0)], axis=0)
        pWm = max(pWm, float((((cs[W:] - cs[:-W]) > 0).mean(axis=0)).max()))
    return p0m, pWm


res = {"note": "balanced-header bound checked on the v2 design configurations",
       "cfgs": CFGS, "rows": []}
worst_ratio = 0.0
for name in POOLS:
    p = os.path.join(TRACE, name + ".npy")
    if not os.path.exists(p):
        continue
    x = np.load(p); x = x[np.isfinite(x) & (x >= 0)][:MAX_BITS]
    for d in DELTAS:
        b = (np.rint(2.0 * x / d).astype(np.int64) & 1).astype(np.int8)
        for (L_h, tau, W) in CFGS:
            H = balanced(L_h)
            p0m, pWm = worst(b, H, tau, W)
            p0t = binom_sf_ge(L_h, 0.5, tau)
            pWt = min(1.0, W * p0t)                      # the CERTIFIED union bound
            r = {"pool": name, "delta": d, "L_h": L_h, "tau": tau, "W": W,
                 "n_headers": int(H.shape[0]), "bias": float(b.mean()),
                 "p0_bound": p0t, "p0_worst": p0m, "p0_ratio": p0m / p0t,
                 "pW_bound_union": pWt, "pW_worst": pWm, "pW_ratio": pWm / pWt}
            res["rows"].append(r)
            worst_ratio = max(worst_ratio, r["p0_ratio"])
res["worst_p0_ratio_over_all_cells"] = worst_ratio
json.dump(res, open(os.path.join(OUT, "probeA4_v2configs.json"), "w"), indent=1)
# summary
print("cells=%d   WORST p0_worst/p0_bound over ALL (pool,Delta,config) = %.3f"
      % (len(res["rows"]), worst_ratio))
print("\nworst 12 cells by p0 ratio:")
for r in sorted(res["rows"], key=lambda z: -z["p0_ratio"])[:12]:
    print("  %-27s d=%-5g %2d/%2d  bias=%.3f  p0 bnd=%.6f worst=%.6f (%.2fx) | "
          "pW unionbnd=%.4f worst=%.4f (%.2fx)"
          % (r["pool"], r["delta"], r["L_h"], r["tau"], r["bias"], r["p0_bound"],
             r["p0_worst"], r["p0_ratio"], r["pW_bound_union"], r["pW_worst"],
             r["pW_ratio"]))
