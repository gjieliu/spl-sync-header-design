"""
Probe A2: is the real-trace false-hit blow-up a property of the HEADER's WEIGHT?

Theory.  Background bit b with marginal P(b=1)=pi; header h with w ones.
  E[#matches] = (L_h - w) + w(2*pi - 1)   ->  independent of pi  iff  w = L_h/2.
So a BALANCED header makes the match statistic insensitive to background bias,
and (under independence) Hoeffding gives  p0 <= exp(-2 (tau - L_h/2)^2 / L_h)
for ANY pi -- i.e. no "iid fair bits" assumption is needed, only balance.

This script measures empirical p0 / pW per header, stratified by header weight w,
on the real ISCX Tor/VPN IPD pools.
"""
import os, json
import numpy as np
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
TRACE = os.environ.get("NFW_TRACE_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..",
    "NFW2_Simulation", "results", "real_traffic"))
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)

POOLS = ["iscx_tor_ipd", "iscx_vpn_ipd", "iscx_vpn_audio_ipd", "iscx_tor_video_ipd"]
DELTAS = [1.0, 5.0, 19.0]
CFGS = [(10, 9, 7), (11, 11, 13), (11, 10, 9)]     # (L_h, tau, W) from Table I rows
PER_W = 8
MAX_BITS = 200_000


def binom_tail_ge(n, p, k):
    return float(sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1)))


def headers_by_weight(L_h, rng, per_w=PER_W):
    """Stratified header sample: several headers at each Hamming weight w."""
    hs, ws = [], []
    for w in range(L_h + 1):
        m = min(per_w, comb(L_h, w))
        seen = set()
        tries = 0
        while len(seen) < m and tries < 200:
            tries += 1
            idx = tuple(sorted(rng.choice(L_h, size=w, replace=False).tolist())) if w else ()
            if idx in seen:
                continue
            seen.add(idx)
            h = np.zeros(L_h, dtype=np.int8)
            if w:
                h[list(idx)] = 1
            hs.append(h); ws.append(w)
    return np.array(hs), np.array(ws)


def rates(bits, H, tau, W):
    """Per-header p0 and pW. H is (n_headers, L_h) in {0,1}."""
    from numpy.lib.stride_tricks import sliding_window_view
    s = (2 * bits.astype(np.float32) - 1.0)
    win = sliding_window_view(s, H.shape[1])
    Hs = (2 * H.T.astype(np.float32) - 1.0)
    agree = (H.shape[1] + win @ Hs) * 0.5
    hit = agree >= tau - 0.25
    p0 = hit.mean(axis=0)
    cs = np.concatenate([np.zeros((1, H.shape[0]), dtype=np.int64),
                         np.cumsum(hit.astype(np.int64), axis=0)], axis=0)
    m = hit.shape[0] - W + 1
    pW = (((cs[W:] - cs[:-W]) > 0).mean(axis=0)) if m > 0 else np.full(H.shape[0], np.nan)
    return p0, pW


def main():
    rng = np.random.default_rng(424242)
    out = {"note": "per-header p0/pW stratified by header Hamming weight, real ISCX IPDs",
           "cfgs": CFGS, "deltas_ms": DELTAS, "per_w": PER_W, "pools": {}}
    for name in POOLS:
        p = os.path.join(TRACE, name + ".npy")
        if not os.path.exists(p):
            continue
        x = np.load(p)
        x = x[np.isfinite(x) & (x >= 0)][:MAX_BITS]
        rec = {}
        for d in DELTAS:
            b = (np.rint(2.0 * x / d).astype(np.int64) & 1).astype(np.int8)
            pi = float(b.mean())
            cell = {"bias": pi, "cfg": {}}
            for (L_h, tau, W) in CFGS:
                H, ws = headers_by_weight(L_h, rng)
                p0, pW = rates(b, H, tau, W)
                p0t = binom_tail_ge(L_h, 0.5, tau)
                bal = np.abs(ws - L_h / 2.0) <= 0.5      # w = L_h/2 (or the two nearest if odd)
                byw = {}
                for w in range(L_h + 1):
                    sel = ws == w
                    if sel.any():
                        byw[w] = {"p0_max": float(p0[sel].max()),
                                  "p0_mean": float(p0[sel].mean()),
                                  "pW_max": float(pW[sel].max())}
                cell["cfg"]["%d/%d/W%d" % (L_h, tau, W)] = {
                    "p0_theory": p0t, "pW_theory": 1 - (1 - p0t)**W,
                    "p0_worst_all": float(p0.max()), "pW_worst_all": float(pW.max()),
                    "p0_worst_balanced": float(p0[bal].max()),
                    "pW_worst_balanced": float(pW[bal].max()),
                    "n_balanced": int(bal.sum()),
                    "hoeffding_p0_bound": float(np.exp(-2 * (tau - L_h / 2.0)**2 / L_h)),
                    "by_weight": byw}
            rec["%g" % d] = cell
            c = cell["cfg"]["10/9/W7"]
            print("%-20s d=%-5g pi=%.3f | p0 th=%.5f  worst(all)=%.5f  worst(BAL)=%.5f | "
                  "pW th=%.4f worst(all)=%.4f worst(BAL)=%.4f"
                  % (name, d, pi, c["p0_theory"], c["p0_worst_all"], c["p0_worst_balanced"],
                     c["pW_theory"], c["pW_worst_all"], c["pW_worst_balanced"]), flush=True)
        out["pools"][name] = rec
    with open(os.path.join(OUT, "probeA2_header_balance.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("WROTE probeA2_header_balance.json", flush=True)


if __name__ == "__main__":
    main()
