"""
Probe A (resubmission, answers R3-C3 / R2-C5):
Test the H0 assumption -- "hard-decided bits of UNMARKED real traffic are iid fair" --
on real per-packet IPD pools extracted from ISCXTor2016 / ISCXVPN2016 pcaps.

Detector bit rule (nearest-lattice QIM hard decision, two interleaved lattices
{k*Delta} and {k*Delta + Delta/2}):   b = round(2x/Delta) mod 2 .

Metrics per (app class, Delta):
  bias      = P(b=1)                      (theory 0.5)
  rho_k     = lag-k autocorr of (2b-1)    (theory 0)
  p0_emp    = P(agreement over L_h >= tau) averaged over random headers  (theory P(Bin(L_h,.5)>=tau))
  pW_emp    = P(max agreement over W consecutive offsets >= tau)         (theory 1-(1-p0)^W)
"""
import os, json, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRACE = os.environ.get("NFW_TRACE_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..",
    "NFW2_Simulation", "results", "real_traffic"))
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)

# (L_h, tau, R) rows of Table I
CONFIGS = [(9, 8, 2), (10, 9, 3), (11, 10, 4), (12, 11, 5), (11, 11, 6)]
DELTAS_MS = [0.05, 0.2, 1.0, 5.0, 19.0, 50.0]
POOLS = ["iscx_tor_ipd", "iscx_vpn_ipd", "iscx_tor_chat_ipd",
         "iscx_tor_video_ipd", "iscx_vpn_audio_ipd", "iscx_tor_file_transfer_ipd"]

N_HEADERS = 24           # random headers averaged over
MAX_BITS = 400_000     # cap per pool for the correlation sweep


def binom_tail_ge(n, p, k):
    """P(Bin(n,p) >= k) by exact summation (no scipy on this box)."""
    from math import comb
    return float(sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1)))


def bits_from_ipd(x, delta):
    return (np.rint(2.0 * x / delta).astype(np.int64) & 1).astype(np.int8)


def autocorr(sig, lags=(1, 2, 3, 4, 5)):
    s = sig.astype(np.float64)
    s = s - s.mean()
    den = float(np.dot(s, s))
    if den <= 0:
        return {str(k): 0.0 for k in lags}
    return {str(k): float(np.dot(s[:-k], s[k:]) / den) for k in lags}


def scan_rates(bits, L_h, tau, W, rng, n_headers=N_HEADERS):
    """Empirical p0 (single offset) and pW (max over W consecutive offsets)."""
    s = (2 * bits.astype(np.int16) - 1)
    n = s.size
    if n < L_h + W + 10:
        return None
    p0s, pWs = [], []
    for _ in range(n_headers):
        h = (2 * rng.integers(0, 2, size=L_h).astype(np.int16) - 1)
        corr = np.correlate(s, h, mode="valid")          # length n-L_h+1
        agree = (L_h + corr) // 2                        # matched-bit count
        hit = agree >= tau
        p0s.append(float(hit.mean()))
        # window hit: any of W consecutive offsets
        m = hit.size - W + 1
        if m <= 0:
            continue
        cs = np.concatenate(([0], np.cumsum(hit.astype(np.int64))))
        wsum = cs[W:] - cs[:-W]
        pWs.append(float((wsum > 0).mean()))
    return float(np.mean(p0s)), float(np.std(p0s)), float(np.mean(pWs))



def scan_rates_multi(bits, L_h, taus_Ws, rng, n_headers=N_HEADERS):
    """One matmul for all headers. Returns {(tau,W): (p0_mean,p0_sd,pW_mean)}."""
    from numpy.lib.stride_tricks import sliding_window_view
    s = (2 * bits.astype(np.float32) - 1.0)
    n = s.size
    if n < L_h + 64:
        return None
    win = sliding_window_view(s, L_h)                       # (n-L_h+1, L_h)
    H = (2 * rng.integers(0, 2, size=(L_h, n_headers)).astype(np.float32) - 1.0)
    corr = win @ H                                          # (m, n_headers)
    agree = (L_h + corr) * 0.5
    res = {}
    for (tau, W) in taus_Ws:
        hit = (agree >= tau - 0.25)
        p0v = hit.mean(axis=0)
        cs = np.concatenate([np.zeros((1, n_headers), dtype=np.int64),
                             np.cumsum(hit.astype(np.int64), axis=0)], axis=0)
        m = hit.shape[0] - W + 1
        if m <= 0:
            continue
        wsum = cs[W:] - cs[:-W]
        pWv = (wsum > 0).mean(axis=0)
        res[(tau, W)] = (float(p0v.mean()), float(p0v.std()), float(pWv.mean()),
                         float(p0v.max()), float(pWv.max()))
    return res


def main():
    rng = np.random.default_rng(20260830)
    out = {"note": "H0 fair-bit assumption on real ISCX Tor/VPN IPDs",
           "configs": CONFIGS, "deltas_ms": DELTAS_MS, "n_headers": N_HEADERS,
           "pools": {}}
    for name in POOLS:
        path = os.path.join(TRACE, name + ".npy")
        if not os.path.exists(path):
            print("skip", name, flush=True)
            continue
        x = np.load(path)
        x = x[np.isfinite(x) & (x >= 0)]
        rec = {"n_ipd": int(x.size),
               "ipd_ms": {"median": float(np.median(x)), "mean": float(x.mean()),
                          "p25": float(np.percentile(x, 25)),
                          "p75": float(np.percentile(x, 75)),
                          "p99": float(np.percentile(x, 99)),
                          "frac_below_1ms": float((x < 1.0).mean())},
               "delta": {}}
        xs = x[:MAX_BITS]
        for d in DELTAS_MS:
            b = bits_from_ipd(xs, d)
            e = {"bias": float(b.mean()), "autocorr": autocorr(2 * b.astype(np.int16) - 1)}
            e["cfg"] = {}
            byL = {}
            for (L_h, tau, R) in CONFIGS:
                byL.setdefault(L_h, []).append((tau, 2 * R + 1))
            cache = {}
            for L_h, tw in byL.items():
                cache[L_h] = scan_rates_multi(b, L_h, tw, rng)
            for (L_h, tau, R) in CONFIGS:
                W = 2 * R + 1
                r = None if cache.get(L_h) is None else cache[L_h].get((tau, W))
                if r is None:
                    continue
                p0e, p0sd, pWe, p0max, pWmax = r
                p0t = binom_tail_ge(L_h, 0.5, tau)
                e["cfg"]["%d/%d/R%d" % (L_h, tau, R)] = {
                    "p0_theory": p0t, "p0_emp": p0e, "p0_emp_sd_over_headers": p0sd, "p0_emp_worst_header": p0max, "pW_emp_worst_header": pWmax,
                    "p0_ratio": p0e / p0t if p0t > 0 else None,
                    "pW_theory": 1 - (1 - p0t)**W, "pW_emp": pWe,
                    "pW_ratio": pWe / (1 - (1 - p0t)**W) if p0t > 0 else None}
            rec["delta"]["%g" % d] = e
            print("%-30s d=%-6g bias=%.4f rho1=%+.4f  p0ratio(10/9)=%s"
                  % (name, d, e["bias"], e["autocorr"]["1"],
                     ("%.2f" % e["cfg"]["10/9/R3"]["p0_ratio"]) if "10/9/R3" in e["cfg"] else "-"),
                  flush=True)
        out["pools"][name] = rec
    with open(os.path.join(OUT, "probeA_real_trace_p0.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("WROTE", os.path.join(OUT, "probeA_real_trace_p0.json"), flush=True)


if __name__ == "__main__":
    main()
