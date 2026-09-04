"""Probe C (answers R2-C3): the 1e-9 target cannot be observed by a finite Monte-Carlo,
so validate the CERTIFIED BOUND's ingredients instead, over a ladder of decades.

The bound P_FA <= M * p0 * P[Bin(N-1, W p0) >= k-1] rests on three pieces:
  (u1) union over the M starts        -- an inequality, nothing to validate;
  (u2) p_W <= W p0                    -- an inequality; we measure the slack;
  (u3) slot independence along a track when 2R <= L_c  -- THIS is the modelling step,
       and it is what the ladder tests: sweep k so the per-track confirmation
       probability spans 1e-2 .. 1e-6 and compare measurement with the bound.
"""
import os, json
import numpy as np
from math import comb

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
CFGS = json.load(open(os.path.join(OUT, "design_v2_table.json")))["rows"]


def binom_sf_ge(n, p, k):
    if k <= 0: return 1.0
    if k > n:  return 0.0
    return float(sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def track_mc(L_h, tau, R, L_c, N, ks, n_tracks, rng, batch=4000):
    """Run the real detector logic (predict, +-R scan, re-anchor on hit, extrapolate on
    miss) on iid fair background bits; return hit-count histogram over N-1 slots."""
    T_f = L_h + L_c
    W = 2 * R + 1
    off = np.arange(-R, R + 1)
    s = rng.integers(0, 2, size=L_h).astype(np.int8)
    counts = []
    done = 0
    while done < n_tracks:
        b = min(batch, n_tracks - done)
        span = (N + 2) * (T_f + R) + L_h + 2 * R + 8
        st = rng.integers(0, 2, size=(b, span), dtype=np.int8)
        anchor = np.full(b, R + 2, dtype=np.int64)
        c = np.zeros(b, dtype=np.int32)
        rows = np.arange(b)[:, None]
        for _ in range(N - 1):
            pred = anchor + T_f
            starts = pred[:, None] + off[None, :]
            idx = starts[:, :, None] + np.arange(L_h)[None, None, :]
            ag = np.sum(st[rows[:, :, None] if False else np.arange(b)[:, None, None], idx]
                        == s[None, None, :], axis=2)
            bi = np.argmax(ag, axis=1)
            best = ag[np.arange(b), bi]
            hit = best >= tau
            c += hit
            anchor = np.where(hit, starts[np.arange(b), bi], pred)
        counts.append(c)
        done += b
    return np.concatenate(counts)


rng = np.random.default_rng(90210)
res = []
for row in CFGS.values() if isinstance(CFGS, dict) else CFGS:
    if not row.get("feasible"):
        continue
    if row["lam"] not in (0.02, 0.05):
        continue
    L_h, tau, R, L_c, N, k0 = row["L_h"], row["tau"], row["R"], row["L_c"], row["N_scr"], row["k_scr"]
    p0 = row["p0_bound"]; W = row["W"]; pW_bound = min(1.0, W * p0)
    NT = 200000
    c = track_mc(L_h, tau, R, L_c, N, k0, NT, rng)
    pW_emp = float(c.sum() / (NT * (N - 1)))
    print("lam=%.3f  L_h=%d tau=%d R=%d N=%d | p0=%.6f  pW union bound=%.5f  "
          "pW empirical=%.5f  (slack %.2fx)" % (row["lam"], L_h, tau, R, N, p0,
          pW_bound, pW_emp, pW_bound / max(pW_emp, 1e-12)), flush=True)
    lad = []
    print("   %-5s %-12s %-12s %-8s" % ("k", "emp P(>=k-1)", "bound", "bound/emp"))
    for k in range(2, k0 + 1):
        emp = float((c >= k - 1).mean())
        bnd = binom_sf_ge(N - 1, pW_bound, k - 1)
        if emp == 0 and bnd < 1e-9:
            break
        lad.append({"k": k, "emp": emp, "bound": bnd, "n_tracks": NT})
        if emp > 0:
            print("   %-5d %-12.3e %-12.3e %-8.1f" % (k, emp, bnd, bnd / emp))
        else:
            print("   %-5d %-12s %-12.3e %-8s" % (k, "0 (<5e-6)", bnd, ">%.0f" % (bnd / (1.0 / NT))))
    res.append({"lam": row["lam"], "L_h": L_h, "tau": tau, "R": R, "N": N, "k_design": k0,
                "p0_bound": p0, "pW_union_bound": pW_bound, "pW_empirical": pW_emp,
                "ladder": lad})
json.dump(res, open(os.path.join(OUT, "probeC_ladder.json"), "w"), indent=1)
print("\nWROTE probeC_ladder.json")
