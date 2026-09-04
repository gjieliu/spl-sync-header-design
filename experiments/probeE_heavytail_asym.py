"""Probe E: (R1-C1) heavy-tailed / non-Gaussian jitter, and (R2-C1) asymmetric edits.

(1) eps under non-Gaussian jitter.  With the two interleaved lattices {m*Delta} and
    {m*Delta + Delta/2}, the nearest-lattice hard decision returns
        b_hat = round(2x/Delta) mod 2 = (b + round(2*xi/Delta)) mod 2,
    so the symbol error rate is EXACTLY  eps = P[ round(2*xi/Delta) is odd ],
    a FOLDED tail probability.  The Gaussian formula eps ~ 2Q(Delta/4 sigma) keeps only
    the first fold; heavy tails put mass in the far folds, which drives eps toward 1/2.
    xi is the difference of adjacent per-packet delays accumulated over K hops.

(2) asymmetric insertion/deletion.  drift mean = (lam_i - lam_d) * T_f is no longer 0,
    so the +-R window must cover a BIASED walk.  We re-size R with the same cumulative
    lock rule and report the cost.
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import design_v2 as D

OUT = os.path.join(HERE, "..", "results")
K_HOPS = 3
NS = 400000


def jitter(kind, n, rng):
    """One-way per-hop per-packet delay samples (unit-free); xi is built from these."""
    if kind == "gaussian":   return rng.normal(0, 1, n)
    if kind == "laplace":    return rng.laplace(0, 1, n)
    if kind == "t5":         return rng.standard_t(5, n)
    if kind == "t3":         return rng.standard_t(3, n)
    if kind == "lognormal":  return rng.lognormal(0, 1.0, n)
    if kind == "pareto3":    return (rng.pareto(3.0, n) + 1.0)
    if kind == "pareto2.5":  return (rng.pareto(2.5, n) + 1.0)
    raise ValueError(kind)


def xi_samples(kind, n, rng, K=K_HOPS):
    x = np.zeros(n)
    for _ in range(K):
        x += jitter(kind, n, rng) - jitter(kind, n, rng)   # adjacent-packet difference
    return x


def eps_of(kind, ratios, rng):
    xi = xi_samples(kind, NS, rng)
    sd = float(xi.std())
    out = {}
    for r in ratios:                       # r = Delta / sigma_xi
        Delta = r * sd
        out[r] = float((np.rint(2.0 * xi / Delta).astype(np.int64) & 1).mean())
    # tail heaviness diagnostics
    return out, sd, float(np.percentile(np.abs(xi), 99.9) / sd)


RATIOS = [4, 6, 7.8, 10, 12, 16, 20]
rng = np.random.default_rng(2718)
print("(1) symbol error rate eps vs Delta/sigma_xi   (K=%d hops, exact folded decision)" % K_HOPS)
print("%-11s %-8s | %s" % ("jitter", "p99.9/sd", "  ".join("%7s" % ("D/s=%g" % r) for r in RATIOS)))
eps_tab = {}
for kind in ["gaussian", "laplace", "t5", "t3", "lognormal", "pareto3", "pareto2.5"]:
    e, sd, tail = eps_of(kind, RATIOS, rng)
    eps_tab[kind] = {"eps": {str(k): v for k, v in e.items()}, "tail_p999_over_sd": tail}
    print("%-11s %-8.2f | %s" % (kind, tail, "  ".join("%7.4f" % e[r] for r in RATIOS)))
print("\n  (paper's operating point: Delta/sigma_xi = 7.8 -> eps = 5%% under Gaussian)")

print("\n(2) asymmetric edits: R needed for the same cumulative-lock target")
print("%-6s %-6s | %s" % ("lam%", "T_f", "  ".join("%-14s" % ("asym=%.2f" % a)
                                                   for a in [0.0, 0.25, 0.5, 1.0])))
asym_tab = []
rows = json.load(open(os.path.join(OUT, "design_v2_table.json")))["rows"]
for key in sorted(rows, key=lambda k: float(k)):
    r = rows[key]
    if not r.get("feasible"):
        continue
    T_f, N_max, lam = r["T_f"], r["N_max"], r["lam"]
    cells, rec = [], {"lam": lam, "T_f": T_f, "N_max": N_max, "R": {}}
    for a in [0.0, 0.25, 0.5, 1.0]:
        R, P_in = D.size_R_cumulative(T_f, lam, N_max, 1e-3, asym=a)
        rec["R"][str(a)] = {"R": int(R), "P_in": float(P_in),
                            "disjoint_ok": bool(2 * R <= r["L_c"])}
        cells.append("R=%-2d %-9s" % (R, "(2R<=L_c)" if 2 * R <= r["L_c"] else "(VIOLATES)"))
    asym_tab.append(rec)
    print("%-6.1f %-6d | %s" % (lam * 100, T_f, "  ".join(cells)))
json.dump({"eps_vs_jitter": eps_tab, "ratios": RATIOS, "K_hops": K_HOPS,
           "asymmetry": asym_tab},
          open(os.path.join(OUT, "probeE_heavytail_asym.json"), "w"), indent=1)
print("\nWROTE probeE_heavytail_asym.json")
