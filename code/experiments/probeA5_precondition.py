"""Is there a MEASURABLE precondition on the background bit stream that flags the
cells where the balanced-header bound fails?  Candidate statistic: the largest
|lag-k autocorrelation| of the hard-decided background bits over k=1..K."""
import os, json
import numpy as np

TRACE = os.environ.get("NFW_TRACE_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..",
    "NFW2_Simulation", "results", "real_traffic"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
K = 16
MAX_BITS = 200_000

prev = json.load(open(os.path.join(OUT, "probeA4_v2configs.json")))
worst = {}
for r in prev["rows"]:
    key = (r["pool"], r["delta"])
    worst[key] = max(worst.get(key, 0.0), r["p0_ratio"])

rows = []
for (pool, d) in sorted(worst):
    p = os.path.join(TRACE, pool + ".npy")
    if not os.path.exists(p):
        continue
    x = np.load(p); x = x[np.isfinite(x) & (x >= 0)][:MAX_BITS]
    b = (np.rint(2.0 * x / d).astype(np.int64) & 1).astype(np.int8)
    s = (2.0 * b - 1.0); s -= s.mean()
    den = float(np.dot(s, s))
    rho = [abs(float(np.dot(s[:-k], s[k:]) / den)) if den > 0 else 0.0
           for k in range(1, K + 1)]
    rows.append({"pool": pool, "delta": d, "bias": float(b.mean()),
                 "rho_max": max(rho), "rho_argmax": int(np.argmax(rho)) + 1,
                 "rho1": rho[0], "p0_ratio_worst": worst[(pool, d)]})
rows.sort(key=lambda r: -r["p0_ratio_worst"])
json.dump({"K": K, "rows": rows}, open(os.path.join(OUT, "probeA5_precondition.json"), "w"), indent=1)
print("%-27s %-6s %-7s %-8s %-6s %-8s"
      % ("pool", "Delta", "bias", "max|rho_k|", "argmax", "p0 ratio"))
for r in rows:
    flag = "  <== BOUND FAILS" if r["p0_ratio_worst"] > 1.5 else ""
    print("%-27s %-6g %-7.3f %-8.4f k=%-4d %-8.2f%s"
          % (r["pool"], r["delta"], r["bias"], r["rho_max"], r["rho_argmax"],
             r["p0_ratio_worst"], flag))
