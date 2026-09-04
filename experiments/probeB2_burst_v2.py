"""Probe B2: the burst study of probeB, re-run on the v2 design configurations so that
Table II of the manuscript is internally consistent.  Gilbert-Elliott edits at matched
mean lambda; the iid-designed parameters are applied unchanged.  Reports the decodable
yield Y_exact (anchor on the true header AND the true frame index), not just raw hits."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from probeB_burst_channel import channel_ge
from probeD2_recapture import profile

RES = os.path.join(HERE, "..", "results")
EPS = 0.05
LAM_B = 0.5
BURSTS = [1, 5, 20]
N_FLOWS = 500


def run(lam, L_h, tau, R, L_c, N_max, E_B, rng, n_flows=N_FLOWS):
    T_f = L_h + L_c
    s = rng.integers(0, 2, size=L_h).astype(np.int8)
    Yh, Ye, runs = [], [], []
    hits = slots = 0
    for _ in range(n_flows):
        sent = np.empty(N_max * T_f, dtype=np.int8)
        for f in range(N_max):
            sent[f * T_f:f * T_f + L_h] = s
            sent[f * T_f + L_h:(f + 1) * T_f] = rng.integers(0, 2, size=L_c)
        recv, s2r = channel_ge(sent, lam, E_B, LAM_B, EPS, rng, want_map=True)
        prof = profile(recv, s)
        tp = s2r[np.arange(N_max) * T_f]
        anchor = int(tp[0]) if tp[0] >= 0 else 0
        yh = ye = 0
        run_m = run_max = 0
        P = prof.size
        for f in range(1, N_max):
            pred = anchor + T_f
            lo, hi = max(0, pred - R), min(P, pred + R + 1)
            if lo >= hi:
                run_m += 1; run_max = max(run_max, run_m); anchor = pred; continue
            seg = prof[lo:hi]
            j = int(np.argmax(seg)); best, bp = int(seg[j]), lo + j
            slots += 1
            if best >= tau:
                hits += 1
                yh += 1
                if tp[f] >= 0 and bp == tp[f]:
                    ye += 1
                anchor = bp; run_m = 0
            else:
                anchor = pred; run_m += 1; run_max = max(run_max, run_m)
        Yh.append(yh); Ye.append(ye); runs.append(run_max)
    Ye = np.array(Ye, dtype=float)
    n = Ye.size
    dead = float(np.mean(np.array(runs) >= 8))
    return {"q_emp": hits / max(slots, 1),
            "q_se": float(np.sqrt(hits / max(slots, 1) * (1 - hits / max(slots, 1)) / max(slots, 1))),
            "Y_hit": float(np.mean(Yh)), "Y_exact": float(Ye.mean()),
            "Y_exact_ci95": float(1.96 * Ye.std(ddof=1) / np.sqrt(n)),
            "Y_exact_med": float(np.median(Ye)),
            "blackout_frac": dead, "n_flows": int(n), "n_slots": int(slots)}


def main():
    rows = json.load(open(os.path.join(RES, "design_v2_table.json")))["rows"]
    rng = np.random.default_rng(4242)
    out = []
    print("%-6s %-10s | %s" % ("lam%", "L_h/tau/R",
          "  ".join("%-27s" % ("E_B=%d" % b) for b in BURSTS)))
    for key in sorted(rows, key=float):
        r = rows[key]
        if not r.get("feasible"):
            continue
        cells, rec = [], {"lam": r["lam"], "L_h": r["L_h"], "tau": r["tau"],
                          "R": r["R"], "N_max": r["N_max"], "muY_design": r["muY"],
                          "by_burst": {}}
        for E_B in BURSTS:
            m = run(r["lam"], r["L_h"], r["tau"], r["R"], r["L_c"],
                    r["N_max"], E_B, rng)
            rec["by_burst"][str(E_B)] = m
            cells.append("q%.3f blk%.2f Ye%5.1f+-%.1f"
                         % (m["q_emp"], m["blackout_frac"], m["Y_exact"], m["Y_exact_ci95"]))
        out.append(rec)
        print("%-6.1f %-10s | %s" % (r["lam"] * 100,
              "%d/%d/%d" % (r["L_h"], r["tau"], r["R"]), "  ".join(cells)), flush=True)
    json.dump(out, open(os.path.join(RES, "probeB2_burst_v2.json"), "w"), indent=1)
    print("\nq = per-slot hit rate (tight: aggregated over all slots);  blk = fraction of "
          "tracks with >=8\nconsecutive missed slots;  Ye = mean decodable frames with "
          "95% CI.\nWROTE probeB2_burst_v2.json")


if __name__ == "__main__":
    main()
