"""Probe D (answers R1-C3 / R2-C6): does the aligned-frame yield certify end-to-end
payload recovery, and what does an explicit re-capture rule buy?

Three yields are separated:
  Y_hit    frames the detector calls aligned            (what the letter reports)
  Y_exact  of those, anchored on the TRUE header position and the TRUE frame index
           -> the only frames an (n,k) cross-frame code can actually use
  slips    hits at a nonzero multiple of T_f (periodic lock kept, frame INDEX wrong)
Re-capture rule: after m consecutive missed slots the track is declared stale and the
detector re-acquires by scanning a widened +-R_rec window around the prediction.
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
import sys; sys.path.insert(0, os.path.dirname(HERE))
from probeB_burst_channel import channel_ge

EPS = 0.05
N_FLOWS = 250
M_STALE = 4          # consecutive misses that trigger re-capture
REC_MULT = 3         # re-capture scans +-(REC_MULT*R)


def run(lam, L_h, tau, R, L_c, N_max, rng, recapture, n_flows=N_FLOWS):
    T_f = L_h + L_c
    s = rng.integers(0, 2, size=L_h).astype(np.int8)
    Yh, Ye, nslip, nhit, nrec, nrec_ok = [], [], 0, 0, 0, 0
    for _ in range(n_flows):
        sent = np.empty(N_max * T_f, dtype=np.int8)
        for f in range(N_max):
            sent[f * T_f:f * T_f + L_h] = s
            sent[f * T_f + L_h:(f + 1) * T_f] = rng.integers(0, 2, size=L_c)
        recv, s2r = channel_ge(sent, lam, 1, 0.5, EPS, rng, want_map=True)
        tp = s2r[np.arange(N_max) * T_f]
        anchor = int(tp[0]) if tp[0] >= 0 else 0
        yh = ye = 0
        run_miss = 0
        for f in range(1, N_max):
            pred = anchor + T_f
            rad = R
            if recapture and run_miss >= M_STALE:
                rad = REC_MULT * R
                nrec += 1
            ba, bp = -1, pred
            for o in range(-rad, rad + 1):
                st = pred + o
                if st < 0 or st + L_h > recv.size:
                    continue
                a = int(np.sum(recv[st:st + L_h] == s))
                if a > ba:
                    ba, bp = a, st
            hit = ba >= tau
            if hit:
                nhit += 1
                yh += 1
                if tp[f] >= 0 and bp == tp[f]:
                    ye += 1
                else:
                    nslip += 1
                if rad != R:
                    nrec_ok += 1
                run_miss = 0
            else:
                run_miss += 1
            anchor = bp if hit else pred
        Yh.append(yh); Ye.append(ye)
    Yh = np.array(Yh); Ye = np.array(Ye)
    return {"Y_hit_mean": float(Yh.mean()), "Y_exact_mean": float(Ye.mean()),
            "Y_exact_over_Y_hit": float(Ye.mean() / max(Yh.mean(), 1e-9)),
            "Y_exact_p05": float(np.percentile(Ye, 5)),
            "Y_exact_min": int(Ye.min()),
            "slip_frac_of_hits": nslip / max(nhit, 1),
            "n_recapture_slots": nrec, "n_recapture_hits": nrec_ok,
            "Y_exact_all": Ye.tolist()}


rows = json.load(open(os.path.join(OUT, "design_v2_table.json")))["rows"]
rng = np.random.default_rng(31337)
res = []
print("%-6s %-9s | %-8s %-8s %-7s %-7s | %-8s %-8s %-7s"
      % ("lam%", "L_h/tau/R", "Y_hit", "Y_exact", "ratio", "slip", "Y_hit+rec",
         "Y_ex+rec", "gain"))
for key in sorted(rows, key=lambda k: float(k)):
    r = rows[key]
    if not r.get("feasible"):
        continue
    a = run(r["lam"], r["L_h"], r["tau"], r["R"], r["L_c"], r["N_max"], rng, False)
    b = run(r["lam"], r["L_h"], r["tau"], r["R"], r["L_c"], r["N_max"], rng, True)
    print("%-6.1f %-9s | %-8.1f %-8.1f %-7.3f %-7.4f | %-8.1f %-8.1f %+6.1f%%"
          % (r["lam"] * 100, "%d/%d/%d" % (r["L_h"], r["tau"], r["R"]),
             a["Y_hit_mean"], a["Y_exact_mean"], a["Y_exact_over_Y_hit"],
             a["slip_frac_of_hits"], b["Y_hit_mean"], b["Y_exact_mean"],
             100 * (b["Y_exact_mean"] / max(a["Y_exact_mean"], 1e-9) - 1)), flush=True)
    res.append({"lam": r["lam"], "L_h": r["L_h"], "tau": r["tau"], "R": r["R"],
                "N_max": r["N_max"], "muY_design": r["muY"],
                "no_recapture": {k: v for k, v in a.items() if k != "Y_exact_all"},
                "with_recapture": {k: v for k, v in b.items() if k != "Y_exact_all"},
                "Y_exact_all_norec": a["Y_exact_all"],
                "Y_exact_all_rec": b["Y_exact_all"],
                "recapture_rule": {"m_stale": M_STALE, "R_mult": REC_MULT}})
json.dump(res, open(os.path.join(OUT, "probeD_endtoend.json"), "w"), indent=1)
print("\nWROTE probeD_endtoend.json")
