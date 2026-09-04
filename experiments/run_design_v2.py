"""Resumable driver for the v2 design table: one lambda per call, rows already in the JSON are skipped, so the search can be run in short slices."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import design_v2 as D

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "design_v2_table.json")
LAMS = [0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
KW = dict(L_obs=2000, track_cap=2000, L_max=2000, require_even=True,
          require_disjoint=True, L_c=10, n_sim=80000, sep_min=10.0)

def load():
    if os.path.exists(OUT):
        return json.load(open(OUT))
    return {"variant": "v2: L_obs=L_max=2000 (= rejection latency); even L_h + balanced "
                       "header; certified union-bound P_FA with disjoint scans; "
                       "scan-statistic P_D>=0.99", "kw": {k: v for k, v in KW.items()},
            "rows": {}}

def main(todo):
    st = load()
    for lam in todo:
        key = "%g" % lam
        if key in st["rows"]:
            print("skip", key); continue
        r = D.search(lam, **KW)
        st["rows"][key] = r if r else {"lam": lam, "feasible": False}
        if r:
            r["feasible"] = True
            print("%-6.1f L_h=%-3d tau=%-3d R=%-2d N=%-4d k=%-4d span=%-5d Nmax=%-4d "
                  "muY=%-6.1f PFA=%.2e PD=%.4f" % (lam*100, r['L_h'], r['tau'], r['R'],
                  r['N_scr'], r['k_scr'], r['track_span'], r['N_max'], r['muY'],
                  r['P_FA_cert'], r['P_D_scan']), flush=True)
        else:
            print("%-6.1f INFEASIBLE" % (lam*100), flush=True)
        json.dump(st, open(OUT, "w"), indent=1)

if __name__ == "__main__":
    todo = [float(a) for a in sys.argv[1:]] or LAMS
    main(todo)
