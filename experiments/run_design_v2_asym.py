"""Robustness of the v2 design to ASYMMETRIC insertion/deletion (R2-C1).
asym = (lam_i - lam_d)/lam in [0,1]; 0 = symmetric baseline, 1 = one-sided edits.
When R grows by one the disjoint-scan condition 2R <= L_c can break, so the codeword
region must grow to keep the false-alarm bound certifiable -- that, not lock loss,
is where asymmetry costs."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import design_v2 as D

RES = os.path.join(HERE, "..", "results")
LAMS = [0.02, 0.03, 0.05]
ASYMS = [0.0, 0.5, 1.0]


def main():
    out = []
    print("%-6s %-6s | design (L_h/tau/R, N, k, muY)" % ("lam%", "asym"))
    for lam in LAMS:
        for a in ASYMS:
            tag, r = "L_c=10", D.search(lam, L_obs=2000, track_cap=2000, asym=a, n_sim=40000)
            if r is None:
                tag, r = "L_c=14", D.search(lam, L_obs=2000, track_cap=2000, asym=a,
                                            L_c=14, n_sim=40000)
            if r is None:
                print("%-6.1f %-6.2f | INFEASIBLE even at L_c=14" % (lam * 100, a))
                out.append({"lam": lam, "asym": a, "feasible": False}); continue
            r.update({"asym": a, "feasible": True, "variant": tag})
            out.append(r)
            print("%-6.1f %-6.2f | L_h=%-3d tau=%-3d R=%-2d N=%-4d k=%-3d muY=%-5.1f "
                  "PD=%.4f  (%s)" % (lam * 100, a, r["L_h"], r["tau"], r["R"],
                  r["N_scr"], r["k_scr"], r["muY"], r["P_D_scan"], tag), flush=True)
    json.dump(out, open(os.path.join(RES, "design_v2_asym.json"), "w"), indent=1)
    print("\nWROTE design_v2_asym.json")


if __name__ == "__main__":
    main()
