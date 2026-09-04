"""Probe J: a genuine two-stage detector, so that a fast-rejection claim can be made
honestly (the submitted letter claimed one that its own computation did not support).

Stage 1  a loose screen over L1 packets: certified P_FA1 <= alpha1, scan P_D1 >= 0.999.
Stage 2  the certified full-window test, UNCHANGED, run only on stage-1 survivors.

Soundness.  Stage 1 only saves time; it never relaxes stage 2, so the overall false alarm
is still bounded by stage 2's certified alpha.  On the detection side a union bound gives
    P_D >= 1 - (1 - P_D1) - (1 - P_D2),
so with P_D1 >= 0.999 the stage-2 target must be raised to 0.991 to keep P_D >= 0.99.
Expected rejection latency over background flows:
    E[L] = (1 - P_FA1) * L1 + P_FA1 * L_obs .
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import design_v2 as D

RES = os.path.join(HERE, "..", "results")
L_OBS = 2000
ALPHAS = [0.01, 0.03, 0.1, 0.3]
L1_GRID = [300, 400, 500, 600, 800]
PD1_TARGET = 0.999
M_Q, F_FA = 0.15, 2.0


def main():
    rows = json.load(open(os.path.join(RES, "design_v2_table.json")))["rows"]
    out = []
    print("%-6s %-10s %-7s | %-4s %-4s %-5s %-9s %-8s %-9s"
          % ("lam%", "L_h/tau/R", "alpha1", "N1", "k1", "L1", "P_FA1", "P_D1", "E[reject]"))
    for key in sorted(rows, key=float):
        r = rows[key]
        if not r.get("feasible"):
            continue
        L_h, tau, R, W, T_f = r["L_h"], r["tau"], r["R"], r["W"], r["T_f"]
        p0c = min(1.0, F_FA * r["p0_bound"])
        qch, qcch = (1 - M_Q) * r["q"], (1 - M_Q) * r["q_cap"]
        disj = 2 * R <= r["L_c"]
        found = None
        for alpha1 in ALPHAS:
            best = None
            for L1 in L1_GRID:
                for N in range(2, L1 // T_f + 1):
                    span = (N - 1) * T_f + L_h
                    if span > L1:
                        break
                    M = L1 - span + 1
                    J = max(0, (L1 - span + 1) // T_f)
                    kmin = None
                    for k in range(2, N + 1):
                        if D.p_fa_certified(p0c, W, N, k, M, disj) <= alpha1:
                            kmin = k
                            break
                    if kmin is None:
                        continue
                    if D.p_d_product(qcch, qch, N, kmin, J) < PD1_TARGET:
                        continue          # FKG upper bound: a valid relaxation
                    pd = D.p_d_scan(qcch, qch, N, kmin, J, L1 // T_f, n_sim=40000)
                    if pd < PD1_TARGET:
                        continue
                    pfa = D.p_fa_certified(p0c, W, N, kmin, M, disj)
                    lat = (1 - pfa) * L1 + pfa * L_OBS
                    if best is None or lat < best[-1]:
                        best = (L1, N, kmin, pfa, pd, lat)
                if best:
                    break
            if best:
                found = (alpha1,) + best
                break
        if found:
            a1, L1, N, k, pfa, pd, lat = found
            out.append({"lam": r["lam"], "L_h": L_h, "tau": tau, "R": R, "alpha1": a1,
                        "L1": L1, "N1": N, "k1": k, "P_FA1": pfa, "P_D1": pd,
                        "E_reject_packets": lat, "L_obs": L_OBS})
            print("%-6.1f %-10s %-7.2g | %-4d %-4d %-5d %-9.3f %-8.5f %-9.0f"
                  % (r["lam"] * 100, "%d/%d/%d" % (L_h, tau, R), a1, N, k, L1, pfa, pd, lat),
                  flush=True)
        else:
            out.append({"lam": r["lam"], "feasible": False})
            print("%-6.1f %-10s   no stage-1 screen with P_D1>=%.3f up to alpha1=%.1f"
                  % (r["lam"] * 100, "%d/%d/%d" % (L_h, tau, R), PD1_TARGET, ALPHAS[-1]))
    json.dump(out, open(os.path.join(RES, "probeJ_two_stage.json"), "w"), indent=1)
    print("\nWROTE probeJ_two_stage.json")


if __name__ == "__main__":
    main()
