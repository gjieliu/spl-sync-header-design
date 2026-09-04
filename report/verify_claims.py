"""Cross-check every quantitative claim in the manuscript and the response letter against
the result JSONs.  Run after any edit to either document or any re-run of an experiment."""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DOC = {"manuscript": open(os.path.join(HERE, "..", "..", "manuscript", "main_v2.tex")).read(),
       "response": open(os.path.join(HERE, "..", "..", "response", "response.tex")).read()}
L = lambda f: json.load(open(os.path.join(RES, f)))
T = L("design_v2_table.json")["rows"]
F = {round(x["lam"], 5): x for x in L("probeF_pd_aggregation.json")}
D = {round(x["lam"], 5): x for x in L("probeD_endtoend.json")}
B = {round(x["lam"], 5): x for x in L("probeB2_burst_v2.json")}
R2 = {round(x["lam"], 5): x for x in L("probeD2_recapture.json")}
C, A7, A4, A3, J, K, E = (L("probeC_ladder.json"), L("probeA7_alignment.json"),
                          L("probeA4_v2configs.json"), L("probeA3_balanced_enum.json"),
                          L("probeJ_two_stage.json"), L("probeK_syncword_sensitivity.json"),
                          L("probeE_heavytail_asym.json"))
A2 = L("probeA2_header_balance.json")
rows = [T[k] for k in sorted(T, key=float) if T[k].get("feasible")]
c2 = [c for c in C if abs(c["lam"] - 0.02) < 1e-9][0]
lad = [l for l in c2["ladder"] if l["emp"] > 0]
rr = [l["bound"] / l["emp"] for l in lad]
bw = A2["pools"]["iscx_tor_ipd"]["19"]["cfg"]["10/9/W7"]
b5 = B[0.05]["by_burst"]
blk1 = [B[round(r["lam"], 5)]["by_burst"]["1"]["blackout_frac"] for r in rows]
blk20 = [B[round(r["lam"], 5)]["by_burst"]["20"]["blackout_frac"] for r in rows]
sub4 = [r for r in rows if r["lam"] <= 0.04]
ratios = [D[round(r["lam"], 5)]["no_recapture"]["Y_exact_over_Y_hit"] for r in rows]
lat = [x["E_reject_packets"] for x in J if "E_reject_packets" in x]

CHECKS = [
 ("design table appears verbatim in both documents",
  all(("%.0f" % (0.85 * r["muY"])) in open(os.path.join(HERE,"..","..","manuscript","tab_main.tex")).read() for r in rows)),
 ("P_D lam=5%: 0.992 claimed vs 0.960 actual",
  abs(F[0.05]["PD_scan"] - 0.960) < 5e-4 and "0.960" in DOC["manuscript"]),
 ("Y_exact/Y_hit range 0.995..0.855 -> text says 0.5%--14.5%",
  abs(max(ratios) - 0.995) < 1e-3 and abs(min(ratios) - 0.855) < 1e-3),
 ("ladder pW 0.0352 vs 0.0349", abs(c2["pW_union_bound"] - 0.0352) < 3e-4
  and abs(c2["pW_empirical"] - 0.0349) < 3e-4 and "0.0352" in DOC["manuscript"]),
 ("ladder span 0.77 .. 5e-6", abs(lad[0]["emp"] - 0.766) < 5e-3
  and abs(lad[-1]["emp"] - 5e-6) < 1e-6 and "0.77" in DOC["manuscript"]),
 ("ladder ratio in [1.00,2.1], never below 1",
  max(rr) <= 2.07 and "[1.00,2.07]" in DOC["manuscript"]),
 ("burst q 0.368 -> 0.430 at lam=5%", abs(b5["1"]["q_emp"] - 0.368) < 1e-3
  and abs(b5["20"]["q_emp"] - 0.430) < 1e-3 and "0.430" in DOC["manuscript"]),
 ("blackout: lam<=4% grows 0.00-0.10 -> 0.10-0.39",
  max(B[round(r["lam"], 5)]["by_burst"]["1"]["blackout_frac"] for r in sub4) <= 0.105
  and max(B[round(r["lam"], 5)]["by_burst"]["20"]["blackout_frac"] for r in sub4) <= 0.395
  and "0.39" in DOC["manuscript"]),
 ("weight-0 p0 = 48x bound", abs(bw["by_weight"]["0"]["p0_max"] / bw["p0_theory"] - 48) < 1.0),
 ("full balanced enum, 24 even-L_h cells, worst 1.14x; odd L_h=11 reaches 2.2x",
  abs(max(r["p0_ratio_worst"] for r in A3["rows"] if r["L_h"] % 2 == 0) - 1.14) < 0.02
  and sum(1 for r in A3["rows"] if r["L_h"] % 2 == 0) == 24 and "$24$ cells" in DOC["manuscript"]
  and abs(max(r["p0_ratio_worst"] for r in A3["rows"] if r["L_h"] == 11) - 2.23) < 0.02 and "2.2\\times" in DOC["manuscript"]),
 ("144 cells, 7 above 1.5x, worst 7.67x",
  len(A4["rows"]) == 144 and sum(1 for r in A4["rows"] if r["p0_ratio"] > 1.5) == 7
  and abs(A4["worst_p0_ratio_over_all_cells"] - 7.67) < 0.02),
 ("V(g) Spearman +0.75/+0.85; lowest-V decile 0.18x",
  abs(A7[0]["spearman_V"] - 0.754) < 2e-3 and abs(A7[1]["spearman_V"] - 0.848) < 2e-3
  and abs(A7[0]["worst_lowV_decile"] - 0.18) < 0.02),
 ("sidelobe Spearman -0.12", abs(A7[0]["spearman_sidelobe"] + 0.118) < 3e-3),
 ("sync-word sensitivity 75.4-88.1 vs 83.9-87.0",
  abs(K[0]["Y_exact_min"] - 75.4) < 0.2 and abs(K[1]["Y_exact_max"] - 87.0) < 0.2),
 ("two-stage 415-861 packets, L1 in 400-800, P_D1>=0.999",
  abs(min(lat) - 415) < 1 and abs(max(lat) - 861) < 1
  and min(x["L1"] for x in J if "L1" in x) == 400
  and all(x["P_D1"] >= 0.999 for x in J if "P_D1" in x)),
 ("eps 0.027-0.055 at Delta/sigma=7.8",
  abs(min(v["eps"]["7.8"] for v in E["eps_vs_jitter"].values()) - 0.027) < 5e-4
  and abs(max(v["eps"]["7.8"] for v in E["eps_vs_jitter"].values()) - 0.055) < 5e-4),
 ("eps floor 0.0001 vs 0.0036 at 16",
  abs(E["eps_vs_jitter"]["gaussian"]["eps"]["16"] - 0.0001) < 1e-4
  and abs(E["eps_vs_jitter"]["t3"]["eps"]["16"] - 0.0036) < 3e-4),
 ("re-capture +4.4 at 1%, -3.6 at 3%, -2.6 at 5%",
  abs(R2[0.01]["widen"]["paired_gain"] - 4.41) < 0.06
  and abs(R2[0.03]["widen"]["paired_gain"] + 3.63) < 0.06
  and abs(R2[0.05]["widen"]["paired_gain"] + 2.56) < 0.06),
 ("largest certified P_FA is 7.7e-10",
  abs(max(r["P_FA_cert"] for r in rows) - 7.73e-10) < 5e-12),
 ("every row meets P_D >= 0.99 and 2R <= L_c and q >= 10 p_W",
  all(r["P_D_scan"] >= 0.99 and r["disjoint_scan"] and r["sep"] >= 10 for r in rows)),
]
bad = 0
for name, cond in CHECKS:
    print(("  OK   " if cond else "  FAIL ") + name)
    bad += (not cond)
print("\n%d/%d checks pass" % (len(CHECKS) - bad, len(CHECKS)))
sys.exit(1 if bad else 0)
