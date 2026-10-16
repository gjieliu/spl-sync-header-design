"""Check reported results and local-revision consistency against the result JSONs.
Run after edits to the manuscript, response or affected calculations."""
import json, os, re, sys, math
from pathlib import Path
from fractions import Fraction
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import design_v2 as core
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
RESP_PATH = os.path.join(HERE, "..", "..", "response", "response.tex")
DOC = {"manuscript": open(os.path.join(HERE, "..", "..", "manuscript", "main_v2.tex")).read(),
       "response": open(RESP_PATH).read() if os.path.exists(RESP_PATH) else None}
RESPONSE_CHECKS = {
    "largest finite-window certificate is below upward display 9.662e-10",
    "response table displays all seven bounds rounded upward",
    "reviewer replies cover all three comments at manuscript section locations",
}
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
lat = [x['E_reject_ipds_upper'] for x in J]
TAB = (Path(HERE).parents[1] / 'manuscript' / 'tab_main.tex').read_text()
REF = Path(RES) / 'r1_reference'
ORIGINAL = json.loads((REF / 'design_v2_table.json').read_text())['rows']
J_ORIGINAL = json.loads((REF / 'probeJ_two_stage.json').read_text())
RECHECK = L('recheck_local.json')

CHECKS = [
 ("displayed conservative yields match the result rows",
  all(('%.0f' % (0.85 * r['muY'])) in TAB for r in rows)),
 ("old product-accepted candidate has nominal P_D 0.960",
  abs(F[0.05]["PD_scan"] - 0.960) < 5e-4 and "0.960" in DOC["manuscript"]),
 ("Y_exact/Y_hit range 0.995..0.855 -> text says 0.5%--14.5%",
  abs(max(ratios) - 0.995) < 1e-3 and abs(min(ratios) - 0.855) < 1e-3),
 ("decodable yield exceeds muY_dec by at least 14% in every row",
  min(D[round(r["lam"], 5)]["no_recapture"]["Y_exact_mean"] / (0.85 * r["muY"])
      for r in rows) >= 1.14 and 'by $14\\%$ or more' in DOC['manuscript']),
 ("ladder pW 0.0352 vs 0.0349", abs(c2["pW_union_bound"] - 0.0352) < 3e-4
  and abs(c2["pW_empirical"] - 0.0349) < 3e-4 and "0.0352" in DOC["manuscript"]),
 ("ladder span 0.77 .. 5e-6", abs(lad[0]["emp"] - 0.766) < 5e-3
  and abs(lad[-1]["emp"] - 5e-6) < 1e-6 and "0.77" in DOC["manuscript"]),
 ("ladder ratio up to 2.07; one sub-unity point disclosed",
  max(rr) <= 2.07 and sum(x < 1 for x in rr) == 1
  and '[1.00,2.07]' in DOC['manuscript']
  and 'one value below unity' in DOC['manuscript']),
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
 ("two-stage iid rejection upper bounds round upward to 418-871 IPDs",
  math.ceil(min(lat)) == 418 and math.ceil(max(lat)) == 871
  and min(x['L1'] for x in J) == 400 and max(x['L1'] for x in J) == 800
  and all(x['P_D1'] >= 0.999 and x['P_FA1'] <= x['alpha1'] for x in J)
  and '$418$--$871$ IPDs' in DOC['manuscript']
  and 'detection may decrease' in DOC['manuscript']),
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
 ("largest finite-window certificate is below upward display 9.662e-10",
  max(r['P_FA_cert'] for r in rows) <= 9.662e-10
  and max(r['P_FA_cert'] for r in rows) > 9.661e-10
  and r'9.662\times10^{-10}' in DOC['manuscript']
  and r'9.662\times10^{-10}' in (DOC["response"] or '')),
 ("every row meets P_D >= 0.99 and 2R <= L_c and q >= 10 p_W",
  all(r["P_D_scan"] >= 0.99 and r["disjoint_scan"] and r["sep"] >= 10 for r in rows)),
]
# Check only affected fields: all other original design/performance values must remain.
affected = {'P_FA_cert', 'M_eff', 'FA_formula'}
CHECKS += [
 ('all seven original parameter/performance rows retained',
  len(rows) == 7 and all(
   {k: v for k, v in r.items() if k not in affected}
   == {k: v for k, v in ORIGINAL['%g' % r['lam']].items() if k not in affected}
   for r in rows)),
 ('certificate floats round upward from recorded exact rational values',
  all(Fraction.from_float(T['%g' % c['lam']]['P_FA_cert'])
      >= Fraction(int(c['exact_numerator']), int(c['exact_denominator']))
      and T['%g' % c['lam']]['P_FA_cert'] == c['corrected']
      for c in RECHECK['changes']['main'])),
 ('response table displays all seven bounds rounded upward',
  DOC["response"] is not None and all((r'%g & %d & %d & %d & %d & %d & $%.3f\times10^{-10}$'
       % (r['lam']*100, r['L_h'], r['tau'], r['R'], r['N_scr'], r['k_scr'],
          math.ceil(r['P_FA_cert']*1e13)/1000)) in DOC['response'] for r in rows)),
 ('all eight original feasible asymmetric designs meet the corrected budget',
  len([r for r in L('design_v2_asym.json') if r.get('feasible')]) == 8
  and all(r['P_FA_cert'] <= 1e-9
          for r in L('design_v2_asym.json') if r.get('feasible'))),
 ('all original first-stage filters retained; only first budget label changes',
  all(all(a[k] == b[k] for k in ['L1', 'N1', 'k1', 'P_D1'])
      for a, b in zip(J_ORIGINAL, J))
  and [(a['lam'], a['alpha1'], b['alpha1']) for a, b in zip(J_ORIGINAL, J)
       if a['alpha1'] != b['alpha1']] == [(0.002, 0.01, 0.03)]),
 ('two-stage rejection values are upper bounds in received IPDs',
  all(math.isclose(x['E_reject_ipds_upper'],
                   x['L1'] + x['P_FA1']*(x['L_obs']-x['L1']), rel_tol=1e-14)
      and 'E_reject_packets' not in x and 'received IPDs' in x['length_unit']
      for x in J)),
 ('54 exact identity checks and 3 exhaustive cases recorded',
  RECHECK['checks']['exact_identity_checks'] == 54
  and len(RECHECK['checks']['exhaustive_cases']) == 3
  and all(Fraction(x['actual']) <= Fraction(x['bound'])
          for x in RECHECK['checks']['exhaustive_cases'])),
 ('seven original nominal detection estimates reproduced',
  len(RECHECK['nominal_detection_reproduced']) == 7
  and all(x['unchanged'] and x['P_D_scan'] == T['%g' % x['lam']]['P_D_scan']
          for x in RECHECK['nominal_detection_reproduced'])),
 ('0.18 real-traffic ratio is limited to the 10/10 same-sample diagnostic',
  r'$(\Lh,\tau)=(10,10)$' in DOC['manuscript']
  and 'on the same sample' in DOC['manuscript']),
 ('reviewer replies cover all three comments at manuscript section locations',
  DOC["response"] is not None and
  all('Reviewer 2, Comment %d' % i in DOC['response'] for i in [1, 2, 3])
  and all(x in DOC['response'] for x in ['Section~IV-A', 'Eq.~(4)',
                                       'Section~V-B', 'abstract and conclusion'])),
 ('abstract and conclusion both state the independent-background scope',
  all('independent background bits' in text and 'dependent backgrounds' in text
      for text in [DOC['manuscript'].split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0],
                   DOC['manuscript'].split(r'\section{Conclusion}')[1]])),
]
bad = skipped = 0
for name, cond in CHECKS:
    if DOC["response"] is None and name in RESPONSE_CHECKS:
        print("  SKIP " + name + "  (response letter not in this copy)")
        skipped += 1
        continue
    print(("  OK   " if cond else "  FAIL ") + name)
    bad += (not cond)
print("\n%d/%d checks pass%s" % (len(CHECKS) - bad - skipped, len(CHECKS) - skipped,
                                 "" if not skipped else " (%d skipped)" % skipped))
sys.exit(1 if bad else 0)
