"""Reproduction driver for the SPL resubmission package.

    python3 run_all.py                 # list the stages
    python3 run_all.py <stage> [...]   # run selected stages
    python3 run_all.py all             # run everything (long; see the note below)

NOTE ON RUNTIME.  The design search (stage `design`) evaluates a scan statistic by
simulation for every candidate, so it takes a few minutes per lambda.  It is driven by
run_design_v2.py, which is RESUMABLE: results already present in
results/design_v2_table.json (inside code/) are skipped, so it can be invoked repeatedly with a subset
of lambdas.  Stages that need the real-trace IPD pools read them from $NFW_TRACE_DIR
(default: ../../../../NFW2_Simulation/results/real_traffic, relative to code/experiments/); see ../DATA.md.

Dependencies: numpy only (no scipy; all binomial tails are exact sums).
"""
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))

STAGES = [
    # name            script                          what it establishes
    ("window",   "probeH_window_semantics.py",   "submitted table uses L=2000, not the text's L_rej=500"),
    ("pd",       "probeF_pd_aggregation.py",     "product-form P_D is optimistic (FKG); scan statistic is correct"),
    ("drift",    "probeI_drift_check.py",        "per-frame drift model Var = lambda*T_f is correct"),
    ("anchor",   "probeG_anchor_error.py",       "hits split into exact / symbol-offset / frame-index slip"),
    ("trace0",   "probeA_real_trace_p0.py",      "real-trace p0/pW vs theory, header-dependent blow-up"),
    ("balance",  "probeA2_header_balance.py",    "p0 stratified by header Hamming weight"),
    ("balenum",  "probeA3_balanced_enum.py",     "worst case over the FULL balanced-header class"),
    ("v2cfg",    "probeA4_v2configs.py",         "balanced bound checked on the v2 configurations"),
    ("precond",  "probeA5_precondition.py",      "autocorrelation alone does not flag the failing cell"),
    ("sidelobe", "probeA6_sidelobe.py",          "FALSIFIED: max|c_k| does not predict p0"),
    ("align",    "probeA7_alignment.py",         "signed spectral alignment V(g) does predict p0"),
    ("burst",    "probeB_burst_channel.py",      "Gilbert-Elliott bursts: yield up, blackouts up"),
    ("design",   "run_design_v2.py",             "the v2 design table (resumable)"),
    ("asym",     "run_design_v2_asym.py",        "robustness to asymmetric insertion/deletion"),
    ("ladder",   "probeC_ladder.py",             "certified FA bound validated over ~5 decades"),
    ("e2e",      "probeD_endtoend.py",           "decodable yield Y_exact vs reported Y_hit"),
    ("recap",    "probeD2_recapture.py",         "paired comparison of re-capture rules"),
    ("jitter",   "probeE_heavytail_asym.py",     "eps under heavy-tailed jitter; R under asymmetry"),
    ("syncsens", "probeK_syncword_sensitivity.py", "why the design needs the separation margin q >= 10 p_W"),
    ("burstv2",  "probeB2_burst_v2.py",          "bursts on the v2 configurations (decodable yield)"),
    ("tables",   "make_tables.py",               "emit the manuscript table from the JSONs"),
    ("figs",     "make_figs.py",                 "emit the manuscript figure"),
    ("verify",   "verify_claims.py",             "cross-check every number in the manuscript and response against the JSONs"),
    ("twostage", "probeJ_two_stage.py",          "honest fast-reject: two-stage screen, E[reject] ~ 830 packets"),
]


def main(argv):
    names = [s[0] for s in STAGES]
    if not argv:
        print(__doc__)
        print("stages:")
        for n, f, w in STAGES:
            print("  %-9s %-32s %s" % (n, f, w))
        return 0
    todo = names if argv == ["all"] else argv
    bad = [t for t in todo if t not in names]
    if bad:
        print("unknown stage(s):", ", ".join(bad)); return 2
    for n, f, w in STAGES:
        if n not in todo:
            continue
        print("\n" + "=" * 78 + "\n== %-8s %s\n== %s\n" % (n, f, w) + "=" * 78, flush=True)
        sub = "report" if f.startswith(("make_", "verify_")) else "experiments"
        rc = subprocess.call([sys.executable, "-u", os.path.join(HERE, sub, f)], cwd=os.path.join(HERE, sub))
        if rc != 0:
            print("!! stage %s exited %d" % (n, rc)); return rc
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
