# Sync-header design for IPD-QIM timing watermarks under insertion–deletion channels

Code and results for the paper

> *Detection-Theoretic Sync-Header Design for IPD-QIM Timing Watermarks under Insertion–Deletion Channels* (IEEE Signal Processing Letters).

The implementation reproduces the paper's designs end to end: `experiments/recheck_local.py` recomputes the certified finite-window false-alarm bounds and the two-stage latency bounds from frozen reference inputs with exact rational arithmetic, reproduces the seven nominal detection estimates, and records consistency checks in `results/recheck_local.json`. The analytical certificate assumes independent background bits with a common marginal; real-traffic measurements remain empirical.

`report/verify_claims.py` checks the reported values against the result files. It does not establish the probability theorem or verify every sentence automatically; the full argument is in the manuscript (Proposition 3).

## Layout

```
code/
├── run_all.py            driver: lists the stages and runs any subset of them
├── design_v2.py          the library (bounds, scan-statistic P_D, drift, constrained search)
├── experiments/          one script per experiment; each writes results/<script>.json
├── report/               make_tables.py, make_figs.py, verify_claims.py
├── results/              committed outputs of every experiment (+ results/legacy/, see below)
├── requirements.txt
└── README.md
```

The manuscript sources live in the sibling folder `../manuscript/` (`report/make_tables.py` writes `tab_main.tex` there and `report/make_figs.py` writes `figs/fig_balance_ladder.pdf`).

## Requirements

Python ≥ 3.10 with `numpy` and `matplotlib` only (`pip install -r requirements.txt`). SciPy is deliberately not used; the binomial tails are computed in `design_v2.py`.

**Real-traffic traces.** The A-series experiments (`probeA*`) read pools of real inter-packet delays (Tor and VPN application classes from the ISCX Tor-nonTor and ISCX VPN-nonVPN datasets, references [15] and [16] of the paper). The pools are not part of this repository. Put them in a folder and point `NFW_TRACE_DIR` at it:

```
export NFW_TRACE_DIR=/path/to/real_traffic
```

Without them, every other stage still runs, and `report/verify_claims.py` still passes because the committed `results/probeA*.json` are used as-is. `../DATA.md` describes the pool format.

## Quick start

```
python run_all.py               # list the stages and what each one establishes
python run_all.py local         # fixed R1 recheck, two-stage update, table, figure, checks
python run_all.py tables figs verify     # regenerate the table and figure, check the text
python run_all.py design        # optional new design search; may replace retained rows
python run_all.py all           # full exploratory pipeline, not the frozen-R1 recheck
```

Each experiment can also be run directly, e.g. `python experiments/probeC_ladder.py`; scripts find `design_v2.py` and `results/` relative to their own location, so the working directory does not matter.

Seeds are fixed in every script, so re-running reproduces the committed JSONs exactly on the same NumPy version; Monte-Carlo figures in the paper are quoted with their sampling error where it matters.

**Run time.** Most probes take seconds to a few minutes. The three long stages are the design search (`design`, minutes per λ; the driver skips any λ already present in `results/design_v2_table.json`, so it can be run λ by λ: `python experiments/run_design_v2.py 0.02 0.03`), the false-alarm ladder (`ladder`), and the paired re-capture study (`recap`).

## Stages

Each stage runs a script in `experiments/` or `report/`. Some diagnostics record earlier development stages of the study (for example the falsified sidelobe hypothesis) and are kept as documentation. The `recheck` stage recomputes the main, asymmetric and two-stage result files from frozen reference snapshots and writes a comparison report. The `local` command avoids the optional design searches; feasibility of the retained configurations is rechecked, without claiming shortest-track optimality under the corrected count.

| stage | script | what it establishes | where it is used |
|---|---|---|---|
| `window` | `probeH_window_semantics.py` | the design table uses a 2000-IPD observation window; a 500-IPD window would make the design infeasible | Sec. V-A |
| `pd` | `probeF_pd_aggregation.py` | the product form 1−(1−p_tr)^J over-counts P_D (Harris/FKG); the scan statistic is the correct quantity (0.992 claimed vs 0.960 at λ=5%) | Prop. 4, Sec. IV-B |
| `drift` | `probeI_drift_check.py` | per-frame drift variance is λT_f on the pure channel map | Sec. II-C |
| `anchor` | `probeG_anchor_error.py` | where a hit lands: exact / symbol offset / frame slip; gives the anchor-exactness ratio ρ = 0.85 | Table I caption, Sec. V-D |
| `trace0` | `probeA_real_trace_p0.py` | single-offset and windowed false-hit rates on real Tor/VPN IPDs vs the fair-bit value; the blow-up depends on the header | Sec. V-B |
| `balance` | `probeA2_header_balance.py` | false-hit rate stratified by header Hamming weight; only balanced headers stay near the bound | Prop. 2, Fig. 1(a) |
| `balenum` | `probeA3_balanced_enum.py` | worst case over the *entire* balanced class, not a sample | Sec. V-B |
| `v2cfg` | `probeA4_v2configs.py` | the balanced bound checked cell by cell on the configurations of Table I (144 cells, 7 above 1.5×) | Sec. V-B, margin f_fa |
| `precond` | `probeA5_precondition.py` | lag-one autocorrelation alone does not flag the failing cells | Sec. V-B |
| `sidelobe` | `probeA6_sidelobe.py` | **falsified hypothesis, kept as a record**: aperiodic sidelobe magnitude does not predict the false-hit rate (Spearman −0.12) | Sec. III |
| `align` | `probeA7_alignment.py` | the covariance-weighted alignment V(g) correlates with the measured rates (Spearman +0.75 / +0.85) | Eq. (6), Sec. V-B |
| `burst` | `probeB_burst_channel.py` | Gilbert–Elliott bursts at matched mean rate: per-slot hit rate up, blackout fraction up | Sec. V-D, "Bursts" |
| `design` | `run_design_v2.py` | the constrained search of Eq. (10): the design half of Table I | Table I, Sec. V-A |
| `asym` | `run_design_v2_asym.py` | the same search under asymmetric insertion/deletion | Sec. V-D, "Asymmetry" |
| `ladder` | `probeC_ladder.py` | single-track binomial tail versus iid measurements at accessible rates; does not certify independence or 10^-9 | Prop. 3, Sec. V-C, Fig. 1(b) |
| `e2e` | `probeD_endtoend.py` | channel Monte-Carlo from the mapped first-frame anchor: tracking yields Y_hit and Y_exact | Table I (validation half), Sec. V-D |
| `recap` | `probeD2_recapture.py` | paired (common-random-numbers) comparison of re-capture rules, 800 flows | Sec. V-D, "Re-capture" |
| `jitter` | `probeE_heavytail_asym.py` | per-symbol error ε under Gaussian, Laplace, Student-t, log-normal and Pareto jitter; scan radius under asymmetry | Eq. (1), Sec. V-D, "Asymmetry and jitter" |
| `syncsens` | `probeK_syncword_sensitivity.py` | why the search needs the separation margin q ≥ 10 p_W: without it chance codeword matches steal the anchor | Sec. III (Prop. 1), Sec. V-A |
| `burstv2` | `probeB2_burst_v2.py` | the burst study repeated on the configurations of Table I | Table I, last column |
| `recheck` | `recheck_local.py` | exact finite-window recomputation of the design rows, nominal detection reproduction, small exhaustive checks | Prop. 3, Table I |
| `twostage` | `probeJ_two_stage.py` | default: fixed first-stage screens, iid false-alarm and rejection-latency upper bounds in received IPDs; `--search` requests a new search | Sec. V-D, "Latency" |
| `tables` | `report/make_tables.py` | writes `../manuscript/tab_main.tex` from the JSONs | Table I |
| `figs` | `report/make_figs.py` | writes `../manuscript/figs/fig_balance_ladder.pdf` | Fig. 1 |
| `verify` | `report/verify_claims.py` | numerical, scope and response checks against `results/`; prints the actual number of checks | — |

`results/legacy/` holds the output JSONs of the pipeline behind an earlier version of the design; only `window` and `pd` read them, to document what that earlier pipeline computed.

## The library: `design_v2.py`

| function | role |
|---|---|
| `binom_sf_ge(n, p, k)` | exact binomial upper tail, no SciPy |
| `p0_balanced_sup(L_h, tau)` | sup over the background marginal π of the single-offset false-hit rate of a balanced header (Prop. 2; attained at π = ½ for even L_h) |
| `p_fa_certified(p0, W, N, k, L_obs, L_h, T_f)` | iid finite-window bound p₀ Σ_t w_t C(t−1,k−2) u^(k−1)(1−u)^(t−k+1), rounded upward from exact rational arithmetic; u=min(1,Wp₀), w_t=[L_obs−L_h−tT_f+(k−1)R+1]_+; requires 2R≤L_c, with no unsupported fallback |
| `p_fa_product`, `p_d_product` | the product forms used by the submitted version, kept for comparison (the P_D one is an upper bound, Prop. 4) |
| `p_d_scan(q_cap, q, N, k, J, N_max)` | nominal scan probability on independent H_i~Bern(q) and cap_i~Bern(q_cap) coordinates, shared across tracks; unchanged sampling, not a pathwise edit-detector guarantee |
| `drift_pmf`, `size_R_cumulative` | per-frame drift distribution and the scan radius R that keeps the lock-loss probability below η_lock over the window |
| `search(lam, ...)` | optional Eq. (10) search: even balanced headers, 2R≤L_c, corrected iid P_FA≤α with f_fa, nominal scan estimate ≥1−δ with m_q and separation q≥10p_W; the local revision rechecks fixed R1 configurations instead of rerunning this search |

The module docstring records the three changes relative to the earlier design (certified false alarm, over-counted detection, explicit observation window).

## Regenerating the paper's numbers after a change

1. Re-run the affected stage(s): `python run_all.py <stage> ...`
2. `python run_all.py tables figs` to rewrite the table and the figure.
3. `python run_all.py verify` must exit successfully; every failing line identifies a disagreement. The `local` command performs the recomputation and report regeneration in order. The first-stage screens are fixed; the lowest-rate screen is reported against the $0.03$ budget. Rejection latency is an iid upper bound in received IPDs, with upward-rounded range 418–871; detection can decrease after screening.
