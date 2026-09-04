# Sync-header design for IPD-QIM timing watermarks under insertion–deletion channels

Code and results for the paper

> *Detection-Theoretic Sync-Header Design for IPD-QIM Timing Watermarks under Insertion–Deletion Channels* (IEEE Signal Processing Letters, submission SPL-48099-2026, revised version).

Every number in the manuscript and in the response to reviewers is produced by a script in this folder, written to `results/*.json`, and re-read by `report/verify_claims.py`, which checks the text against the JSONs. Nothing is transcribed by hand.

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

The manuscript sources live in the sibling folder `../manuscript/` (`report/make_tables.py` writes `tab_main.tex` there and `report/make_figs.py` writes `figs/fig_balance_ladder.pdf`); the point-by-point reply lives in `../response/`.

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
python run_all.py all           # everything, in dependency order (long; see below)
python run_all.py tables figs verify     # regenerate the table and figure, check the text
python run_all.py design        # only the design search (resumable, see below)
```

Each experiment can also be run directly, e.g. `python experiments/probeC_ladder.py`; scripts find `design_v2.py` and `results/` relative to their own location, so the working directory does not matter.

Seeds are fixed in every script, so re-running reproduces the committed JSONs exactly on the same NumPy version; Monte-Carlo figures in the paper are quoted with their sampling error where it matters.

**Run time.** Most probes take seconds to a few minutes. The three long stages are the design search (`design`, minutes per λ; the driver skips any λ already present in `results/design_v2_table.json`, so it can be run λ by λ: `python experiments/run_design_v2.py 0.02 0.03`), the false-alarm ladder (`ladder`), and the paired re-capture study (`recap`).

## Stages

Each stage is one script in `experiments/` (or `report/`), writes one JSON of the same name into `results/`, and supports one place in the paper. "R1-C2" etc. name the reviewer comments the experiment answers; the mapping is spelled out in `../response/response.tex`.

| stage | script | what it establishes | where it is used |
|---|---|---|---|
| `window` | `probeH_window_semantics.py` | the submitted table was computed with an observation window of 2000 packets while the text said 500; under the text's semantics the design is infeasible | response, "A defect we found ourselves" |
| `pd` | `probeF_pd_aggregation.py` | the product form 1−(1−p_tr)^J over-counts P_D (Harris/FKG); the scan statistic is the correct quantity (0.992 claimed vs 0.960 at λ=5%) | Prop. 4, Sec. IV-B |
| `drift` | `probeI_drift_check.py` | per-frame drift variance is λT_f on the pure channel map | Sec. II-C |
| `anchor` | `probeG_anchor_error.py` | where a hit lands: exact / symbol offset / frame slip; gives the anchor-exactness ratio ρ = 0.85 | Table I caption, Sec. V-D |
| `trace0` | `probeA_real_trace_p0.py` | single-offset and windowed false-hit rates on real Tor/VPN IPDs vs the fair-bit value; the blow-up depends on the header | Sec. V-B |
| `balance` | `probeA2_header_balance.py` | false-hit rate stratified by header Hamming weight; only balanced headers stay near the bound | Prop. 2, Fig. 1(a) |
| `balenum` | `probeA3_balanced_enum.py` | worst case over the *entire* balanced class, not a sample | Sec. V-B |
| `v2cfg` | `probeA4_v2configs.py` | the balanced bound checked cell by cell on the configurations of Table I (144 cells, 7 above 1.5×) | Sec. V-B, margin f_fa |
| `precond` | `probeA5_precondition.py` | lag-one autocorrelation alone does not flag the failing cells | Sec. V-B |
| `sidelobe` | `probeA6_sidelobe.py` | **falsified hypothesis, kept as a record**: aperiodic sidelobe magnitude does not predict the false-hit rate (Spearman −0.12) | response, R3 |
| `align` | `probeA7_alignment.py` | the signed spectral alignment V(g) does predict it (Spearman +0.75 / +0.85) | Remark 1, Eq. (6) |
| `burst` | `probeB_burst_channel.py` | Gilbert–Elliott bursts at matched mean rate: per-slot hit rate up, blackout fraction up | Sec. V-D, "Bursts" |
| `design` | `run_design_v2.py` | the constrained search of Eq. (10): the design half of Table I | Table I, Sec. V-A |
| `asym` | `run_design_v2_asym.py` | the same search under asymmetric insertion/deletion | Sec. V-D, "Asymmetry" |
| `ladder` | `probeC_ladder.py` | the certified false-alarm bound against measurement as k is weakened, over about five decades | Prop. 3, Sec. V-C, Fig. 1(b) |
| `e2e` | `probeD_endtoend.py` | end-to-end channel Monte-Carlo: delivered frames Y_hit and decodable frames Y_exact | Table I (validation half), Sec. V-D |
| `recap` | `probeD2_recapture.py` | paired (common-random-numbers) comparison of re-capture rules, 800 flows | Sec. V-D, "Re-capture" |
| `jitter` | `probeE_heavytail_asym.py` | per-symbol error ε under Gaussian, Laplace, Student-t, log-normal and Pareto jitter; scan radius under asymmetry | Eq. (1), Sec. V-D, "Asymmetry and jitter" |
| `syncsens` | `probeK_syncword_sensitivity.py` | why the search needs the separation margin q ≥ 10 p_W: without it chance codeword matches steal the anchor | Sec. III (Prop. 1), Sec. V-A |
| `burstv2` | `probeB2_burst_v2.py` | the burst study repeated on the configurations of Table I | Table I, last column |
| `twostage` | `probeJ_two_stage.py` | an honest two-stage screen: first-stage length, P_D1 ≥ 0.999, expected rejection latency | Sec. V-D, "Latency" |
| `tables` | `report/make_tables.py` | writes `../manuscript/tab_main.tex` from the JSONs | Table I |
| `figs` | `report/make_figs.py` | writes `../manuscript/figs/fig_balance_ladder.pdf` | Fig. 1 |
| `verify` | `report/verify_claims.py` | 20 checks that the numbers in `main_v2.tex`, `tab_main.tex` and `response.tex` match `results/` | — |

`results/legacy/` holds the output JSONs of the pipeline behind the *submitted* version (the code is in `../../SPL Submission V1/code/`). Only `window` and `pd` read them, to show what the submitted design actually computed.

## The library: `design_v2.py`

| function | role |
|---|---|
| `binom_sf_ge(n, p, k)` | exact binomial upper tail, no SciPy |
| `p0_balanced_sup(L_h, tau)` | sup over the background marginal π of the single-offset false-hit rate of a balanced header (Prop. 2; attained at π = ½ for even L_h) |
| `p_fa_certified(p0, W, N, k, M, disjoint=True)` | certified false-alarm bound M·p₀·P[Bin(N−1, Wp₀) ≥ k−1] under disjoint scans 2R ≤ L_c (Prop. 3); odd/even decimation fallback otherwise |
| `p_fa_product`, `p_d_product` | the product forms used by the submitted version, kept for comparison (the P_D one is an upper bound, Prop. 4) |
| `p_d_scan(q_cap, q, N, k, J, N_max)` | P_D as a scan statistic on the per-slot Bernoulli model, capture rate q_cap and re-anchoring rate q distinguished |
| `drift_pmf`, `size_R_cumulative` | per-frame drift distribution and the scan radius R that keeps the lock-loss probability below η_lock over the window |
| `search(lam, ...)` | the constrained design of Eq. (10): even L_h, balanced header, disjoint scans, certified P_FA ≤ α with margin f_fa, scan-statistic P_D ≥ 1−δ with margin m_q, separation q ≥ 10·p_W; returns the argmax of the decodable-yield objective |

The module docstring records the three changes relative to the submitted design (certified false alarm, over-counted detection, explicit observation window).

## Regenerating the paper's numbers after a change

1. Re-run the affected stage(s): `python run_all.py <stage> ...`
2. `python run_all.py tables figs` to rewrite the table and the figure.
3. `python run_all.py verify` — it must print `20/20 checks pass`; every failing line names the number and the file that disagree.
