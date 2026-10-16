"""
Probe B (resubmission, answers R1-C2 / R2-C1):
Gilbert-Elliott BURSTY insertion/deletion channel at matched mean corruption rate lambda.
Take the Table-I parameters designed under the iid model UNCHANGED and measure the damage.

GE model: two states G (no edit) and B (edit w.p. lam_B per position).
  stationary  pi_B = lam / lam_B ;  mean bad-run length E_B => p_BG = 1/E_B,
  p_GB = p_BG * pi_B / (1 - pi_B).      E_B = 1 with lam_B = lam  <=>  iid baseline.
Edit type: deletion (fuse 2 IPDs -> 1 random symbol) or insertion (split 1 -> 2 random) w.p. 1/2.

Reported per (config, E_B):  q_emp (per-slot re-anchor hit), cumulative lock-loss over the
decode window, mean aligned-frame yield, drift variance per frame, and P(|drift|>R).
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)

# lambda(%), L_h, tau, R, N_max  -- Table I of the submitted letter
ROWS = [(0.005, 10, 9, 3, 100),
        (0.01, 10, 9, 3, 100),
        (0.02, 11, 10, 4, 95),
        (0.03, 12, 11, 5, 90),
        (0.05, 11, 11, 6, 95)]
L_C = 10
EPS = 0.05
BURSTS = [1, 2, 5, 10, 20]
DEAD_RUN = 8       # >=8 consecutive missed frames == track effectively dead
LAM_B = 0.5          # per-position edit probability inside a burst
N_FLOWS = 400


def ge_edit_mask(n, lam, E_B, lam_B, rng):
    """Markov-modulated edit indicator + edit type. E_B=1 & lam_B=lam -> iid."""
    if E_B <= 1:
        edit = rng.random(n) < lam
    else:
        pi_B = lam / lam_B
        if pi_B >= 0.95:
            raise ValueError("lam too large for lam_B")
        p_BG = 1.0 / E_B
        p_GB = p_BG * pi_B / (1.0 - pi_B)
        u = rng.random(n)
        state = np.empty(n, dtype=bool)
        s = rng.random() < pi_B
        for i in range(n):
            state[i] = s
            s = (u[i] >= p_BG) if s else (u[i] < p_GB)
        edit = state & (rng.random(n) < lam_B)
    is_ins = rng.random(n) < 0.5
    return edit, is_ins


def channel_ge(sent, lam, E_B, lam_B, eps, rng, want_map=False):
    sent = np.asarray(sent, dtype=np.int8)
    n = int(sent.size)
    edit, is_ins = ge_edit_mask(n, lam, E_B, lam_B, rng)
    flip = (rng.random(n) < eps).astype(np.int8)
    rpool = rng.integers(0, 2, size=2 * n).astype(np.int8)
    recv = np.empty(2 * n, dtype=np.int8)
    s2r = np.full(n, -1, dtype=np.int64)
    ri = rp = i = 0
    while i < n:
        if edit[i] and not is_ins[i] and i + 1 < n:      # deletion: fuse 2 -> 1
            s2r[i] = ri; recv[ri] = rpool[rp]; ri += 1; rp += 1; i += 2
        elif edit[i] and is_ins[i]:                       # insertion: split 1 -> 2
            s2r[i] = ri; recv[ri] = rpool[rp]; recv[ri + 1] = rpool[rp + 1]
            ri += 2; rp += 2; i += 1
        else:
            s2r[i] = ri; recv[ri] = sent[i] ^ flip[i]; ri += 1; i += 1
    recv = recv[:ri]
    return (recv, s2r) if want_map else recv


def run_cfg(lam, L_h, tau, R, N_max, E_B, rng, n_flows=N_FLOWS):
    """Metrics, with the frame-index SLIP separated from the per-frame drift.

    slip:  the detector re-locks on a *neighbouring* frame's header (anchor error
           = a nonzero multiple of T_f).  Periodic lock survives and the frame is
           counted as 'aligned', but the frame INDEX is wrong -- so aligned-frame
           yield alone does not certify end-to-end codeword decoding.
    fresh drift: recorded only when the previous anchor sat exactly on the previous
           header, i.e. one clean frame of random walk (theory var = lam*T_f).
    """
    T_f = L_h + L_C
    s = rng.integers(0, 2, size=L_h).astype(np.int8)
    hits = slots = 0
    n_hit = n_slip = n_offbyk = 0
    yields, lost, drifts, runs_max, aligned_true = [], 0, [], [], []
    for _ in range(n_flows):
        sent = np.empty(N_max * T_f, dtype=np.int8)
        for f in range(N_max):
            sent[f * T_f:f * T_f + L_h] = s
            sent[f * T_f + L_h:(f + 1) * T_f] = rng.integers(0, 2, size=L_C)
        recv, s2r = channel_ge(sent, lam, E_B, LAM_B, EPS, rng, want_map=True)
        tp = s2r[np.arange(N_max) * T_f]           # true recv position of each header
        a0 = int(tp[0]) if tp[0] >= 0 else 0
        anchor = a0
        anchor_is_true = (tp[0] >= 0)
        prev_idx = 0
        y = y_true = 0
        run, run_max = 0, 0
        for f in range(1, N_max):
            predicted = anchor + T_f
            if tp[f] >= 0 and anchor_is_true and prev_idx == f - 1:
                drifts.append(int(tp[f] - predicted))
            best_a, best_p = -1, predicted
            for off in range(-R, R + 1):
                st = predicted + off
                if st < 0 or st + L_h > recv.size:
                    continue
                a = int(np.sum(recv[st:st + L_h] == s))
                if a > best_a:
                    best_a, best_p = a, st
            hit = best_a >= tau
            hits += hit; slots += 1; y += hit
            if hit:
                n_hit += 1
                on_true = (tp[f] >= 0 and best_p == tp[f])
                y_true += on_true
                if not on_true:
                    n_slip += 1
                    # is the anchor sitting on SOME frame's header?
                    d_ok = np.flatnonzero((tp >= 0) & (tp == best_p))
                    if d_ok.size:
                        n_offbyk += 1
                        prev_idx = int(d_ok[0])
                    else:
                        prev_idx = -99
                    anchor_is_true = bool(d_ok.size)
                else:
                    prev_idx = f
                    anchor_is_true = True
            anchor = best_p if hit else predicted
            if not hit:
                anchor_is_true = False
            run = 0 if hit else run + 1
            run_max = max(run_max, run)
        yields.append(y)
        aligned_true.append(y_true)
        runs_max.append(run_max)
        lost += (run_max >= DEAD_RUN)
    d = np.array(drifts, dtype=np.float64)
    return {"q_emp": hits / max(slots, 1),
            "miss_run_max_mean": float(np.mean(runs_max)),
            "miss_run_max_p95": float(np.percentile(runs_max, 95)),
            "yield_mean": float(np.mean(yields)), "yield_min": int(np.min(yields)),
            "yield_p05": float(np.percentile(yields, 5)),
            "yield_true_index_mean": float(np.mean(aligned_true)),
            "slip_frac_of_hits": n_slip / max(n_hit, 1),
            "slip_onto_other_header_frac": n_offbyk / max(n_slip, 1),
            "track_dead_frac": lost / n_flows,
            "drift_var_fresh": float(d.var()) if d.size else None,
            "drift_mean_fresh": float(d.mean()) if d.size else None,
            "n_fresh": int(d.size),
            "P_out_emp": float((np.abs(d) > R).mean()) if d.size else None}


def main():
    rng = np.random.default_rng(20260830)
    out = {"model": "Gilbert-Elliott bursty ins/del, matched mean lambda",
           "lam_B": LAM_B, "eps": EPS, "L_c": L_C, "n_flows": N_FLOWS, "rows": []}
    for (lam, L_h, tau, R, N_max) in ROWS:
        for E_B in BURSTS:
            if lam / LAM_B >= 0.95:
                continue
            r = run_cfg(lam, L_h, tau, R, N_max, E_B, rng)
            r.update({"lam": lam, "L_h": L_h, "tau": tau, "R": R,
                      "N_max": N_max, "E_B": E_B,
                      "drift_var_iid_theory": lam * (L_h + L_C)})
            out["rows"].append(r)
            print("lam=%.3f L_h=%d R=%d E_B=%-3d q=%.3f yield=%.1f (min %d, p05 %.0f) "
                  "dead=%.3f dvar=%.3f(th %.3f) Pout=%.4f slip=%.4f Ytrue=%.1f"
                  % (lam, L_h, R, E_B, r["q_emp"], r["yield_mean"], r["yield_min"],
                     r["yield_p05"], r["track_dead_frac"], r["drift_var_fresh"] or -1,
                     r["drift_var_iid_theory"], r["P_out_emp"] or -1,
                     r["slip_frac_of_hits"], r["yield_true_index_mean"]), flush=True)
    with open(os.path.join(OUT, "probeB_burst_channel.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("WROTE probeB_burst_channel.json", flush=True)


if __name__ == "__main__":
    main()
