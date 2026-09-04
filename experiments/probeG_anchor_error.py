"""Probe G: WHERE a hit actually lands.  Splits the anchor error (best_pos - true_pos)
of every declared hit into
    0            exactly on the true header  -> the frame is decodable at its true index
    0<|e|<=R     a symbol-offset anchor      -> counted as "aligned" but read shifted
    |e| ~ m*T_f  a FRAME-INDEX SLIP          -> periodic lock kept, frame number wrong
This is the empirical content of R2-C6: aligned-frame yield over-counts decodable frames.
"""
import os, sys, json, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from probeB_burst_channel import channel_ge

RES = os.path.join(HERE, "..", "results")
EPS = 0.05
L_C = 10
N_FLOWS = 300
# (lam, L_h, tau, R, N_max): rows of the SUBMITTED table
CASES = [(0.005, 10, 9, 3, 100), (0.01, 10, 9, 3, 100), (0.02, 11, 10, 4, 95),
         (0.03, 12, 11, 5, 90), (0.05, 11, 11, 6, 95)]


def main():
    rng = np.random.default_rng(13)
    out = []
    for (lam, L_h, tau, R, N_max) in CASES:
        T_f = L_h + L_C
        s = rng.integers(0, 2, size=L_h).astype(np.int8)
        errs = []
        for _ in range(N_FLOWS):
            sent = np.empty(N_max * T_f, dtype=np.int8)
            for f in range(N_max):
                sent[f * T_f:f * T_f + L_h] = s
                sent[f * T_f + L_h:(f + 1) * T_f] = rng.integers(0, 2, size=L_C)
            recv, s2r = channel_ge(sent, lam, 1, 0.5, EPS, rng, want_map=True)
            tp = s2r[np.arange(N_max) * T_f]
            anchor = int(tp[0]) if tp[0] >= 0 else 0
            for f in range(1, N_max):
                pred = anchor + T_f
                ba, bp = -1, pred
                for o in range(-R, R + 1):
                    st = pred + o
                    if st < 0 or st + L_h > recv.size:
                        continue
                    a = int(np.sum(recv[st:st + L_h] == s))
                    if a > ba:
                        ba, bp = a, st
                hit = ba >= tau
                if hit and tp[f] >= 0:
                    errs.append(int(bp - tp[f]))
                anchor = bp if hit else pred
        e = np.array(errs)
        exact = float((e == 0).mean())
        off = float(((e != 0) & (np.abs(e) <= R)).mean())
        slip = float((np.abs(e) > R).mean())
        top = sorted(collections.Counter(e.tolist()).items(), key=lambda kv: -kv[1])[:8]
        out.append({"lam": lam, "L_h": L_h, "tau": tau, "R": R, "T_f": T_f,
                    "n_hits": int(e.size), "P_exact": exact,
                    "P_symbol_offset": off, "P_frame_slip": slip,
                    "top_errors": [[int(k), int(v)] for k, v in top]})
        print("lam=%.3f %2d/%2d R=%d  hits=%6d | exact=%.4f  0<|e|<=R=%.4f  |e|>R=%.4f"
              % (lam, L_h, tau, R, e.size, exact, off, slip))
        print("    most common errors (T_f=%d):" % T_f,
              ", ".join("%+d:%d" % (k, v) for k, v in top))
    json.dump(out, open(os.path.join(RES, "probeG_anchor_error.json"), "w"), indent=1)
    print("\nWROTE probeG_anchor_error.json")


if __name__ == "__main__":
    main()
