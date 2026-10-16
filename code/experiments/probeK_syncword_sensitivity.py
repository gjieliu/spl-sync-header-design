"""Probe K: why the design needs a separation margin  q >= sep_min * p_W.

Without it the search accepts sync words so short that p_W ~ 0.1.  A chance full match
inside the codeword region then falls within the +-R window and STEALS the anchor, so the
H1-side yield becomes strongly dependent on the particular sync word -- an effect the
per-slot model does not contain.  Here the decodable yield is measured over ten randomly
drawn BALANCED sync words, for a thin-separation configuration and for the corresponding
configuration under the separation margin.

Note the contrast with Probe A6/A7: on the H0 side, sidelobe MAGNITUDE has no predictive
power (the signed spectral alignment does).  On the H1 side it is magnitude that matters,
because a large sidelobe makes a shifted self-match inside the scan window more likely.
"""
import os, sys, json, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from probeB_burst_channel import channel_ge
from probeD2_recapture import profile

RES = os.path.join(HERE, "..", "results")
EPS, L_C, N_FLOWS, N_WORDS = 0.05, 10, 150, 10
# (lam, L_h, tau, R, N_max, muY, label): thin separation vs the margined design
CASES = [(0.005, 6, 6, 3, 125, 75.8, "thin (q/p_W = 6.5, sep_min off)"),
         (0.005, 10, 9, 3, 100, 73.9, "margined (q/p_W = 11.6, sep_min = 10)"),
         (0.02, 10, 9, 4, 100, 63.5, "thin (q/p_W = 7.7, sep_min off)"),
         (0.02, 8, 8, 4, 111, 53.3, "margined (q/p_W = 16.1, sep_min = 10)")]


def sidelobe(g):
    L = g.size
    return max(abs(int(np.dot(g[:L - k], g[k:]))) for k in range(1, L))


def yield_for(lam, L_h, tau, R, N_max, s, rng, nf=N_FLOWS):
    T_f = L_h + L_C
    hits = slots = 0
    Ye = []
    for _ in range(nf):
        sent = np.empty(N_max * T_f, dtype=np.int8)
        for f in range(N_max):
            sent[f * T_f:f * T_f + L_h] = s
            sent[f * T_f + L_h:(f + 1) * T_f] = rng.integers(0, 2, size=L_C)
        recv, s2r = channel_ge(sent, lam, 1, 0.5, EPS, rng, want_map=True)
        prof = profile(recv, s)
        tp = s2r[np.arange(N_max) * T_f]
        anchor = int(tp[0]) if tp[0] >= 0 else 0
        ye = 0
        P = prof.size
        for f in range(1, N_max):
            pred = anchor + T_f
            lo, hi = max(0, pred - R), min(P, pred + R + 1)
            if lo >= hi:
                anchor = pred; continue
            seg = prof[lo:hi]
            j = int(np.argmax(seg)); best, bp = int(seg[j]), lo + j
            slots += 1
            if best >= tau:
                hits += 1
                if tp[f] >= 0 and bp == tp[f]:
                    ye += 1
                anchor = bp
            else:
                anchor = pred
        Ye.append(ye)
    return hits / max(slots, 1), float(np.mean(Ye))


def main():
    out = []
    for (lam, L_h, tau, R, N_max, muY, label) in CASES:
        rng = np.random.default_rng(99)
        bal = [np.array(h, dtype=np.int8)
               for h in itertools.product([0, 1], repeat=L_h) if sum(h) == L_h // 2]
        pick = [bal[i] for i in rng.choice(len(bal), size=min(N_WORDS, len(bal)),
                                           replace=False)]
        rows = []
        for s in pick:
            q, ye = yield_for(lam, L_h, tau, R, N_max, s, rng)
            rows.append({"sidelobe": sidelobe(2 * s.astype(np.int16) - 1),
                         "q_emp": q, "Y_exact": ye,
                         "word": "".join(map(str, s.tolist()))})
        rows.sort(key=lambda r: r["sidelobe"])
        ys = np.array([r["Y_exact"] for r in rows])
        below = int((ys < muY).sum())
        out.append({"lam": lam, "L_h": L_h, "tau": tau, "R": R, "N_max": N_max,
                    "muY": muY, "label": label, "words": rows,
                    "Y_exact_min": float(ys.min()), "Y_exact_max": float(ys.max()),
                    "spread_pct": float(100 * (ys.max() - ys.min()) / ys.mean()),
                    "n_words_below_muY": below})
        print("lam=%.3f  %d/%d/R%d  muY=%.1f  %s" % (lam, L_h, tau, R, muY, label))
        print("   Y_exact over %d balanced sync words: %.1f .. %.1f  (spread %.0f%%), "
              "%d below muY" % (len(rows), ys.min(), ys.max(),
                                100 * (ys.max() - ys.min()) / ys.mean(), below))
        for r in rows:
            print("     max|c_k|=%d  q=%.3f  Y_exact=%5.1f  %s%s"
                  % (r["sidelobe"], r["q_emp"], r["Y_exact"], r["word"],
                     "   <-- at or below muY" if r["Y_exact"] < muY else ""))
    json.dump(out, open(os.path.join(RES, "probeK_syncword_sensitivity.json"), "w"), indent=1)
    print("\nWROTE probeK_syncword_sensitivity.json")


if __name__ == "__main__":
    main()
