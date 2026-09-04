"""Manuscript figure: (a) the balance result on real traffic, (b) the certified bound
validated over a decade ladder.  Kept deliberately small (single column, ~1.45 in) so
that the technical content fits SPL's four-page limit; legibility is protected by using
direct labels instead of dense legends."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "..", "manuscript", "figs")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 6, "axes.linewidth": 0.5, "font.family": "serif",
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 1.8, "ytick.major.size": 1.8,
                     "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
                     "legend.frameon": False})

A = json.load(open(os.path.join(RES, "probeA2_header_balance.json")))
C = json.load(open(os.path.join(RES, "probeC_ladder.json")))
KEY = "10/9/W7"

fig, ax = plt.subplots(1, 2, figsize=(3.45, 1.45))

# ---------- (a) worst-case false hit vs sync-header Hamming weight ----------
series = [("iscx_tor_ipd", "19", "o-", "Tor"),
          ("iscx_tor_video_ipd", "19", "s-", "Tor video"),
          ("iscx_vpn_ipd", "1", "^-", "VPN")]
p0t = None
for pool, d, mk, lab in series:
    cell = A["pools"][pool][d]["cfg"][KEY]
    p0t = cell["p0_theory"]
    bw = cell["by_weight"]
    ws = sorted(int(w) for w in bw)
    ys = [max(bw[str(w)]["p0_max"], 3e-5) for w in ws]
    ax[0].semilogy(ws, ys, mk, ms=2.2, lw=0.8)
    ax[0].annotate(lab, xy=(ws[-1], ys[-1]), xytext=(-1, 3 if lab == "VPN" else -7),
                   textcoords="offset points", ha="right", fontsize=5.4)
ax[0].axhline(p0t, color="k", ls="--", lw=0.8)
ax[0].text(-0.3, p0t * 0.42, "fair-bit bound", ha="left", fontsize=5.4)
ax[0].axvline(5, color="0.6", ls=":", lw=0.8)
ax[0].text(5.25, 1.1e-4, "balanced", fontsize=5.4, color="0.35")
ax[0].set_xlabel("Hamming weight $w$", labelpad=0.5)
ax[0].set_ylabel("worst-case $p_0$", labelpad=0.5)
ax[0].set_xticks([0, 2, 4, 6, 8, 10])
ax[0].set_ylim(5e-5, 4)
ax[0].set_title("(a) real background traffic", fontsize=6, pad=1.5)

# ---------- (b) decade ladder, one configuration ----------
c = [x for x in C if abs(x["lam"] - 0.02) < 1e-9][0]
lad = [l for l in c["ladder"] if l["emp"] > 0]
ks = [l["k"] for l in lad]
ax[1].semilogy(ks, [l["bound"] for l in lad], "o-", ms=2.2, lw=0.8, color="C0")
ax[1].semilogy(ks, [l["emp"] for l in lad], "s--", ms=2.2, lw=0.8, color="C3",
               mfc="none")
ax[1].annotate("certified bound", xy=(ks[-1], lad[-1]["bound"]), xytext=(-2, 5),
               textcoords="offset points", ha="right", fontsize=5.4, color="C0")
ax[1].annotate("measured", xy=(ks[-3], lad[-3]["emp"]), xytext=(-4, -9),
               textcoords="offset points", ha="right", fontsize=5.4, color="C3")
ax[1].set_xlabel("confirmations $k$ (weakened)", labelpad=0.5)
ax[1].set_ylabel("false confirmation", labelpad=0.5)
ax[1].set_title("(b) bound vs. measurement", fontsize=6, pad=1.5)
fig.tight_layout(pad=0.2, w_pad=1.2)
out = os.path.join(FIG, "fig_balance_ladder.pdf")
fig.savefig(out, bbox_inches="tight")
print("WROTE", out)
