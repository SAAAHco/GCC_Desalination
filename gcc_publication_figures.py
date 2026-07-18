#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication figures (category mode) -> EPS + PDF + PNG
======================================================

Outputs each new figure as vector EPS (Elsevier's preferred vector format),
with PDF and 300 dpi PNG alongside. Category mode keeps Kuwait more vulnerable
than Qatar, consistent with Fig 1b and the manuscript text.

EPS notes: PostScript has no alpha channel, so tier bands use solid light fills
(not transparency); fonts are embedded (Type 42) so text stays editable; the
heatmap cells are an embedded raster (standard for a colour grid) while all text
and axes remain vector.

Files written per figure: <name>.eps, <name>.pdf, <name>.png
  Fig_1b_composite_distributions
  SI_rank_probability_heatmap
  Fig_5c_time_to_crisis

Run:  python gcc_publication_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["ps.fonttype"] = 42     # embed TrueType in EPS
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt

SEED = 20260629
N = 10_000
PNG_DPI = 300
RASTER_DPI = 600
rng = np.random.default_rng(SEED)

COUNTRIES = ["Bahrain", "Kuwait", "Qatar", "Oman", "UAE", "Saudi Arabia"]
NC = len(COUNTRIES)
TIER_ACUTE, TIER_MODERATE = 2.5, 4.0

# Solid, EPS-safe tier tints (no alpha).
C_ACUTE, C_SEVERE, C_MOD = "#fdecea", "#fff4e6", "#eaf6ea"

CATEGORY_SCORES = {
    "Bahrain":      [2, 1, 3, 1, 3, 1, 1],
    "Kuwait":       [1, 3, 3, 2, 3, 3, 1],
    "Qatar":        [3, 2, 1, 4, 4, 2, 3],
    "Oman":         [5, 4, 5, 5, 2, 2, 1],
    "UAE":          [4, 6, 3, 6, 2, 5, 4],
    "Saudi Arabia": [7, 6, 5, 3, 1, 7, 7],
}
CATEGORY_NOISE_SD = 0.75
DIRICHLET_CONC = 30.0


def save(fig, name):
    """Write EPS (vector), PDF (vector), and PNG (raster) for one figure."""
    fig.savefig(name + ".eps", format="eps", dpi=RASTER_DPI, bbox_inches="tight")
    fig.savefig(name + ".pdf", format="pdf", bbox_inches="tight")
    fig.savefig(name + ".png", format="png", dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


# ---- Layer-1 category-mode Monte Carlo ------------------------------------
base = np.array([CATEGORY_SCORES[c] for c in COUNTRIES])
cat = np.clip(base[None, :, :] + rng.normal(0, CATEGORY_NOISE_SD, (N, NC, 7)), 1, 7)
W = rng.dirichlet(np.full(7, DIRICHLET_CONC), size=N)
comp = np.einsum("nck,nk->nc", cat, W)

order = np.argsort(comp, axis=1, kind="stable")
ranks = np.empty_like(order)
ranks[np.arange(N)[:, None], order] = np.arange(1, NC + 1)[None, :]
rank_prob = np.array([[np.mean(ranks[:, i] == r) for r in range(1, NC + 1)]
                      for i in range(NC)])

print("Category-mode composite (median, 95% CI, P(acute), P(rank1)):")
for i, c in enumerate(COUNTRIES):
    x = comp[:, i]
    print(f"  {c:14s} {np.median(x):.2f}  [{np.percentile(x,2.5):.2f}, "
          f"{np.percentile(x,97.5):.2f}]  {np.mean(x<TIER_ACUTE):.2f}  {rank_prob[i,0]:.2f}")
print(f"  P(Kuwait > Qatar in vulnerability) = {np.mean(comp[:,1] < comp[:,2]):.2f}")

# ---- Figure 1b : composite distributions ----------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.axhspan(1, TIER_ACUTE, facecolor=C_ACUTE, edgecolor="none", zorder=0)
ax.axhspan(TIER_ACUTE, TIER_MODERATE, facecolor=C_SEVERE, edgecolor="none", zorder=0)
ax.axhspan(TIER_MODERATE, 6, facecolor=C_MOD, edgecolor="none", zorder=0)
bp = ax.boxplot([comp[:, i] for i in range(NC)], vert=True, patch_artist=True,
                whis=(2.5, 97.5), showfliers=False, widths=0.62, zorder=3,
                medianprops=dict(color="black", linewidth=1.4))
for patch in bp["boxes"]:
    patch.set_facecolor("#9ecae1"); patch.set_edgecolor("#33618a")
ax.axhline(TIER_ACUTE, ls="--", color="#d62728", lw=1, zorder=2)
ax.axhline(TIER_MODERATE, ls="--", color="#2ca02c", lw=1, zorder=2)
ax.text(NC + 0.55, TIER_ACUTE, "acute", color="#d62728", va="center", fontsize=9)
ax.text(NC + 0.55, TIER_MODERATE, "moderate", color="#2ca02c", va="center", fontsize=9)
ax.set_xticks(range(1, NC + 1)); ax.set_xticklabels(COUNTRIES, rotation=18, ha="right")
ax.set_ylabel("Composite vulnerability score\n(1 = most vulnerable, 7 = defence-in-depth)")
ax.set_ylim(1, 6)
ax.set_title("Composite vulnerability with 95% credible intervals (10,000 draws)")
save(fig, "Fig_1b_composite_distributions")

# ---- Supplementary : rank-probability heatmap -----------------------------
fig, ax = plt.subplots(figsize=(7.0, 4.6))
# pcolormesh keeps the heatmap as vector quads (small EPS); imshow would embed a raster.
mesh = ax.pcolormesh(rank_prob, cmap="Blues", vmin=0, vmax=1,
                     edgecolors="white", linewidth=0.6)
ax.invert_yaxis()                       # Bahrain (row 0) at top
ax.set_aspect("auto")
ax.set_xticks(np.arange(NC) + 0.5); ax.set_xticklabels([f"rank {r}" for r in range(1, NC + 1)])
ax.set_yticks(np.arange(NC) + 0.5); ax.set_yticklabels(COUNTRIES)
for i in range(NC):
    for j in range(NC):
        v = rank_prob[i, j]
        if v >= 0.005:
            ax.text(j + 0.5, i + 0.5, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.5 else "black", fontsize=9)
ax.set_title("Rank-probability matrix (rank 1 = most vulnerable)")
fig.colorbar(mesh, ax=ax, label="probability")
save(fig, "SI_rank_probability_heatmap")

# ---- Figure 5c : S3 grid-cascade time-to-crisis exceedance -----------------
NAT = {
    "Bahrain":      dict(reserve=(1, 2),   grid=("uni", 0.80, 1.00)),
    "Kuwait":       dict(reserve=(2, 4),   grid=("uni", 0.80, 1.00)),
    "Qatar":        dict(reserve=(5, 9),   grid=("pt", 1.00)),
    "Oman":         dict(reserve=(5, 10),  grid=("pt", 0.14)),
    "UAE":          dict(reserve=(80, 95), grid=("pt", 0.81)),
    "Saudi Arabia": dict(reserve=(5, 10),  grid=("uni", 0.80, 1.00)),
}
def draw(spec, n):
    if spec[0] == "pt":  return np.full(n, float(spec[1]))
    if spec[0] == "uni": return rng.uniform(spec[1], spec[2], n)
    raise ValueError(spec)

fig, ax = plt.subplots(figsize=(7.2, 4.6))
grid_days = np.linspace(0, 30, 240)
colors = plt.cm.tab10(np.linspace(0, 1, NC))
for i, c in enumerate(COUNTRIES):
    nat = NAT[c]
    g = draw(nat["grid"], N)
    sub = rng.uniform(0.0, 0.10, N)
    R = rng.uniform(*nat["reserve"], N)
    d = np.clip(g - sub, 1e-9, 1.0)
    Tc = R / d
    ax.plot(grid_days, [np.mean(Tc <= t) for t in grid_days],
            label=c, color=colors[i], lw=1.8)
ax.axvline(3, ls="--", color="grey", lw=1)
ax.text(3.2, 0.05, "72 h", color="grey", fontsize=9)
ax.set_xlabel("days after grid outage")
ax.set_ylabel("P(reserves exhausted by this day)")
ax.set_ylim(0, 1.02)
ax.set_title("S3 energy-grid cascade: time-to-crisis exceedance curves")
ax.legend(fontsize=8, loc="lower right")
save(fig, "Fig_5c_time_to_crisis")

print("\nWrote EPS + PDF + PNG for:")
for n in ["Fig_1b_composite_distributions", "SI_rank_probability_heatmap",
          "Fig_5c_time_to_crisis"]:
    print("  " + n + ".eps")
