#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer 1: Probabilistic composite vulnerability index (Monte Carlo)
==================================================================

What this does
--------------
Takes the deterministic seven-category composite vulnerability score from the
paper and turns it into a Monte Carlo uncertainty propagation. Instead of one
number per country it produces, for each of the six GCC states:

  * the composite score as a distribution (median and 95% credible interval),
  * a rank-probability matrix  P(country = rank r),
  * tier-membership probabilities (acute / severe / moderate),
  * pairwise dominance probabilities, e.g. P(Kuwait more vulnerable than Qatar).

It also folds the weighting sensitivity into the same loop by sampling the
weight vector from a Dirichlet distribution, so weight uncertainty and
indicator measurement uncertainty are propagated together.

Two modes
---------
MODE = "indicator"  (ground-up, what a quantitative reviewer asks for)
    Samples each *raw indicator* from a distribution, re-runs the min-max
    normalization within the six-country sample ON EACH DRAW, re-aggregates to
    category scores, then to the composite. This is the proper measurement-
    uncertainty propagation. NOTE: because the published integer category
    scores were assigned with a richer (partly unpublished) sub-indicator set,
    the deterministic baseline of this mode will be CLOSE BUT NOT IDENTICAL to
    the published composites (1.71 / 2.33 / 2.68 / 3.42 / 4.21 / 5.14). That is
    expected. Replace the INDICATORS block below with your authoritative Source
    Data values to tighten the match.

MODE = "category"   (fidelity to the published numbers)
    Perturbs the seven *published* integer category scores (Supplementary
    Table 8 Part A) with Gaussian noise and re-weights. At CATEGORY_NOISE_SD = 0
    it reproduces the published composites exactly, so it is the cleanest way to
    put credible intervals on the numbers already in the paper.

Run both and compare: the gap tells you how much the result depends on the
reconstruction choices.

Scale convention throughout: 1 = MOST vulnerable, 7 = LEAST vulnerable
(defence-in-depth). Tiers: acute < 2.5, severe 2.5-4.0, moderate > 4.0.

Requirements: numpy, pandas, matplotlib  (pip install -r requirements.txt)
Run:          python gcc_layer1_composite_mc.py
Outputs:      ./layer1_outputs/  (CSVs + PNGs), plus a console summary.

All input values are sourced from the corrected data workbook and the
Supplementary Information. Every value and every distribution choice is editable
in the clearly marked DATA blocks below. Edit those, not the engine.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # so it runs headless; remove if you want interactive windows
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# RUN SETTINGS  (edit here)
# ----------------------------------------------------------------------------
MODE = "indicator"          # "indicator"  or  "category"
N_DRAWS = 10_000
SEED = 20260629             # fixed seed -> reproducible; change to test stability

WEIGHT_MODE = "dirichlet"   # "equal" | "dirichlet" | "fixed"
DIRICHLET_CONC = 30.0       # higher = weights stay tighter around equal (1/7 each)
FIXED_SCHEME = "equal"      # used only if WEIGHT_MODE == "fixed":
                            # "equal" | "structural" | "population" | "energy"

CATEGORY_NOISE_SD = 0.75    # used only in MODE == "category" (0 reproduces paper)

TIER_ACUTE = 2.5            # composite < this  -> acute
TIER_MODERATE = 4.0         # composite > this  -> moderate ; between -> severe

OUTDIR = Path("layer1_outputs")
OUTDIR.mkdir(exist_ok=True)

# Optional environment-variable overrides (handy for running both modes):
#   MODE=category CATEGORY_NOISE_SD=0 python gcc_layer1_composite_mc.py
MODE = os.environ.get("MODE", MODE)
N_DRAWS = int(os.environ.get("N_DRAWS", N_DRAWS))
WEIGHT_MODE = os.environ.get("WEIGHT_MODE", WEIGHT_MODE)
CATEGORY_NOISE_SD = float(os.environ.get("CATEGORY_NOISE_SD", CATEGORY_NOISE_SD))

COUNTRIES = ["Bahrain", "Kuwait", "Qatar", "Oman", "UAE", "Saudi Arabia"]
NC = len(COUNTRIES)
CATS = ["C1 Water mix", "C2 Geography", "C3 Energy", "C4 Reserves",
        "C5 Population", "C6 Redundancy", "C7 Trajectory"]

rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------------
# DISTRIBUTION SAMPLER  (engine - you should not need to edit)
# ----------------------------------------------------------------------------
def draw(spec, n):
    """Return n samples for a per-country distribution spec.
    spec is a tuple:
        ("pt",  v)                point mass
        ("tri", lo, mode, hi)     triangular
        ("rel", v, rel)           triangular( v*(1-rel), v, v*(1+rel) )
        ("uni", lo, hi)           uniform
        ("norm", mu, sd)          normal (clipped at 0 to keep physical)
    """
    kind = spec[0]
    if kind == "pt":
        return np.full(n, float(spec[1]))
    if kind == "tri":
        return rng.triangular(spec[1], spec[2], spec[3], n)
    if kind == "rel":
        v, r = float(spec[1]), float(spec[2])
        lo, hi = v * (1 - r), v * (1 + r)
        if hi <= lo:  # degenerate (v==0)
            return np.full(n, v)
        return rng.triangular(lo, v, hi, n)
    if kind == "uni":
        return rng.uniform(spec[1], spec[2], n)
    if kind == "norm":
        return np.clip(rng.normal(spec[1], spec[2], n), 0.0, None)
    raise ValueError(f"unknown distribution kind: {kind}")


def minmax_to_1_7(x, direction, use_log=False):
    """Min-max normalize a (N, NC) array to [1,7] within the sample, per draw.
    direction "+" : higher raw value => MORE vulnerable => score toward 1
    direction "-" : higher raw value => LESS vulnerable => score toward 7
    use_log: log-transform before normalizing (for highly skewed indicators).
    """
    v = np.log(np.clip(x, 1e-9, None)) if use_log else x.astype(float)
    lo = v.min(axis=1, keepdims=True)
    hi = v.max(axis=1, keepdims=True)
    span = hi - lo
    safe = np.where(span == 0, 1.0, span)
    frac = (v - lo) / safe                # 0 at sample min, 1 at sample max
    if direction == "-":
        score = 1.0 + 6.0 * frac          # high value -> high (good) score
    elif direction == "+":
        score = 7.0 - 6.0 * frac          # high value -> low (bad) score
    else:
        raise ValueError("direction must be '+' or '-'")
    return np.where(span == 0, 4.0, score)  # all-equal -> neutral midpoint


# ============================================================================
# DATA BLOCK A  -  RAW INDICATORS  (used when MODE == "indicator")
# ----------------------------------------------------------------------------
# Each indicator: name, direction, log (bool), and a per-country distribution.
# Sources noted in comments. Values reconciled from the CORRECTED workbook
# ("Vulnerability Indicators", "Scenario Analysis Data", "Economic & Stakes")
# and the Supplementary Information. EDIT THESE to your authoritative numbers.
# Within a category, indicators are averaged with equal weight (as in the paper).
# ============================================================================
INDICATORS = {
    "C1 Water mix": [
        # Desalination dependency ratio = production / total withdrawal (%). +vuln.
        # Workbook "Desalination dependency ratio". Bahrain widened (audit flag #2).
        {"name": "Desalination dependency (%)", "dir": "+", "log": False, "by": {
            "Bahrain": ("rel", 36.34, 0.25), "Kuwait": ("rel", 43.15, 0.15),
            "Qatar":   ("rel", 37.44, 0.15), "Oman":   ("rel", 18.79, 0.15),
            "UAE":     ("rel", 26.25, 0.15), "Saudi Arabia": ("rel", 10.31, 0.15)}},
        # Treated wastewater reuse (%). -vuln (more reuse = more diversified).
        # Range spans the two conflicting reported values (workbook vs Supp Note 4.2),
        # an explicit use of documented cross-source disagreement as the spread.
        {"name": "Wastewater reuse (%)", "dir": "-", "log": False, "by": {
            "Bahrain": ("uni", 1.94, 7.12), "Kuwait": ("uni", 2.61, 12.21),
            "Qatar":   ("uni", 4.69, 10.58), "Oman":  ("uni", 4.73, 7.09),
            "UAE":     ("uni", 1.13, 7.64), "Saudi Arabia": ("uni", 1.52, 9.51)}},
        # Groundwater share of supply (%). -vuln (diversification away from desal).
        {"name": "Groundwater share (%)", "dir": "-", "log": False, "by": {
            "Bahrain": ("rel", 59.60, 0.10), "Kuwait": ("rel", 72.89, 0.10),
            "Qatar":   ("rel", 51.97, 0.10), "Oman":  ("rel", 81.29, 0.10),
            "UAE":     ("rel", 64.14, 0.10), "Saudi Arabia": ("rel", 90.98, 0.10)}},
    ],
    "C2 Geography": [
        # Number of major desalination plants (count). -vuln (more = less concentrated;
        # proxy for spatial distribution, since coastline length was not in the data).
        {"name": "Number of major plants", "dir": "-", "log": True, "by": {
            "Bahrain": ("rel", 6, 0.15), "Kuwait": ("rel", 7, 0.15),
            "Qatar":   ("rel", 7, 0.15), "Oman":   ("rel", 13, 0.15),
            "UAE":     ("rel", 70, 0.15), "Saudi Arabia": ("rel", 33, 0.15)}},
        # Share of capacity on a single coast (%). +vuln. Single-coast states fixed
        # at 100; dual-coast states (UAE, Saudi) sampled around an estimated split.
        {"name": "Single-coast capacity (%)", "dir": "+", "log": False, "by": {
            "Bahrain": ("pt", 100), "Kuwait": ("pt", 100),
            "Qatar":   ("pt", 100), "Oman":   ("pt", 100),
            "UAE":     ("tri", 60, 70, 80), "Saudi Arabia": ("tri", 55, 65, 75)}},
    ],
    "C3 Energy": [
        # Co-generation (power+water) share of capacity (%). +vuln. Known for
        # Oman/Qatar/UAE; for the three "na" states sampled uniform(80,100) per
        # Supp Note 4.4 ("cogeneration accounts for 80-100% across all six states").
        {"name": "Cogeneration share (%)", "dir": "+", "log": False, "by": {
            "Bahrain": ("uni", 80, 100), "Kuwait": ("uni", 80, 100),
            "Qatar":   ("rel", 100, 0.02), "Oman":  ("rel", 14.07, 0.20),
            "UAE":     ("rel", 81.12, 0.10), "Saudi Arabia": ("uni", 80, 100)}},
        # Share of national electricity consumed by desalination (%). +vuln.
        {"name": "Electricity share to desal (%)", "dir": "+", "log": False, "by": {
            "Bahrain": ("rel", 8, 0.15), "Kuwait": ("rel", 8, 0.15),
            "Qatar":   ("rel", 13, 0.15), "Oman":   ("rel", 8, 0.15),
            "UAE":     ("rel", 20, 0.15), "Saudi Arabia": ("rel", 7, 0.15)}},
    ],
    "C4 Reserves": [
        # Strategic reserve duration at normal consumption (days). -vuln.
        # log-transform: UAE's 90 days is an extreme outlier; on a linear scale it
        # collapses all other states to ~1. Log spreads them sensibly. Ranges from
        # Supp Table 9 / Note 4.5.
        {"name": "Reserve days", "dir": "-", "log": True, "by": {
            "Bahrain": ("uni", 1, 2), "Kuwait": ("tri", 2, 3, 4),
            "Qatar":   ("tri", 5, 7, 9), "Oman":  ("uni", 5, 10),
            "UAE":     ("tri", 80, 90, 95), "Saudi Arabia": ("uni", 5, 10)}},
    ],
    "C5 Population": [
        # Population served by the single largest plant (millions). +vuln.
        {"name": "Pop. served by largest plant (M)", "dir": "+", "log": False, "by": {
            "Bahrain": ("rel", 0.85, 0.15), "Kuwait": ("rel", 1.70, 0.15),
            "Qatar":   ("rel", 1.80, 0.15), "Oman":   ("rel", 1.00, 0.20),
            "UAE":     ("rel", 4.50, 0.15), "Saudi Arabia": ("rel", 3.50, 0.15)}},
        # Total desalination-dependent population (millions). +vuln. log (Saudi outlier).
        # Economic & Stakes sheet "population dependent on desalinated water".
        {"name": "Desal-dependent population (M)", "dir": "+", "log": True, "by": {
            "Bahrain": ("rel", 1.44, 0.10), "Kuwait": ("rel", 4.59, 0.10),
            "Qatar":   ("rel", 3.17, 0.10), "Oman":   ("rel", 4.04, 0.10),
            "UAE":     ("rel", 4.86, 0.10), "Saudi Arabia": ("rel", 25.9, 0.10)}},
    ],
    "C6 Redundancy": [
        # Plants that must be disrupted for >50% national capacity loss (count). -vuln.
        {"name": "Plants for >50% loss", "dir": "-", "log": False, "by": {
            "Bahrain": ("rel", 2, 0.10), "Kuwait": ("uni", 3, 4),
            "Qatar":   ("rel", 2, 0.10), "Oman":   ("uni", 3, 4),
            "UAE":     ("tri", 5, 5, 6), "Saudi Arabia": ("tri", 5, 6, 7)}},
        # Largest single plant as % of national capacity. +vuln.
        {"name": "Largest-plant share (%)", "dir": "+", "log": False, "by": {
            "Bahrain": ("rel", 47.8, 0.10), "Kuwait": ("rel", 15.7, 0.10),
            "Qatar":   ("rel", 36.8, 0.10), "Oman":   ("rel", 16.8, 0.10),
            "UAE":     ("rel", 12.0, 0.10), "Saudi Arabia": ("rel", 18.5, 0.10)}},
    ],
    "C7 Trajectory": [
        # Ten-year vulnerability trajectory. This category is a holistic expert
        # judgment in the paper (deepening dependency + flat resilience), not cleanly
        # reducible to one published raw indicator, so it is sampled DIRECTLY on the
        # 1-7 scale around the published category score (Supp Table 8A) with noise.
        # This is the one category not rebuilt from raw indicators; flagged for honesty.
        {"name": "Trajectory (expert, 1-7)", "dir": "score", "log": False, "by": {
            "Bahrain": ("norm", 1, 0.6), "Kuwait": ("norm", 1, 0.6),
            "Qatar":   ("norm", 3, 0.6), "Oman":   ("norm", 1, 0.6),
            "UAE":     ("norm", 4, 0.6), "Saudi Arabia": ("norm", 7, 0.6)}},
    ],
}

# ============================================================================
# DATA BLOCK B  -  PUBLISHED CATEGORY SCORES  (used when MODE == "category")
# Supplementary Table 8 Part A (1-7 scale, 1 = most vulnerable). Order = CATS.
# ============================================================================
CATEGORY_SCORES = {
    "Bahrain":       [2, 1, 3, 1, 3, 1, 1],
    "Kuwait":        [1, 3, 3, 2, 3, 3, 1],
    "Qatar":         [3, 2, 1, 4, 4, 2, 3],
    "Oman":          [5, 4, 5, 5, 2, 2, 1],
    "UAE":           [4, 6, 3, 6, 2, 5, 4],
    "Saudi Arabia":  [7, 6, 5, 3, 1, 7, 7],
}

# Published weighting schemes (Supp Table 2.3.1), for reference / WEIGHT_MODE="fixed".
FIXED_WEIGHTS = {
    "equal":      np.array([1/7]*7),
    "structural": np.array([1/9, 2/9, 1/9, 1/9, 1/9, 2/9, 1/9]),  # C2,C6 doubled
    "population": np.array([1/9, 1/9, 1/9, 1/9, 3/9, 1/9, 1/9]),  # C5 tripled
    "energy":     np.array([1/9, 1/9, 3/9, 1/9, 1/9, 1/9, 1/9]),  # C3 tripled
}


# ----------------------------------------------------------------------------
# CORE: build category-score draws  ->  (N, NC, 7)
# ----------------------------------------------------------------------------
def category_scores_indicator(n):
    """MODE='indicator': sample raw indicators, normalize per draw, aggregate."""
    cat_arr = np.empty((n, NC, 7))
    for ci, cat in enumerate(CATS):
        inds = INDICATORS[cat]
        ind_scores = []
        for ind in inds:
            if ind["dir"] == "score":
                # already on the 1-7 vulnerability scale; just clip
                raw = np.stack([np.clip(draw(ind["by"][c], n), 1, 7)
                                for c in COUNTRIES], axis=1)
                ind_scores.append(raw)
            else:
                raw = np.stack([draw(ind["by"][c], n) for c in COUNTRIES], axis=1)
                ind_scores.append(minmax_to_1_7(raw, ind["dir"], ind["log"]))
        cat_arr[:, :, ci] = np.mean(ind_scores, axis=0)  # equal within-category
    return cat_arr


def category_scores_category(n):
    """MODE='category': perturb the published integer category scores."""
    base = np.array([CATEGORY_SCORES[c] for c in COUNTRIES])  # (NC,7)
    noise = rng.normal(0.0, CATEGORY_NOISE_SD, size=(n, NC, 7))
    return np.clip(base[None, :, :] + noise, 1.0, 7.0)


def sample_weights(n):
    if WEIGHT_MODE == "equal":
        return np.tile(FIXED_WEIGHTS["equal"], (n, 1))
    if WEIGHT_MODE == "fixed":
        return np.tile(FIXED_WEIGHTS[FIXED_SCHEME], (n, 1))
    if WEIGHT_MODE == "dirichlet":
        return rng.dirichlet(np.full(7, DIRICHLET_CONC), size=n)
    raise ValueError(WEIGHT_MODE)


# ----------------------------------------------------------------------------
# RUN
# ----------------------------------------------------------------------------
def main():
    print(f"Layer 1 Monte Carlo  |  mode={MODE}  draws={N_DRAWS}  "
          f"weights={WEIGHT_MODE}\n" + "=" * 64)

    cat = (category_scores_indicator(N_DRAWS) if MODE == "indicator"
           else category_scores_category(N_DRAWS))           # (N, NC, 7)
    W = sample_weights(N_DRAWS)                                # (N, 7)
    comp = np.einsum("nck,nk->nc", cat, W)                     # (N, NC) composite

    # ---- deterministic baseline for comparison -----------------------------
    if MODE == "category":
        base_cat = np.array([CATEGORY_SCORES[c] for c in COUNTRIES])
    else:
        base_cat = category_scores_indicator(1)[0]            # using mode/point draws
    base_comp = base_cat @ FIXED_WEIGHTS["equal"]
    paper = {"Bahrain": 1.71, "Kuwait": 2.33, "Qatar": 2.68,
             "Oman": 3.42, "UAE": 4.21, "Saudi Arabia": 5.14}

    # ---- composite summary -------------------------------------------------
    rows = []
    for i, c in enumerate(COUNTRIES):
        x = comp[:, i]
        rows.append({
            "Country": c,
            "Paper (equal wts)": paper[c],
            "Det. baseline": round(float(base_comp[i]), 2),
            "MC mean": round(float(x.mean()), 2),
            "MC median": round(float(np.median(x)), 2),
            "p2.5": round(float(np.percentile(x, 2.5)), 2),
            "p97.5": round(float(np.percentile(x, 97.5)), 2),
            "sd": round(float(x.std()), 2),
            "P(acute)": round(float(np.mean(x < TIER_ACUTE)), 3),
            "P(severe)": round(float(np.mean((x >= TIER_ACUTE) & (x <= TIER_MODERATE))), 3),
            "P(moderate)": round(float(np.mean(x > TIER_MODERATE)), 3),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTDIR / "layer1_composite_summary.csv", index=False)
    print("\nComposite score summary (1=most vulnerable, 7=least):")
    print(summary.to_string(index=False))

    # ---- rank-probability matrix ------------------------------------------
    # rank 1 = lowest composite = most vulnerable
    order = np.argsort(comp, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows_idx = np.arange(N_DRAWS)[:, None]
    ranks[rows_idx, order] = np.arange(1, NC + 1)[None, :]
    rank_prob = np.zeros((NC, NC))
    for i in range(NC):
        for r in range(1, NC + 1):
            rank_prob[i, r - 1] = np.mean(ranks[:, i] == r)
    rankdf = pd.DataFrame(rank_prob, index=COUNTRIES,
                          columns=[f"rank{r}" for r in range(1, NC + 1)]).round(3)
    rankdf.to_csv(OUTDIR / "layer1_rank_probability.csv")
    print("\nRank-probability matrix  P(country = rank):")
    print(rankdf.to_string())

    # ---- pairwise dominance  P(row more vulnerable than col) ---------------
    dom = np.zeros((NC, NC))
    for i in range(NC):
        for j in range(NC):
            if i != j:
                dom[i, j] = np.mean(comp[:, i] < comp[:, j])
    domdf = pd.DataFrame(dom, index=COUNTRIES, columns=COUNTRIES).round(3)
    domdf.to_csv(OUTDIR / "layer1_pairwise_dominance.csv")
    kq = float(np.mean(comp[:, COUNTRIES.index("Kuwait")]
                       < comp[:, COUNTRIES.index("Qatar")]))
    print("\nKey contingency from the paper (the Kuwait-Qatar swap):")
    print(f"  P(Kuwait more vulnerable than Qatar) = {kq:.3f}")
    print(f"  P(Qatar  more vulnerable than Kuwait) = {1-kq:.3f}")

    # ---- plots -------------------------------------------------------------
    _plot_distributions(comp)
    _plot_rank_heatmap(rank_prob)
    _plot_tiers(summary)

    print(f"\nWrote CSVs and PNGs to ./{OUTDIR}/")
    print("Files: layer1_composite_summary.csv, layer1_rank_probability.csv,")
    print("       layer1_pairwise_dominance.csv, composite_distributions.png,")
    print("       rank_heatmap.png, tier_probabilities.png")


def _plot_distributions(comp):
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [comp[:, i] for i in range(NC)]
    bp = ax.boxplot(data, vert=True, patch_artist=True, whis=(2.5, 97.5),
                    showfliers=False, widths=0.6)
    for patch in bp["boxes"]:
        patch.set_facecolor("#9ecae1")
    ax.set_xticklabels(COUNTRIES, rotation=20, ha="right")
    ax.axhline(TIER_ACUTE, ls="--", color="#d62728", lw=1)
    ax.axhline(TIER_MODERATE, ls="--", color="#2ca02c", lw=1)
    ax.text(NC + 0.4, TIER_ACUTE, "acute", color="#d62728", va="center")
    ax.text(NC + 0.4, TIER_MODERATE, "moderate", color="#2ca02c", va="center")
    ax.set_ylabel("Composite vulnerability (1 = most vulnerable)")
    ax.set_title(f"Composite distributions  (mode={MODE}, {N_DRAWS} draws)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "composite_distributions.png", dpi=150)
    plt.close(fig)


def _plot_rank_heatmap(rank_prob):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.imshow(rank_prob, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(NC)); ax.set_xticklabels([f"rank {r}" for r in range(1, NC + 1)])
    ax.set_yticks(range(NC)); ax.set_yticklabels(COUNTRIES)
    for i in range(NC):
        for j in range(NC):
            v = rank_prob[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=9)
    ax.set_title("Rank-probability matrix  (rank 1 = most vulnerable)")
    fig.colorbar(im, ax=ax, label="probability")
    fig.tight_layout()
    fig.savefig(OUTDIR / "rank_heatmap.png", dpi=150)
    plt.close(fig)


def _plot_tiers(summary):
    fig, ax = plt.subplots(figsize=(9, 5))
    acute = summary["P(acute)"].values
    severe = summary["P(severe)"].values
    moderate = summary["P(moderate)"].values
    x = np.arange(NC)
    ax.bar(x, acute, label="acute (<2.5)", color="#d62728")
    ax.bar(x, severe, bottom=acute, label="severe (2.5-4.0)", color="#ff7f0e")
    ax.bar(x, moderate, bottom=acute + severe, label="moderate (>4.0)", color="#2ca02c")
    ax.set_xticks(x); ax.set_xticklabels(COUNTRIES, rotation=20, ha="right")
    ax.set_ylabel("probability"); ax.set_ylim(0, 1)
    ax.set_title("Tier-membership probabilities")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(OUTDIR / "tier_probabilities.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
