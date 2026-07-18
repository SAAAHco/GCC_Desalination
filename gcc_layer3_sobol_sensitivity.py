#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer 3B: Sobol global sensitivity analysis
===========================================

Ranks which unmeasured parameters drive the variance of the disruption-loss
output, turning the paper's data gaps into a prioritized data-collection agenda.

For each state, the model evaluated is the cumulative economic loss over a fixed
horizon from a coordinated strike on the three largest plants (the same core as
Layer 2). The uncertain inputs are exactly the operational unknowns the paper
flags as the highest-priority data targets:

    substitution        inter-plant substitution fraction
    reserve_days        strategic reserve duration (country-specific range)
    restart_days        time to restore a struck plant (drone-days to missile-years)
    zipf_exponent       steepness of the plant-size distribution (capacity spread)
    largest_share       largest plant's share of national capacity

(The full-outage daily loss is held at its country midpoint, because it scales
the output linearly and would dominate the indices trivially; the point here is
to rank the OPERATIONAL unknowns, not the known cost range.)

Sobol total-order indices (ST) measure each parameter's total contribution to
output variance, including interactions. The parameter with the largest ST is
the one whose measurement would most reduce uncertainty: collect that first.

Requirements: numpy, pandas, matplotlib, SALib
Run:          python gcc_layer3_sobol_sensitivity.py
Outputs:      ./layer3_outputs/ (CSV + heatmap PNG) and a console summary.

Note: the absolute indices depend on the parameter ranges below. Widen or narrow
them to your own engineering estimates; the RANKING is the robust output.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from SALib.sample.sobol import sample as sobol_sample
from SALib.analyze.sobol import analyze as sobol_analyze

SEED = 20260629
N_BASE = 1024              # Saltelli base sample (power of 2). Evals = N_BASE*(2k+2).
HORIZON_DAYS = 365.0
OUTDIR = Path("layer3_outputs")
OUTDIR.mkdir(exist_ok=True)

COUNTRIES = ["Bahrain", "Kuwait", "Qatar", "Oman", "UAE", "Saudi Arabia"]

# Per-country fixed context (same sources as Layer 2).
CTX = {
    "Bahrain":      dict(n_plants=6,  reserve=(1, 2),   daily_loss_mid=20.0,  largest=47.8),
    "Kuwait":       dict(n_plants=7,  reserve=(2, 4),   daily_loss_mid=102.5, largest=15.7),
    "Qatar":        dict(n_plants=7,  reserve=(5, 9),   daily_loss_mid=135.0, largest=36.8),
    "Oman":         dict(n_plants=13, reserve=(5, 10),  daily_loss_mid=42.5,  largest=16.8),
    "UAE":          dict(n_plants=70, reserve=(80, 95), daily_loss_mid=315.0, largest=12.0),
    "Saudi Arabia": dict(n_plants=33, reserve=(5, 10),  daily_loss_mid=550.0, largest=18.5),
}

PARAMS = ["substitution", "reserve_days", "restart_days", "zipf_exponent", "largest_share"]


def model_eval(X, ctx):
    """Vectorized cumulative-loss model for one country.
    X columns = PARAMS order. Returns array of cumulative loss ($M) over horizon.
    """
    sub = X[:, 0]
    R = X[:, 1]
    restart = X[:, 2]
    zipf = X[:, 3]
    s1 = X[:, 4] / 100.0                              # largest share as fraction
    n = ctx["n_plants"]

    # top-3 cumulative capacity share from a sampled Zipf size distribution,
    # with the top plant fixed to s1 and the rest split by rank^(-zipf).
    ranks = np.arange(2, n + 1, dtype=float)          # ranks 2..n
    w = ranks[None, :] ** (-zipf[:, None])            # (N, n-1)
    wsum = w.sum(axis=1)
    w2 = 2.0 ** (-zipf)
    w3 = 3.0 ** (-zipf)
    L = s1 + (1 - s1) * (w2 + w3) / wsum              # top-3 cumulative loss fraction
    L = np.clip(L, 0.0, 1.0)

    d = np.clip(L - sub, 0.0, 1.0)                    # net deficit
    has = d > 1e-9
    T_crisis = np.where(has, R / np.where(d > 0, d, 1.0), np.inf)
    daily_loss = ctx["daily_loss_mid"] * d
    end = np.minimum(HORIZON_DAYS, restart)
    days_in_crisis = np.clip(end - T_crisis, 0.0, None)
    return daily_loss * days_in_crisis               # $M over horizon


def run():
    print("Layer 3B  Sobol global sensitivity analysis\n" + "=" * 64)
    print(f"Output: cumulative loss ($M) over {int(HORIZON_DAYS)} d from a top-3 "
          f"coordinated strike.\nParameters: {', '.join(PARAMS)}\n")

    all_rows = []
    ST_matrix = np.zeros((len(PARAMS), len(COUNTRIES)))
    for ci, c in enumerate(COUNTRIES):
        ctx = CTX[c]
        problem = {
            "num_vars": len(PARAMS),
            "names": PARAMS,
            "bounds": [
                [0.0, 0.40],                                  # substitution
                [ctx["reserve"][0], ctx["reserve"][1]],       # reserve_days
                [10.0, 1825.0],                               # restart_days
                [0.6, 1.4],                                   # zipf_exponent
                [ctx["largest"] * 0.9, ctx["largest"] * 1.1], # largest_share (%)
            ],
        }
        X = sobol_sample(problem, N_BASE, seed=SEED)
        Y = model_eval(X, ctx)
        Si = sobol_analyze(problem, Y, seed=SEED, print_to_console=False)
        for pi, p in enumerate(PARAMS):
            ST_matrix[pi, ci] = max(Si["ST"][pi], 0.0)
            all_rows.append({"Country": c, "Parameter": p,
                             "S1": round(float(Si["S1"][pi]), 3),
                             "ST": round(float(Si["ST"][pi]), 3)})
        dom = PARAMS[int(np.argmax(Si["ST"]))]
        print(f"  {c:13s} dominant unknown: {dom}")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTDIR / "layer3_sobol_indices.csv", index=False)

    # overall priority = mean total-order index across countries
    mean_ST = ST_matrix.mean(axis=1)
    order = np.argsort(-mean_ST)
    print("\nOverall data-collection priority (mean total-order index across states):")
    for rank, pi in enumerate(order, start=1):
        print(f"  {rank}. {PARAMS[pi]:14s} mean ST = {mean_ST[pi]:.3f}")

    _heatmap(ST_matrix)
    print(f"\nWrote ./{OUTDIR}/layer3_sobol_indices.csv and sobol_ST_heatmap.png")


def _heatmap(ST):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    im = ax.imshow(ST, cmap="magma", aspect="auto", vmin=0, vmax=max(ST.max(), 0.1))
    ax.set_xticks(range(len(COUNTRIES))); ax.set_xticklabels(COUNTRIES, rotation=20, ha="right")
    ax.set_yticks(range(len(PARAMS))); ax.set_yticklabels(PARAMS)
    for i in range(len(PARAMS)):
        for j in range(len(COUNTRIES)):
            ax.text(j, i, f"{ST[i, j]:.2f}", ha="center", va="center",
                    color="white" if ST[i, j] < 0.6 * ST.max() else "black", fontsize=8)
    ax.set_title("Sobol total-order indices: which unknown drives loss variance")
    fig.colorbar(im, ax=ax, label="total-order index (ST)")
    fig.tight_layout(); fig.savefig(OUTDIR / "sobol_ST_heatmap.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
