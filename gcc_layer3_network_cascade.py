#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer 3A: Coupled-network cascade model
=======================================

Formalizes the Rinaldi critical-infrastructure interdependency framing as a
mechanistic model. The GCC water-energy system is represented as a multi-layer
dependency network and failures are propagated through it.

Node layers
    STRAIT                shared regional fuel/shipping hub (geographic + physical
                          shared upstream dependency for every state)
    GAS_<country>         national fuel intake (depends partly on the Strait,
                          partly on domestic supply)
    SCADA_<country>       digital control surface (cyber dependency; structurally
                          present, not exercised in March 2026)
    GRID_<country>        power / grid control node (logical-control dependency;
                          the 12 March 2026 Kuwait cascade ran through this)
    P_<country>_<i>       desalination plant nodes, weighted by capacity share;
                          cogeneration-coupled plants depend on GRID, self-
                          generating plants depend only on GAS
    DEMAND_<country>      population-weighted demand served

The four Rinaldi interdependency types are all present: physical (gas -> grid ->
cogeneration plant -> demand), geographic (plants share a coast and the Strait),
logical (GRID control), and cyber (SCADA). A node is available only if it is not
removed and its upstream dependencies are available; the Strait is modeled as a
capacity stressor (its loss removes the Strait-reliant fraction of every state's
fuel) rather than an all-or-nothing switch.

What it computes
    1. Robustness curves: regional demand served vs nodes removed, under a
       greedy TARGETED attack and under RANDOM failure. The gap is the
       targeted-attack vulnerability.
    2. Single-node criticality: regional demand lost from removing each node
       alone, ranked (identifies the systemic choke points).
    3. Minimum critical node set: fewest nodes whose loss drops regional demand
       below 50% (greedy upper bound).
    4. A country vulnerability ranking derived from the PHYSICAL model
       (min nodes for >50% national loss, and worst single-node loss), compared
       against the composite-index ranking. Agreement is triangulation.
    5. A Strait-removal demonstration of the shared-upstream cascade.

IMPORTANT: the topology is constructed from aggregate national data (plant
counts, cogeneration shares, coast configuration, Strait reliance), NOT a
surveyed plant-to-substation-to-pipeline map. It is a stylized model whose
STRUCTURE is the contribution; replace the assumptions in the DATA block with a
real topology when you have one. Per-plant shares default to a Zipf proxy and
can be overridden in PER_PLANT.

Requirements: numpy, pandas, matplotlib, networkx
Run:          python gcc_layer3_network_cascade.py
Outputs:      ./layer3_outputs/  (CSVs + PNGs) and a console summary.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

SEED = 20260629
rng = np.random.default_rng(SEED)
OUTDIR = Path("layer3_outputs")
OUTDIR.mkdir(exist_ok=True)

COUNTRIES = ["Bahrain", "Kuwait", "Qatar", "Oman", "UAE", "Saudi Arabia"]
INDEX_RANK = {c: r for r, c in enumerate(COUNTRIES, start=1)}  # composite-index order

# ============================================================================
# DATA BLOCK  (edit here)
# n_plants        : number of major plants                 [Vulnerability Indicators]
# largest_share   : largest plant % of national capacity   (anchors the Zipf proxy)
# cogen_frac      : capacity fraction that is cogeneration-coupled (grid-dependent)
#                   [Supp Note 4.4; Qatar/UAE/Oman known, others ~0.9]
# strait_reliance : fraction of national fuel that depends on the Strait hub
#                   (ASSUMPTION - tune to your trade/pipeline data)
# dependent_pop_M : desal-dependent population (regional demand weighting)
# dual_coast_primary : for dual-coast states, share of capacity on the primary coast
# ============================================================================
NATIONAL = {
    "Bahrain":      dict(n_plants=6,  largest_share=47.8, cogen_frac=0.90,
                         strait_reliance=0.50, dependent_pop_M=1.44, dual_coast_primary=1.0),
    "Kuwait":       dict(n_plants=7,  largest_share=15.7, cogen_frac=0.90,
                         strait_reliance=0.50, dependent_pop_M=4.59, dual_coast_primary=1.0),
    "Qatar":        dict(n_plants=7,  largest_share=36.8, cogen_frac=1.00,
                         strait_reliance=0.20, dependent_pop_M=3.17, dual_coast_primary=1.0),
    "Oman":         dict(n_plants=13, largest_share=16.8, cogen_frac=0.14,
                         strait_reliance=0.30, dependent_pop_M=4.04, dual_coast_primary=1.0),
    "UAE":          dict(n_plants=70, largest_share=12.0, cogen_frac=0.81,
                         strait_reliance=0.40, dependent_pop_M=4.86, dual_coast_primary=0.70),
    "Saudi Arabia": dict(n_plants=33, largest_share=18.5, cogen_frac=0.90,
                         strait_reliance=0.30, dependent_pop_M=25.9, dual_coast_primary=0.65),
}

# Optional: override the Zipf plant-size proxy with real shares (must sum to ~1).
# e.g. PER_PLANT = {"Bahrain": [0.478, 0.30, 0.12, 0.06, 0.03, 0.012]}
PER_PLANT = {}

ZIPF_EXPONENT = 1.0   # decay of the synthetic plant-size distribution


# ----------------------------------------------------------------------------
# Build per-country plant shares, cogeneration flags, and coast labels
# ----------------------------------------------------------------------------
def plant_shares(country):
    nat = NATIONAL[country]
    if country in PER_PLANT:
        s = np.array(PER_PLANT[country], float)
        return s / s.sum()
    npl, s1 = nat["n_plants"], nat["largest_share"] / 100.0
    if npl <= 1:
        return np.array([1.0])
    ranks = np.arange(2, npl + 1, dtype=float)
    w = ranks ** (-ZIPF_EXPONENT)
    w = w / w.sum()
    return np.concatenate([[s1], (1 - s1) * w])


def build_plant_meta(country):
    """Return list of dicts: share, cogen(bool), coast('A'/'B')."""
    nat = NATIONAL[country]
    shares = plant_shares(country)
    order = np.argsort(-shares)                      # largest first
    cogen_target = nat["cogen_frac"]
    primary = nat["dual_coast_primary"]
    meta = [None] * len(shares)
    cum_cogen = 0.0
    cum_primary = 0.0
    for k in order:
        s = shares[k]
        is_cogen = cum_cogen < cogen_target - 1e-9   # largest plants are cogen-coupled
        if is_cogen:
            cum_cogen += s
        coast = "A" if cum_primary < primary - 1e-9 else "B"
        if coast == "A":
            cum_primary += s
        meta[k] = dict(share=float(s), cogen=bool(is_cogen), coast=coast)
    return meta

PLANT_META = {c: build_plant_meta(c) for c in COUNTRIES}


# ----------------------------------------------------------------------------
# Availability propagation given a removed-node set
# ----------------------------------------------------------------------------
def node_names():
    nodes = ["STRAIT"]
    for c in COUNTRIES:
        nodes += [f"SCADA_{c}", f"GRID_{c}", f"DEMAND_{c}"]
        nodes += [f"P_{c}_{i}" for i in range(len(PLANT_META[c]))]
    return nodes


def availability(removed):
    """Return dict node -> availability in [0,1] given the removed set.

    Fuel is distributed (no single national gas node): each plant draws fuel
    that is strait_reliance from the shared Strait and the rest domestic. The
    grid-control node (and its SCADA surface) is shared per country and feeds the
    cogeneration-coupled plants only; self-generating plants need fuel but not
    the grid. So a GRID/SCADA loss removes the cogeneration-coupled fraction (the
    12 March 2026 logical-control cascade), and a Strait loss removes the
    strait-reliant fuel fraction for every state at once.
    """
    a = {}
    a["STRAIT"] = 0.0 if "STRAIT" in removed else 1.0
    for c in COUNTRIES:
        sr = NATIONAL[c]["strait_reliance"]
        fuel = sr * a["STRAIT"] + (1 - sr) * 1.0          # domestic fuel always on
        scada = 0.0 if f"SCADA_{c}" in removed else 1.0
        grid = 0.0 if f"GRID_{c}" in removed else scada   # grid needs its control surface
        a[f"SCADA_{c}"], a[f"GRID_{c}"] = scada, grid
        for i, m in enumerate(PLANT_META[c]):
            if f"P_{c}_{i}" in removed:
                a[f"P_{c}_{i}"] = 0.0
            elif m["cogen"]:
                a[f"P_{c}_{i}"] = min(fuel, grid)          # needs fuel AND grid
            else:
                a[f"P_{c}_{i}"] = fuel                     # self-gen: fuel only
    return a


def served_country(removed):
    a = availability(removed)
    out = {}
    for c in COUNTRIES:
        out[c] = float(sum(m["share"] * a[f"P_{c}_{i}"]
                           for i, m in enumerate(PLANT_META[c])))
    return out


def served_regional(removed):
    sc = served_country(removed)
    w = np.array([NATIONAL[c]["dependent_pop_M"] for c in COUNTRIES])
    s = np.array([sc[c] for c in COUNTRIES])
    return float((w * s).sum() / w.sum())


# Attack candidate set: structural nodes + each country's largest plants.
# (Attacking 70 tiny UAE plants one by one is not a realistic targeted strategy.)
def candidate_targets():
    cand = ["STRAIT"]
    for c in COUNTRIES:
        npl = len(PLANT_META[c])
        cand += [f"GRID_{c}", f"SCADA_{c}"] + [f"P_{c}_{i}" for i in range(min(2, npl))]
    return cand


# ----------------------------------------------------------------------------
# Analyses
# ----------------------------------------------------------------------------
def robustness_curves(max_steps=18, n_random=400):
    cand = candidate_targets()
    # greedy targeted
    removed, served_t = set(), [served_regional(set())]
    order_t = []
    for _ in range(min(max_steps, len(cand))):
        best, best_val = None, 1e9
        for nd in cand:
            if nd in removed:
                continue
            v = served_regional(removed | {nd})
            if v < best_val:
                best_val, best = v, nd
        removed.add(best); order_t.append(best); served_t.append(best_val)
    # random failure (average)
    steps = min(max_steps, len(cand))
    rand_mat = np.zeros((n_random, steps + 1))
    for r in range(n_random):
        perm = list(rng.permutation(cand))
        rem = set()
        rand_mat[r, 0] = served_regional(rem)
        for k in range(steps):
            rem.add(perm[k]); rand_mat[r, k + 1] = served_regional(rem)
    served_r = rand_mat.mean(axis=0)
    return served_t, served_r, order_t


def single_node_criticality():
    base = served_regional(set())
    rows = []
    for nd in candidate_targets():
        loss = base - served_regional({nd})
        ntype = ("Strait" if nd == "STRAIT" else
                 "Grid" if nd.startswith("GRID") else
                 "SCADA" if nd.startswith("SCADA") else "Plant")
        rows.append({"Node": nd, "Type": ntype,
                     "Regional demand lost": round(loss, 3)})
    return pd.DataFrame(rows).sort_values("Regional demand lost", ascending=False)


def country_network_vuln(threshold=0.5):
    """Continuous network-vulnerability score per country (mean national demand
    loss over single-node strikes on its candidate nodes), plus the structural
    minimum-node count for >50% national loss. The continuous score differentiates
    states (it reflects cogeneration coupling, Strait reliance, and plant
    concentration jointly); the discrete count is a structural headline."""
    rows = []
    base_c = served_country(set())
    for c in COUNTRIES:
        npl = len(PLANT_META[c])
        plants = [f"P_{c}_{i}" for i in range(min(3, npl))]
        cands = ["STRAIT", f"GRID_{c}", f"SCADA_{c}"] + plants
        losses = [base_c[c] - served_country({nd})[c] for nd in cands]
        score = float(np.mean(losses))
        worst = float(np.max(losses))
        # greedy minimum nodes to drop this country's served below threshold
        removed, k, cur = set(), 0, base_c[c]
        while cur >= threshold and k < len(cands):
            best, best_val = None, 1e9
            for nd in cands:
                if nd in removed:
                    continue
                v = served_country(removed | {nd})[c]
                if v < best_val:
                    best_val, best = v, nd
            if best is None or best_val >= cur - 1e-9:
                break
            removed.add(best); cur = best_val; k += 1
        min_nodes = k if cur < threshold else np.nan
        rows.append({"Country": c, "Net vuln score": round(score, 3),
                     "Worst single-node loss": round(worst, 3),
                     "Min nodes for >50% loss": min_nodes,
                     "Index rank": INDEX_RANK[c]})
    df = (pd.DataFrame(rows)
          .sort_values("Net vuln score", ascending=False).reset_index(drop=True))
    df["Network rank"] = np.arange(1, len(df) + 1)
    return df


def strait_demo():
    sc = served_country({"STRAIT"})
    rows = [{"Country": c, "Served if Strait removed": round(sc[c], 3),
             "Regional demand lost vs baseline": round(1 - sc[c], 3)}
            for c in COUNTRIES]
    df = pd.DataFrame(rows)
    return df, served_regional({"STRAIT"})


# ----------------------------------------------------------------------------
# Schematic diagram (aggregates plants per country for readability)
# ----------------------------------------------------------------------------
def draw_schematic():
    G = nx.DiGraph()
    G.add_node("STRAIT")
    for c in COUNTRIES:
        ab = c.split()[0][:3].upper()
        G.add_edge("STRAIT", f"PLANTS\n{ab}")    # distributed fuel (Strait-reliant share)
        G.add_edge(f"SCADA\n{ab}", f"GRID\n{ab}")
        G.add_edge(f"GRID\n{ab}", f"PLANTS\n{ab}")  # grid feeds cogeneration plants
        G.add_edge(f"PLANTS\n{ab}", f"DEMAND\n{ab}")
    layer = {}
    for n in G.nodes:
        layer[n] = (0 if n == "STRAIT" else 1 if n.startswith("SCADA")
                    else 2 if n.startswith("GRID") else 3 if n.startswith("PLANTS")
                    else 4)
    pos = {}
    from collections import defaultdict
    cols = defaultdict(list)
    for n, l in layer.items():
        cols[l].append(n)
    for l, ns in cols.items():
        for j, n in enumerate(sorted(ns)):
            pos[n] = (l, -(j - len(ns) / 2))
    colmap = {0: "#000000", 1: "#8c564b", 2: "#ff7f0e", 3: "#1f77b4", 4: "#2ca02c"}
    colors = [colmap[layer[n]] for n in G.nodes]
    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color=colors, font_size=6,
            node_size=900, font_color="white", arrows=True,
            edge_color="#999999", ax=ax)
    ax.set_title("Coupled water-energy dependency network (schematic; plants aggregated)")
    fig.tight_layout(); fig.savefig(OUTDIR / "network_schematic.png", dpi=150)
    plt.close(fig)


def plot_robustness(served_t, served_r):
    fig, ax = plt.subplots(figsize=(8, 5))
    k = np.arange(len(served_t))
    ax.plot(k, served_t, "o-", color="#d62728", label="targeted (greedy)")
    ax.plot(np.arange(len(served_r)), served_r, "s-", color="#1f77b4",
            label="random failure (mean)")
    ax.axhline(0.5, ls="--", color="grey", lw=1)
    ax.set_xlabel("nodes removed"); ax.set_ylabel("regional demand served (fraction)")
    ax.set_ylim(0, 1.02); ax.set_title("Network robustness: targeted vs random attack")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUTDIR / "robustness_curves.png", dpi=150)
    plt.close(fig)


def plot_criticality(df, top=10):
    d = df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(d["Node"], d["Regional demand lost"], color="#d62728")
    ax.set_xlabel("regional demand lost from removing this node alone")
    ax.set_title("Single-node criticality (top choke points)")
    fig.tight_layout(); fig.savefig(OUTDIR / "single_node_criticality.png", dpi=150)
    plt.close(fig)


def main():
    print("Layer 3A  Coupled-network cascade model\n" + "=" * 64)

    base = served_regional(set())
    print(f"Baseline regional demand served (no attack): {base:.3f}")

    crit = single_node_criticality()
    crit.to_csv(OUTDIR / "layer3_single_node_criticality.csv", index=False)
    print("\nSingle-node criticality (regional demand lost from removing one node):")
    print(crit.to_string(index=False))

    served_t, served_r, order_t = robustness_curves()
    pd.DataFrame({"nodes_removed": np.arange(len(served_t)),
                  "served_targeted": np.round(served_t, 3),
                  "served_random": np.round(served_r, 3)}
                 ).to_csv(OUTDIR / "layer3_robustness_curve.csv", index=False)
    crit_set = next((i for i, v in enumerate(served_t) if v < 0.5), None)
    print(f"\nTargeted attack order: {' -> '.join(order_t[:6])} ...")
    print(f"Minimum critical set (regional demand <50%): {crit_set} nodes")

    cvuln = country_network_vuln()
    cvuln.to_csv(OUTDIR / "layer3_country_network_vuln.csv", index=False)
    print("\nCountry vulnerability from the PHYSICAL model vs the composite index:")
    print(cvuln.to_string(index=False))
    # rank-correlation between the two orderings
    sub = cvuln.dropna(subset=["Min nodes for >50% loss"])
    if len(sub) >= 2:
        rho = np.corrcoef(sub["Network rank"], sub["Index rank"])[0, 1]
        print(f"  Network-rank vs index-rank correlation: {rho:.2f} "
              f"(agreement = triangulation)")

    sdf, sreg = strait_demo()
    sdf.to_csv(OUTDIR / "layer3_strait_cascade.csv", index=False)
    print(f"\nShared-upstream demonstration - remove the Strait node alone:")
    print(sdf.to_string(index=False))
    print(f"  Regional demand served with Strait removed: {sreg:.3f} "
          f"(loss {1-sreg:.3f}) from a SINGLE shared node")

    draw_schematic()
    plot_robustness(served_t, served_r)
    plot_criticality(crit)

    print(f"\nWrote CSVs and PNGs to ./{OUTDIR}/")
    print("CSVs: layer3_single_node_criticality.csv, layer3_robustness_curve.csv,")
    print("      layer3_country_network_vuln.csv, layer3_strait_cascade.csv")
    print("PNGs: network_schematic.png, robustness_curves.png,")
    print("      single_node_criticality.png")


if __name__ == "__main__":
    main()
