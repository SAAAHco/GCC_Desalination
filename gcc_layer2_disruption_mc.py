#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer 2: Monte Carlo disruption-scenario engine (S1-S4)
=======================================================

Turns the four parameterized disruption scenarios into a stochastic model.
A small deterministic core computes, for a given strike:

    capacity lost      L  = sum of struck plants' shares of national capacity
    net supply deficit d  = max(0, L - substitution)
    time to crisis     T_crisis = reserve_days / d        (days)
        (simplifying assumption, stated openly: reserves defend FULL normal
         consumption until exhausted, before backup power and restart effects)
    population at risk     (S1: population served by the struck plant;
                            S2: dependent population scaled by the deficit)
    daily economic loss = full-outage daily loss x deficit
    cumulative loss        over a fixed horizon, for the days between reserve
                           exhaustion and either plant restart or horizon end
    population-days        million-person-days without safe water (-> health cost)

That core is wrapped in Monte Carlo over the parameters that are not precisely
known: reserve days, inter-plant substitution, plant restart time (conditioned
on weapon type: drones repair in days, missiles in months-years), the grid-
knockout fraction (S3), and Gulf current speed (S4). Outputs are distributions,
exceedance-probability curves, and tail probabilities (e.g. P(crisis onset
< 72 h), P(a k-plant attack exceeds 50% capacity loss)).

Scenarios
    S1  single-plant strike      -> ranks the worst single nodes (sharpens Al Dur)
    S2  coordinated multi-plant   -> P(>50% loss) and the "plants for >50%" distribution
    S3  energy-grid cascade       -> grid loss knocks out cogeneration-coupled desal;
                                     validated against the 12 March 2026 Kuwait event
    S4  seawater contamination    -> reduced-form SCREENING proxy (NOT a transport model):
                                     P(a release reaches an intake within a time window)

Requirements: numpy, pandas, matplotlib
Run:          python gcc_layer2_disruption_mc.py
Outputs:      ./layer2_outputs/  (CSVs + PNGs), plus a console summary.

Input values are from the corrected workbook ("Scenario Analysis Data",
"Economic & Stakes Data", "Vulnerability Indicators") and the Supplementary
Information. All values and distribution choices are editable in the DATA and
PARAMETER blocks below. The unknown-parameter ranges are the honest part of the
model; widen or narrow them to your own engineering estimates.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# RUN SETTINGS
# ----------------------------------------------------------------------------
N_DRAWS = 10_000
SEED = 20260629
HORIZON_DAYS = 365.0       # window over which cumulative loss is accumulated
CRISIS_HOURS = 72.0        # tail threshold for "rapid" crisis onset
rng = np.random.default_rng(SEED)

OUTDIR = Path("layer2_outputs")
OUTDIR.mkdir(exist_ok=True)

COUNTRIES = ["Bahrain", "Kuwait", "Qatar", "Oman", "UAE", "Saudi Arabia"]

# ============================================================================
# DATA BLOCK A  -  largest desalination plant per state
# (capacity m3/day, share of national capacity %, population served, millions)
# Source: workbook "Scenario Analysis Data"; shares derived from installed cap.
# To analyse more than the largest plant, add rows; S1 scans every row here.
# ============================================================================
PLANTS = [
    # country,        name,             cap_m3d,  share_pct, pop_served_M
    ("Bahrain",       "Al Dur",          409150,   47.8,      0.85),
    ("Kuwait",        "Az Zour North",   486000,   15.7,      1.70),
    ("Qatar",         "Ras Laffan B/C",  900000,   36.8,      1.80),
    ("Oman",          "Ghubrah 3",       281000,   16.8,      1.00),
    ("UAE",           "Taweelah",        909200,   12.0,      4.50),
    ("Saudi Arabia",  "Ras Al-Khair",   1036000,   18.5,      3.50),
]

# ============================================================================
# DATA BLOCK B  -  national parameters
# reserve_days     : (lo, hi) sampled uniform           [Supp Table 9 / Note 4.5]
# dependent_pop_M  : desal-dependent population, M       [Economic & Stakes sheet]
# daily_loss_full  : (lo, hi) $M/day at 100% outage      [Economic & Stakes sheet]
# health_cost      : $ per illness episode               [Economic & Stakes sheet]
# grid_knockout    : fraction of desal lost if the grid goes down (cogeneration
#                    coupling). Qatar/UAE/Oman known; others uniform(0.80,1.00)
#                    per Supp Note 4.4.
# n_plants         : number of major plants              [Vulnerability Indicators]
# largest_share    : largest plant % of national cap     (anchors the S2 size model)
# ============================================================================
NATIONAL = {
    "Bahrain":      dict(reserve_days=(1, 2),   dependent_pop_M=1.44, daily_loss_full=(15, 25),
                         health_cost=650, grid_knockout=("uni", 0.80, 1.00), n_plants=6,  largest_share=47.8),
    "Kuwait":       dict(reserve_days=(2, 4),   dependent_pop_M=4.59, daily_loss_full=(85, 120),
                         health_cost=720, grid_knockout=("uni", 0.80, 1.00), n_plants=7,  largest_share=15.7),
    "Qatar":        dict(reserve_days=(5, 9),   dependent_pop_M=3.17, daily_loss_full=(110, 160),
                         health_cost=850, grid_knockout=("pt", 1.00),        n_plants=7,  largest_share=36.8),
    "Oman":         dict(reserve_days=(5, 10),  dependent_pop_M=4.04, daily_loss_full=(35, 50),
                         health_cost=580, grid_knockout=("pt", 0.14),        n_plants=13, largest_share=16.8),
    "UAE":          dict(reserve_days=(80, 95), dependent_pop_M=4.86, daily_loss_full=(250, 380),
                         health_cost=820, grid_knockout=("pt", 0.81),        n_plants=70, largest_share=12.0),
    "Saudi Arabia": dict(reserve_days=(5, 10),  dependent_pop_M=25.9, daily_loss_full=(450, 650),
                         health_cost=750, grid_knockout=("uni", 0.80, 1.00), n_plants=33, largest_share=18.5),
}

# ============================================================================
# PARAMETER BLOCK  -  the unknowns the model is honest about.
# These are the highest-priority data targets named in the paper. Edit freely.
# ============================================================================
SUBSTITUTION = ("uni", 0.05, 0.30)        # fraction of lost capacity other plants cover
SUBSTITUTION_GRID = ("uni", 0.00, 0.10)   # smaller under a grid outage (others also coupled)
RESTART_DRONE_DAYS = ("tri", 3, 10, 30)         # drone damage: days to a few weeks
RESTART_MISSILE_DAYS = ("tri", 120, 365, 1825)  # missile damage: months to ~5 years
WEATHER_DISPERSION = ("uni", 1.0, 2.5)    # S4: spread beyond mean advection (tidal excursion)
CURRENT_SPEED_MS = ("uni", 0.05, 0.25)    # S4: Gulf residual current, m/s
ATTACK_RATE_PER_1000_PD = 0.5             # health: illness episodes per 1000 person-days

# ----------------------------------------------------------------------------
# sampler (same mini-language as Layer 1)
# ----------------------------------------------------------------------------
def draw(spec, n):
    k = spec[0]
    if k == "pt":   return np.full(n, float(spec[1]))
    if k == "uni":  return rng.uniform(spec[1], spec[2], n)
    if k == "tri":  return rng.triangular(spec[1], spec[2], spec[3], n)
    if k == "norm": return np.clip(rng.normal(spec[1], spec[2], n), 0, None)
    raise ValueError(k)


def crisis_metrics(L, sub, reserve_days, dep_pop_M, daily_loss_full,
                   restart_days, pop_local_M=None):
    """Vectorized deterministic core over arrays of length N.
    L           : fraction of national capacity lost (0-1)
    sub         : substitution fraction (0-1)
    pop_local_M : if given (S1), population at risk = this; else dep_pop_M * deficit.
    """
    d = np.clip(L - sub, 0.0, 1.0)                      # net deficit fraction
    has = d > 1e-9
    T_crisis = np.where(has, reserve_days / np.where(d > 0, d, 1.0), np.inf)
    pop_risk = pop_local_M if pop_local_M is not None else dep_pop_M * d
    pop_risk = np.where(has, pop_risk, 0.0)
    daily_loss = daily_loss_full * d                   # $M/day during crisis
    end = np.minimum(HORIZON_DAYS, restart_days)
    days_in_crisis = np.clip(end - T_crisis, 0.0, None)
    cum_loss = daily_loss * days_in_crisis             # $M over horizon
    pop_days = pop_risk * days_in_crisis               # million-person-days
    episodes = pop_days * 1000.0 * ATTACK_RATE_PER_1000_PD
    return dict(deficit=d, T_crisis=T_crisis, pop_risk=pop_risk,
                daily_loss=daily_loss, cum_loss=cum_loss, pop_days=pop_days,
                episodes=episodes)


def q(a, p):  # finite percentile helper (ignores inf time-to-crisis)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, p)) if a.size else float("nan")


# ============================================================================
# SCENARIO 1 : single-plant strike  (scan every plant in PLANTS)
# ============================================================================
def run_S1():
    rows = []
    for country, name, cap, share, popM in PLANTS:
        nat = NATIONAL[country]
        L = draw(("tri", share*0.9/100, share/100, share*1.1/100), N_DRAWS)
        sub = draw(SUBSTITUTION, N_DRAWS)
        R = draw(("uni",) + nat["reserve_days"], N_DRAWS)
        Floss = draw(("uni",) + nat["daily_loss_full"], N_DRAWS)
        poploc = draw(("tri", popM*0.85, popM, popM*1.15), N_DRAWS)
        m_drone = crisis_metrics(L, sub, R, nat["dependent_pop_M"], Floss,
                                 draw(RESTART_DRONE_DAYS, N_DRAWS), pop_local_M=poploc)
        m_missile = crisis_metrics(L, sub, R, nat["dependent_pop_M"], Floss,
                                   draw(RESTART_MISSILE_DAYS, N_DRAWS), pop_local_M=poploc)
        Tc = m_drone["T_crisis"]  # weapon-independent
        rows.append({
            "Country": country, "Plant": name, "Share %": share,
            "Pop at risk (M) med": round(q(m_drone["pop_risk"], 50), 2),
            "T_crisis days med": round(q(Tc, 50), 1),
            "T_crisis p2.5": round(q(Tc, 2.5), 1),
            "T_crisis p97.5": round(q(Tc, 97.5), 1),
            "P(crisis<72h)": round(float(np.mean(Tc < CRISIS_HOURS/24.0)), 3),
            "Cum loss $M med (drone)": round(q(m_drone["cum_loss"], 50), 0),
            "Cum loss $M med (missile)": round(q(m_missile["cum_loss"], 50), 0),
        })
    df = pd.DataFrame(rows).sort_values("Pop at risk (M) med", ascending=False)
    df.to_csv(OUTDIR / "layer2_S1_single_plant.csv", index=False)
    return df


# ============================================================================
# SCENARIO 2 : coordinated multi-plant disruption
# Plant-size model: top plant share fixed to the (sampled) largest share; the
# remaining capacity is split across n-1 plants by a sampled Zipf decay. This is
# a modeling proxy, since only the largest plant's share is in the source data.
# ============================================================================
def _plant_shares(country, n):
    """Return (N, n_plants) array of capacity shares summing to 1."""
    nat = NATIONAL[country]
    npl = nat["n_plants"]
    s1 = draw(("tri", nat["largest_share"]*0.9/100, nat["largest_share"]/100,
               nat["largest_share"]*1.1/100), n)                 # (N,)
    if npl <= 1:
        return s1[:, None]
    a = rng.uniform(0.6, 1.4, size=(n, 1))                       # Zipf steepness per draw
    ranks = np.arange(2, npl + 1)[None, :].astype(float)         # ranks 2..npl
    w = ranks ** (-a)
    w = w / w.sum(axis=1, keepdims=True)
    rest = (1.0 - s1)[:, None] * w
    return np.concatenate([s1[:, None], rest], axis=1)           # (N, npl)


def run_S2():
    rows = []
    kcurve = {}
    for country in COUNTRIES:
        nat = NATIONAL[country]
        npl = nat["n_plants"]
        shares = _plant_shares(country, N_DRAWS)
        shares_sorted = -np.sort(-shares, axis=1)                # descending
        cum = np.cumsum(shares_sorted, axis=1)                   # cumulative top-k loss
        kmax = min(5, npl)
        pk = [float(np.mean(cum[:, k-1] > 0.5)) for k in range(1, kmax + 1)]
        while len(pk) < 5:
            pk.append(float("nan"))
        kcurve[country] = pk
        reach = cum > 0.5
        first = np.where(reach.any(axis=1), np.argmax(reach, axis=1) + 1, npl)
        k_demo = min(3, npl)
        Ltop = cum[:, k_demo - 1]
        sub = draw(SUBSTITUTION, N_DRAWS)
        R = draw(("uni",) + nat["reserve_days"], N_DRAWS)
        Floss = draw(("uni",) + nat["daily_loss_full"], N_DRAWS)
        m = crisis_metrics(Ltop, sub, R, nat["dependent_pop_M"], Floss,
                           draw(RESTART_MISSILE_DAYS, N_DRAWS))
        rows.append({
            "Country": country, "N plants": npl,
            "P(>50% | top-1)": round(pk[0], 3),
            "P(>50% | top-2)": round(pk[1], 3),
            "P(>50% | top-3)": round(pk[2], 3),
            "Min plants for >50% (med)": int(np.median(first)),
            "Top-3 strike: pop risk (M) med": round(q(m["pop_risk"], 50), 2),
            "Top-3 strike: T_crisis days med": round(q(m["T_crisis"], 50), 1),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "layer2_S2_coordinated.csv", index=False)
    return df, kcurve


# ============================================================================
# SCENARIO 3 : energy-grid cascade
# A grid outage takes the cogeneration-coupled fraction of desal offline.
# ============================================================================
def run_S3():
    rows = []
    Tc_by_country = {}
    for country in COUNTRIES:
        nat = NATIONAL[country]
        g = draw(nat["grid_knockout"], N_DRAWS)
        sub = draw(SUBSTITUTION_GRID, N_DRAWS)
        R = draw(("uni",) + nat["reserve_days"], N_DRAWS)
        Floss = draw(("uni",) + nat["daily_loss_full"], N_DRAWS)
        m = crisis_metrics(g, sub, R, nat["dependent_pop_M"], Floss,
                           draw(RESTART_DRONE_DAYS, N_DRAWS))   # grid restored relatively fast
        Tc = m["T_crisis"]
        Tc_by_country[country] = Tc
        rows.append({
            "Country": country,
            "Grid knockout frac med": round(q(g, 50), 2),
            "Deficit med": round(q(m["deficit"], 50), 2),
            "T_crisis days med": round(q(Tc, 50), 1),
            "T_crisis p2.5": round(q(Tc, 2.5), 1),
            "T_crisis p97.5": round(q(Tc, 97.5), 1),
            "P(crisis<72h)": round(float(np.mean(Tc < CRISIS_HOURS/24.0)), 3),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "layer2_S3_grid_cascade.csv", index=False)
    return df, Tc_by_country


# ============================================================================
# SCENARIO 4 : seawater contamination  (SCREENING proxy, NOT a transport model)
# P(a release at a facility reaches a desal intake within a time window) modeled
# as advection: effective range = current_speed * window * dispersion factor.
# ============================================================================
PAIRS = [
    # label,                               distance_km   (source)
    ("Bapco Sitra -> Sitra plant",          20),   # Supp Note 7.3
    ("Mesaieed -> Ras Laffan B/C",          40),   # Supp Note 7.3
    ("Yanbu (co-located refinery+desal)",    5),   # Supp Note 7.3
    ("Strait -> Oman nearest intake",       40),   # Scenario sheet
    ("Strait -> UAE nearest intake",        75),
    ("Strait -> Qatar nearest intake",     510),
    ("Strait -> Bahrain nearest intake",   460),
    ("Strait -> Saudi nearest intake",     620),
    ("Strait -> Kuwait nearest intake",    840),
]


def run_S4():
    rows = []
    for label, dist in PAIRS:
        speed_kmd = draw(CURRENT_SPEED_MS, N_DRAWS) * 86.4       # m/s -> km/day
        disp = draw(WEATHER_DISPERSION, N_DRAWS)
        reach72 = (speed_kmd * disp * 3.0) >= dist               # within 72 h
        reach120 = (speed_kmd * disp * 5.0) >= dist              # within 120 h
        rows.append({
            "Facility -> intake": label,
            "Distance km": dist,
            "P(reach <72h)": round(float(np.mean(reach72)), 3),
            "P(reach <120h)": round(float(np.mean(reach120)), 3),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "layer2_S4_contamination.csv", index=False)
    return df


# ----------------------------------------------------------------------------
# PLOTS
# ----------------------------------------------------------------------------
def plot_S2_kcurve(kcurve):
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = np.arange(1, 6)
    for c in COUNTRIES:
        ax.plot(ks, kcurve[c], marker="o", label=c)
    ax.axhline(0.5, ls="--", color="grey", lw=1)
    ax.set_xlabel("number of largest plants struck (k)")
    ax.set_ylabel("P( >50% national capacity loss )")
    ax.set_title("S2: probability a coordinated k-plant strike crosses 50%")
    ax.set_xticks(ks); ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUTDIR / "S2_capacity_loss_curve.png", dpi=150)
    plt.close(fig)


def plot_S3_bar(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(COUNTRIES))
    vals = [df.set_index("Country").loc[c, "P(crisis<72h)"] for c in COUNTRIES]
    ax.bar(x, vals, color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(COUNTRIES, rotation=20, ha="right")
    ax.set_ylabel("P( crisis onset < 72 h )"); ax.set_ylim(0, 1)
    ax.set_title("S3: grid-cascade probability of rapid (<72h) water crisis")
    fig.tight_layout(); fig.savefig(OUTDIR / "S3_rapid_crisis_prob.png", dpi=150)
    plt.close(fig)


def plot_S3_exceedance(Tc_by_country):
    fig, ax = plt.subplots(figsize=(8, 5))
    grid = np.linspace(0, 30, 200)
    for c in COUNTRIES:
        tc = Tc_by_country[c]
        tc = tc[np.isfinite(tc)]
        if tc.size == 0:
            continue
        exceed = [np.mean(tc <= t) for t in grid]   # P(crisis by day t)
        ax.plot(grid, exceed, label=c)
    ax.set_xlabel("days after grid outage")
    ax.set_ylabel("P( reserves exhausted by this day )")
    ax.set_title("S3: time-to-crisis exceedance curves")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUTDIR / "S3_time_to_crisis_curve.png", dpi=150)
    plt.close(fig)


def plot_S4(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(df))
    ax.barh(y - 0.2, df["P(reach <72h)"], height=0.4, label="<72h", color="#1f77b4")
    ax.barh(y + 0.2, df["P(reach <120h)"], height=0.4, label="<120h", color="#9ecae1")
    ax.set_yticks(y); ax.set_yticklabels(df["Facility -> intake"], fontsize=8)
    ax.set_xlabel("P( plume reaches intake within window )")
    ax.set_title("S4: contamination screening proxy")
    ax.invert_yaxis(); ax.legend()
    fig.tight_layout(); fig.savefig(OUTDIR / "S4_contamination_screen.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    print(f"Layer 2 disruption-scenario Monte Carlo  |  draws={N_DRAWS}  "
          f"horizon={int(HORIZON_DAYS)}d\n" + "=" * 64)

    s1 = run_S1()
    print("\nS1  Single-plant strike (sorted by population at risk):")
    print(s1.to_string(index=False))

    s2, kcurve = run_S2()
    print("\nS2  Coordinated multi-plant disruption:")
    print(s2.to_string(index=False))

    s3, tc = run_S3()
    print("\nS3  Energy-grid cascade (validate against Kuwait, 12 Mar 2026):")
    print(s3.to_string(index=False))
    kuw = s3.set_index("Country").loc["Kuwait"]
    print(f"    -> Kuwait grid cascade: median T_crisis {kuw['T_crisis days med']} d, "
          f"P(crisis<72h) {kuw['P(crisis<72h)']}  "
          f"(observed 12 Mar: reserves ~3 d, desalination 'temporarily affected')")

    s4 = run_S4()
    print("\nS4  Seawater-contamination SCREENING proxy (not a transport model):")
    print(s4.to_string(index=False))

    plot_S2_kcurve(kcurve)
    plot_S3_bar(s3)
    plot_S3_exceedance(tc)
    plot_S4(s4)

    print(f"\nWrote CSVs and PNGs to ./{OUTDIR}/")
    print("CSVs: layer2_S1_single_plant.csv, layer2_S2_coordinated.csv,")
    print("      layer2_S3_grid_cascade.csv, layer2_S4_contamination.csv")
    print("PNGs: S2_capacity_loss_curve.png, S3_rapid_crisis_prob.png,")
    print("      S3_time_to_crisis_curve.png, S4_contamination_screen.png")


if __name__ == "__main__":
    main()
