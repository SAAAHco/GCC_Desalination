# GCC_Desalination

Analysis code for the paper **"Engineering scarcity into vulnerability: desalination, water security, and the governance of conflict risk in the Gulf Cooperation Council"** (Ashkanani, Shuaibi and Albatayneh), submitted to *Journal of Hydrology: Regional Studies*.

The code quantifies the vulnerability of desalination-dependent water supply in the six Gulf Cooperation Council (GCC) states to interstate conflict. It takes the deterministic seven-category composite vulnerability index and the four disruption scenarios from the paper and extends them into a fully probabilistic assessment: a Monte Carlo composite index, a stochastic disruption-scenario engine, a coupled multi-layer dependency-network cascade model, and a Sobol global sensitivity analysis.

Every script is self-contained: the input values are embedded in clearly marked `DATA` blocks near the top of each file, sourced from the corrected data workbook and the Supplementary Information. No external data files are required to run the analysis.

All scripts use a fixed random seed (`SEED = 20260629`), so results are reproducible bit for bit and match the archived files in [`reference_outputs/`](reference_outputs/).

## Repository contents

| File | Paper element | Description |
|---|---|---|
| `gcc_layer1_composite_mc.py` | Fig. 1b; Supp. Note 2.4; Figs. S1 to S3 | Layer 1. Monte Carlo composite vulnerability index: score distributions, rank-probability matrix, tier-membership and pairwise-dominance probabilities. |
| `gcc_layer2_disruption_mc.py` | Fig. 5; Supp. Note 7; Figs. S4 to S6 | Layer 2. Stochastic disruption-scenario engine for scenarios S1 to S4 (single-plant strike, coordinated multi-plant, energy-grid cascade, contamination screening proxy). |
| `gcc_layer3_network_cascade.py` | Supp. Note 9; Figs. S7 to S9 | Layer 3A. Coupled multi-layer dependency network: robustness curves, single-node criticality, and a network-derived country ranking. |
| `gcc_layer3_sobol_sensitivity.py` | Supp. Note 9; Fig. S10 | Layer 3B. Sobol global sensitivity analysis ranking the unmeasured operational parameters by their contribution to disruption-loss variance. |
| `gcc_publication_figures.py` | Figs. 1b, 5c and SI heatmap | Renders selected figures as vector EPS (Elsevier preferred) plus PDF and 300 dpi PNG. |
| `run_all.py` | — | Convenience runner for the full pipeline. |
| `data/` | — | Machine-readable model input data (indicator distributions, plant and national parameters, weighting schemes); see `data/README.md`. |
| `reference_outputs/` | — | Archived CSV outputs the scripts reproduce, for verification. |

The scripts are independent and do not import one another; each can be run on its own.

## Requirements

- Python 3.10 or newer
- `numpy`, `pandas`, `matplotlib` (all scripts)
- `networkx` (Layer 3A only)
- `SALib` (Layer 3B only)

Install everything with:

```bash
pip install -r requirements.txt
```

## Usage

Run the whole pipeline:

```bash
python run_all.py
```

Or run a single layer:

```bash
python run_all.py --layer 1     # options: 1, 2, 3a, 3b, figs
```

Or run any script directly:

```bash
python gcc_layer1_composite_mc.py
python gcc_layer2_disruption_mc.py
python gcc_layer3_network_cascade.py
python gcc_layer3_sobol_sensitivity.py
python gcc_publication_figures.py
```

### Outputs

Each script writes to its own folder (created automatically):

- `layer1_outputs/` — `layer1_composite_summary.csv`, `layer1_rank_probability.csv`, `layer1_pairwise_dominance.csv`, and PNG figures.
- `layer2_outputs/` — `layer2_S1_single_plant.csv` through `layer2_S4_contamination.csv`, and PNG figures.
- `layer3_outputs/` — `layer3_single_node_criticality.csv`, `layer3_robustness_curve.csv`, `layer3_country_network_vuln.csv`, `layer3_strait_cascade.csv`, `layer3_sobol_indices.csv`, and PNG figures.
- Repository root — the vector publication figures (`Fig_1b_composite_distributions.*`, `SI_rank_probability_heatmap.*`, `Fig_5c_time_to_crisis.*`).

These generated folders are git-ignored; the committed copies for comparison are in `reference_outputs/`.

### Configuration

Run settings sit at the top of each script and can be changed without touching the engine, for example:

- `N_DRAWS` — Monte Carlo draws (default 10,000).
- `SEED` — random seed (change to test stability).
- `MODE` in Layer 1 — `"indicator"` (rebuilds category scores from raw indicators) or `"category"` (perturbs the published category scores; reproduces the paper's numbers exactly at zero noise).
- `WEIGHT_MODE` in Layer 1 — `"equal"`, `"dirichlet"`, or `"fixed"`.

Layer 1 also accepts environment-variable overrides, for example:

```bash
MODE=category CATEGORY_NOISE_SD=0 python gcc_layer1_composite_mc.py
```

## Reproducibility

With the default settings, the CSVs produced by each script are numerically identical to those in `reference_outputs/`. To verify Layer 1, for example:

```bash
python gcc_layer1_composite_mc.py
diff <(tr -d '\r' < layer1_outputs/layer1_composite_summary.csv) \
     <(tr -d '\r' < reference_outputs/layer1_composite_summary.csv)
```

## Data and modelling notes

- The models are deliberately transparent rather than black-box: they codify expert-informed, rule-based computations that can be audited line by line. No machine learning is used.
- Several operational parameters (inter-plant substitution, plant restart time, backup-power duration) are not published for GCC plants and are represented as distributions, not point values. These ranges are the honest part of the model and can be narrowed to your own engineering estimates. The Sobol analysis (Layer 3B) ranks which of them matters most.
- Scenario 4 (contamination) is a reduced-form screening proxy, not a hydrodynamic transport model.
- The network topology (Layer 3A) is built from aggregate national data, not a surveyed plant-to-substation map; its structure, rather than its exact parameters, is the contribution.

## Citation

If you use this code, please cite the article and the software (see `CITATION.cff`):

> Ashkanani, Z., Shuaibi, N., Albatayneh, R., 2026. Engineering scarcity into vulnerability: desalination, water security, and the governance of conflict risk in the Gulf Cooperation Council. *Journal of Hydrology*.

## License

Released under the MIT License (see `LICENSE`).

## Contact
Data and analysis code for the Journal of Hydrology submission. Uploaded by Dr. Zainab Ashkanani, Email: Ashkanani@saaah.co
