# Data

Machine-readable input data for the analysis, extracted from the `DATA` blocks
embedded in the scripts. The scripts remain the executable source of truth; these
CSVs mirror them one to one so the inputs can be inspected without reading code.

| File | Contents | Used by |
|---|---|---|
| `indicator_distributions.csv` | All 35 vulnerability indicators: category, direction of vulnerability, log-transform flag, and the per-country probability distribution (type and parameters) | Layer 1 |
| `category_scores.csv` | Published seven-category scores per state (Supplementary Table 8 Part A; 1 = most vulnerable) | Layer 1, figures |
| `weighting_schemes.csv` | The four category-weight vectors: equal, structural-emphasis, population-emphasis, energy-emphasis | Layer 1 |
| `plants_largest.csv` | Largest desalination plant per state: capacity, share of national capacity, population served | Layer 2 |
| `national_scenario_parameters.csv` | Per-state scenario inputs: reserve-day ranges, dependent population, full-outage daily loss range, health cost, grid-knockout distribution, plant count, largest-plant share | Layer 2 |
| `operational_unknowns.csv` | Distribution ranges for the unmeasured operational parameters (substitution, restart times, current speed, dispersion) | Layers 2 and 3B |
| `network_parameters.csv` | Per-state network-model inputs: cogeneration fraction, Strait reliance, dual-coast split | Layer 3A |
| `contamination_pairs.csv` | Facility-to-intake distances for the Scenario 4 screening proxy | Layer 2 |
| `conflict_chronology_core_26_events.csv` | The 26-event March 2026 conflict chronology (2 to 19 March): date, country, target type, facility, attack method, damage, water impact, source key | Paper Supplementary Table 1 |
| `conflict_chronology_additional_event.csv` | The confirmed 29 to 30 March 2026 Doha West strike (event 27) | Paper Supplementary Table 2 |
| `conflict_timeline_sources.csv` | Full reference, including URL, for every source key cited in the chronology | Chronology provenance |
| `vulnerability_indicator_matrix_headline.csv` | Headline values of the cross-GCC vulnerability indicator matrix | Paper Supplementary Table 3 |
| `global_benchmarks.csv` | GCC average versus Israel, Singapore, Australia, Spain, and Algeria | Paper Supplementary Table 4 |
| `resilience_pathways_matrix.csv` | Five resilience layers across the six states | Paper Supplementary Table 5 |
| `economic_and_stakes_data.csv` | Unit costs and cost comparisons underlying the avoided-cost analysis | Paper Supplementary Table 6 |
| `data_audit_log.csv` | The 12 data-quality corrections applied during data preparation, by severity | Paper Supplementary Table 7 |

Distribution mini-language (`dist_type`): `pt` point mass (param1 = value);
`uni` uniform (param1 = low, param2 = high); `tri` triangular (param1 = low,
param2 = mode, param3 = high); `rel` triangular at value v with relative spread r,
i.e. tri(v(1-r), v, v(1+r)) (param1 = v, param2 = r); `norm` normal clipped at 0
(param1 = mean, param2 = sd).

Derived results reproduced by the scripts are archived in `../reference_outputs/`.

Value provenance: the corrected project data workbook and the Supplementary
Information of the paper (GWI DesalData 2023; IDA 2023; FAO AQUASTAT 2023;
GCC-Stat 2023, 2024; IRENA 2023; national utility publications; full source
mapping in Supplementary Table 7 of the paper).
