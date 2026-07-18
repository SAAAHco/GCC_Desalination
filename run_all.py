#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the full GCC desalination vulnerability pipeline end to end.

This executes each analysis script in order and writes its CSVs and figures into
that script's own output folder (layer1_outputs/, layer2_outputs/, layer3_outputs/)
plus the vector publication figures in the repository root.

Every script is seeded (SEED = 20260629), so results are reproducible bit for bit
and match the files in reference_outputs/.

Usage:
    python run_all.py            # run everything
    python run_all.py --layer 1  # run a single layer (1, 2, 3a, 3b, or figs)

Requirements: see requirements.txt  (numpy, pandas, matplotlib, networkx, SALib).
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

STEPS = [
    ("1",   "Layer 1: composite vulnerability Monte Carlo",   "gcc_layer1_composite_mc.py"),
    ("2",   "Layer 2: disruption scenarios S1 to S4",          "gcc_layer2_disruption_mc.py"),
    ("3a",  "Layer 3A: coupled-network cascade",               "gcc_layer3_network_cascade.py"),
    ("3b",  "Layer 3B: Sobol global sensitivity",              "gcc_layer3_sobol_sensitivity.py"),
    ("figs", "Publication figures (EPS, PDF, PNG)",            "gcc_publication_figures.py"),
]


def run_step(script: str) -> int:
    print(f"\n{'=' * 70}\nRunning {script}\n{'=' * 70}")
    result = subprocess.run([sys.executable, str(HERE / script)], cwd=HERE)
    return result.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the GCC desalination analysis pipeline.")
    ap.add_argument("--layer", choices=[s[0] for s in STEPS],
                    help="Run only one step (default: run all).")
    args = ap.parse_args()

    steps = [s for s in STEPS if (args.layer is None or s[0] == args.layer)]
    failed = []
    for key, desc, script in steps:
        print(f"\n>>> {desc}")
        code = run_step(script)
        if code != 0:
            failed.append(script)
            print(f"!!! {script} exited with code {code}")

    print(f"\n{'=' * 70}")
    if failed:
        print("Finished with errors in:", ", ".join(failed))
        return 1
    print("All steps completed. Outputs are in layer1_outputs/, layer2_outputs/,")
    print("layer3_outputs/, and the publication figures in the repository root.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
