"""Quick-start example for UAV access-backhaul serviceability evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from baselines import coverage_first_baseline, random_baseline
from graph_builder import build_demo_scenario
from serviceability_evaluator import evaluate_serviceability


def main() -> None:
    scenario = build_demo_scenario()
    budget = 4

    methods = {
        "random": random_baseline(scenario, budget=budget, seed=7),
        "coverage_first": coverage_first_baseline(scenario, budget=budget),
    }

    for name, selected in methods.items():
        result = evaluate_serviceability(scenario, selected)
        print(f"\nMethod: {name}")
        print(f"Selected UAV candidates: {selected}")
        for key, value in result.items():
            print(f"{key}: {value:.3f}")


if __name__ == "__main__":
    main()
