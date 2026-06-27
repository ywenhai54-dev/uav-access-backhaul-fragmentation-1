"""Simple baseline deployment methods for the quick-start demo."""

from __future__ import annotations

import random
from typing import List

from graph_builder import Scenario, distance


def random_baseline(scenario: Scenario, budget: int, seed: int = 7) -> List[int]:
    rng = random.Random(seed)
    return sorted(rng.sample(range(len(scenario.candidate_xy)), budget))


def coverage_first_baseline(
    scenario: Scenario,
    budget: int,
    access_radius_km: float = 2.0,
) -> List[int]:
    """Greedy baseline that selects candidates covering the most uncovered POIs."""
    selected: List[int] = []
    uncovered = set(range(len(scenario.poi_xy)))
    candidates = set(range(len(scenario.candidate_xy)))

    while len(selected) < budget and candidates:
        best = None
        best_gain = -1

        for j in candidates:
            covered = {
                i
                for i, p in enumerate(scenario.poi_xy)
                if i in uncovered and distance(p, scenario.candidate_xy[j]) <= access_radius_km
            }
            if len(covered) > best_gain:
                best = j
                best_gain = len(covered)

        if best is None:
            break

        selected.append(best)
        candidates.remove(best)
        uncovered -= {
            i
            for i, p in enumerate(scenario.poi_xy)
            if distance(p, scenario.candidate_xy[best]) <= access_radius_km
        }

    return sorted(selected)
