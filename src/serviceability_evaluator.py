"""Serviceability evaluation for the quick-start UAV-BS demo."""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List

from graph_builder import Scenario, distance, has_synthetic_los


def covered_pois(
    scenario: Scenario,
    selected: Iterable[int],
    access_radius_km: float,
) -> Dict[int, List[int]]:
    """Return candidate-to-POI coverage mapping."""
    selected = list(selected)
    mapping: Dict[int, List[int]] = {j: [] for j in selected}

    for p_idx, p in enumerate(scenario.poi_xy):
        for j in selected:
            if distance(p, scenario.candidate_xy[j]) <= access_radius_km:
                mapping[j].append(p_idx)

    return mapping


def backhaul_reachable_candidates(
    scenario: Scenario,
    selected: Iterable[int],
    backhaul_radius_km: float,
    max_hops: int,
) -> List[int]:
    """Return selected UAV candidates that can reach a GBS within hop limits."""
    selected = list(selected)
    adjacency: Dict[int, List[int]] = {j: [] for j in selected}
    reachable = set()

    for j in selected:
        cand = scenario.candidate_xy[j]
        for gbs in scenario.gbs_xy:
            if distance(cand, gbs) <= backhaul_radius_km and has_synthetic_los(cand, gbs):
                reachable.add(j)

    for a in selected:
        for b in selected:
            if a == b:
                continue
            pa = scenario.candidate_xy[a]
            pb = scenario.candidate_xy[b]
            if distance(pa, pb) <= backhaul_radius_km and has_synthetic_los(pa, pb):
                adjacency[a].append(b)

    q = deque([(j, 0) for j in reachable])
    visited = set(reachable)

    while q:
        current, hop = q.popleft()
        if hop >= max_hops:
            continue
        for nb in adjacency[current]:
            if nb not in visited:
                visited.add(nb)
                q.append((nb, hop + 1))

    return sorted(visited)


def evaluate_serviceability(
    scenario: Scenario,
    selected: Iterable[int],
    access_radius_km: float = 2.0,
    backhaul_radius_km: float = 4.5,
    max_hops: int = 2,
    capacity_per_gateway: int = 5,
) -> Dict[str, float]:
    """Evaluate coverage and capacity-constrained joint serviceability."""
    selected = list(selected)
    coverage = covered_pois(scenario, selected, access_radius_km)
    reachable = set(
        backhaul_reachable_candidates(
            scenario,
            selected,
            backhaul_radius_km=backhaul_radius_km,
            max_hops=max_hops,
        )
    )

    covered = set()
    serviceable = set()

    remaining_capacity = {j: capacity_per_gateway for j in reachable}

    for j, poi_list in coverage.items():
        covered.update(poi_list)
        if j not in reachable:
            continue
        for p_idx in poi_list:
            if remaining_capacity[j] > 0:
                serviceable.add(p_idx)
                remaining_capacity[j] -= 1

    n_poi = len(scenario.poi_xy)
    return {
        "selected_uavs": float(len(selected)),
        "poi_coverage": len(covered) / n_poi,
        "binary_joint_serviceability": len(
            {p for j, ps in coverage.items() if j in reachable for p in ps}
        ) / n_poi,
        "capacity_constrained_jswpr": len(serviceable) / n_poi,
        "backhaul_reachable_uav_ratio": len(reachable) / max(len(selected), 1),
    }
