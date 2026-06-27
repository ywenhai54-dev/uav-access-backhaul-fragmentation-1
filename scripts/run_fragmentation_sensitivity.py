cat > scripts/run_fragmentation_sensitivity.py <<'EOF'
"""Access-backhaul fragmentation sensitivity experiment.

This script provides a longer runnable example for the repository
`uav-access-backhaul-fragmentation`.

It evaluates UAV-BS deployment under synthetic DEM/Fresnel-like terrain,
sparse surviving ground base stations, access coverage constraints,
backhaul reachability, and capacity-limited joint serviceability.

The experiment compares three simple deployment methods:

1. random
2. coverage_first
3. backhaul_aware

Outputs:
- outputs/fragmentation_sensitivity_results.csv
- outputs/fragmentation_sensitivity_summary.csv
- figures/fragmentation_sensitivity_jswpr.png
- figures/fragmentation_sensitivity_fragmentation.png
"""

from __future__ import annotations

import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from graph_builder import Scenario, distance, terrain_height  # noqa: E402


Point = Tuple[float, float]


@dataclass
class EvalResult:
    method: str
    seed: int
    capacity: int
    selected: List[int]
    poi_coverage: float
    binary_joint_serviceability: float
    capacity_constrained_jswpr: float
    fragmentation_ratio: float
    backhaul_reachable_uav_ratio: float


def build_fragmentation_scenario(seed: int = 0) -> Scenario:
    """Build a larger synthetic terrain-and-demand scenario.

    The scenario is intentionally designed to create access-backhaul mismatch:
    POIs are concentrated around a central valley and ridge area, whereas the
    surviving GBSs are located near two safe-area corners.
    """
    rng = random.Random(seed)

    poi_xy: List[Point] = []
    candidate_xy: List[Point] = []

    # Demand cluster 1: valley region.
    for _ in range(18):
        poi_xy.append(
            (
                rng.gauss(4.5, 1.0),
                rng.gauss(5.5, 0.8),
            )
        )

    # Demand cluster 2: ridge-side region.
    for _ in range(14):
        poi_xy.append(
            (
                rng.gauss(7.0, 0.8),
                rng.gauss(3.2, 0.9),
            )
        )

    # Background POIs.
    for _ in range(8):
        poi_xy.append((rng.uniform(1.0, 9.0), rng.uniform(1.0, 9.0)))

    # Candidate UAV locations on a coarse grid with random jitter.
    for x in [1.0, 2.5, 4.0, 5.5, 7.0, 8.5]:
        for y in [1.0, 2.7, 4.4, 6.1, 7.8]:
            candidate_xy.append(
                (
                    min(max(x + rng.uniform(-0.25, 0.25), 0.3), 9.7),
                    min(max(y + rng.uniform(-0.25, 0.25), 0.3), 9.7),
                )
            )

    # Sparse surviving ground base stations.
    gbs_xy = [(0.4, 0.6), (9.4, 8.7)]

    elevation: Dict[str, float] = {}
    for i, p in enumerate(poi_xy):
        elevation[f"poi_{i}"] = terrain_height(*p)
    for i, p in enumerate(candidate_xy):
        elevation[f"cand_{i}"] = terrain_height(*p)
    for i, p in enumerate(gbs_xy):
        elevation[f"gbs_{i}"] = terrain_height(*p)

    return Scenario(
        poi_xy=poi_xy,
        candidate_xy=candidate_xy,
        gbs_xy=gbs_xy,
        elevation=elevation,
    )


def synthetic_fresnel_clearance(a: Point, b: Point, receiver_is_gbs: bool) -> bool:
    """Approximate LoS/Fresnel clearance on the synthetic terrain.

    UAVs are assumed to fly 300 m above ground. GBSs are assumed to have
    30 m antenna height. The sampled terrain should stay below the
    interpolated link height with a clearance margin.
    """
    h_a = terrain_height(*a) + 300.0
    h_b = terrain_height(*b) + (30.0 if receiver_is_gbs else 300.0)

    d_km = distance(a, b)
    margin = 20.0 + 6.0 * d_km

    for k in range(1, 15):
        t = k / 15.0
        x = a[0] * (1.0 - t) + b[0] * t
        y = a[1] * (1.0 - t) + b[1] * t
        link_h = h_a * (1.0 - t) + h_b * t
        if terrain_height(x, y) + margin > link_h:
            return False

    return True


def get_covered_pois(
    scenario: Scenario,
    selected: Sequence[int],
    access_radius_km: float,
) -> Dict[int, Set[int]]:
    """Return UAV-candidate to covered-POI mapping."""
    mapping: Dict[int, Set[int]] = {j: set() for j in selected}

    for p_idx, poi in enumerate(scenario.poi_xy):
        for j in selected:
            cand = scenario.candidate_xy[j]
            if distance(poi, cand) <= access_radius_km:
                mapping[j].add(p_idx)

    return mapping


def get_backhaul_reachable(
    scenario: Scenario,
    selected: Sequence[int],
    backhaul_radius_km: float,
    max_relay_hops: int,
) -> Set[int]:
    """Find selected UAVs that can reach a surviving GBS with limited relays."""
    selected = list(selected)
    direct_reachable: Set[int] = set()
    adjacency: Dict[int, Set[int]] = {j: set() for j in selected}

    for j in selected:
        cand = scenario.candidate_xy[j]
        for gbs in scenario.gbs_xy:
            if distance(cand, gbs) <= backhaul_radius_km:
                if synthetic_fresnel_clearance(cand, gbs, receiver_is_gbs=True):
                    direct_reachable.add(j)
                    break

    for a in selected:
        for b in selected:
            if a == b:
                continue
            pa = scenario.candidate_xy[a]
            pb = scenario.candidate_xy[b]
            if distance(pa, pb) <= backhaul_radius_km:
                if synthetic_fresnel_clearance(pa, pb, receiver_is_gbs=False):
                    adjacency[a].add(b)

    reachable = set(direct_reachable)
    frontier = [(j, 0) for j in direct_reachable]

    while frontier:
        current, hop = frontier.pop(0)
        if hop >= max_relay_hops:
            continue
        for nb in adjacency[current]:
            if nb not in reachable:
                reachable.add(nb)
                frontier.append((nb, hop + 1))

    return reachable


def evaluate_deployment(
    scenario: Scenario,
    selected: Sequence[int],
    method: str,
    seed: int,
    capacity: int,
    access_radius_km: float = 1.8,
    backhaul_radius_km: float = 4.0,
    max_relay_hops: int = 2,
) -> EvalResult:
    """Evaluate access coverage, backhaul reachability, and JSWPR."""
    coverage = get_covered_pois(scenario, selected, access_radius_km)
    reachable = get_backhaul_reachable(
        scenario,
        selected,
        backhaul_radius_km=backhaul_radius_km,
        max_relay_hops=max_relay_hops,
    )

    covered_pois: Set[int] = set()
    binary_serviceable_pois: Set[int] = set()
    capacity_serviceable_pois: Set[int] = set()

    for poi_set in coverage.values():
        covered_pois.update(poi_set)

    remaining_capacity = {j: capacity for j in reachable}

    for j in selected:
        if j not in reachable:
            continue

        for poi_idx in sorted(coverage[j]):
            binary_serviceable_pois.add(poi_idx)
            if remaining_capacity[j] > 0:
                capacity_serviceable_pois.add(poi_idx)
                remaining_capacity[j] -= 1

    n_poi = len(scenario.poi_xy)
    poi_coverage = len(covered_pois) / n_poi
    binary_js = len(binary_serviceable_pois) / n_poi
    cap_jswpr = len(capacity_serviceable_pois) / n_poi

    if poi_coverage > 0:
        fragmentation_ratio = max(poi_coverage - cap_jswpr, 0.0) / poi_coverage
    else:
        fragmentation_ratio = 0.0

    return EvalResult(
        method=method,
        seed=seed,
        capacity=capacity,
        selected=list(selected),
        poi_coverage=poi_coverage,
        binary_joint_serviceability=binary_js,
        capacity_constrained_jswpr=cap_jswpr,
        fragmentation_ratio=fragmentation_ratio,
        backhaul_reachable_uav_ratio=len(reachable) / max(len(selected), 1),
    )


def random_baseline(
    scenario: Scenario,
    budget: int,
    rng: random.Random,
) -> List[int]:
    return sorted(rng.sample(range(len(scenario.candidate_xy)), budget))


def coverage_first_baseline(
    scenario: Scenario,
    budget: int,
    access_radius_km: float,
) -> List[int]:
    """Greedy selection that maximizes covered POIs."""
    selected: List[int] = []
    remaining_candidates = set(range(len(scenario.candidate_xy)))
    uncovered = set(range(len(scenario.poi_xy)))

    while len(selected) < budget and remaining_candidates:
        best = None
        best_gain = -1

        for j in remaining_candidates:
            cand = scenario.candidate_xy[j]
            gain = sum(
                1
                for p_idx in uncovered
                if distance(scenario.poi_xy[p_idx], cand) <= access_radius_km
            )
            if gain > best_gain:
                best = j
                best_gain = gain

        if best is None:
            break

        selected.append(best)
        remaining_candidates.remove(best)

        cand = scenario.candidate_xy[best]
        newly_covered = {
            p_idx
            for p_idx in uncovered
            if distance(scenario.poi_xy[p_idx], cand) <= access_radius_km
        }
        uncovered -= newly_covered

    return sorted(selected)


def backhaul_aware_baseline(
    scenario: Scenario,
    budget: int,
    access_radius_km: float,
    backhaul_radius_km: float,
) -> List[int]:
    """Greedy method balancing access coverage and backhaul proximity.

    This is not the proposed TG-RDD algorithm. It is only a transparent
    baseline for demonstrating the access-backhaul fragmentation effect.
    """
    selected: List[int] = []
    remaining = set(range(len(scenario.candidate_xy)))
    uncovered = set(range(len(scenario.poi_xy)))

    while len(selected) < budget and remaining:
        best = None
        best_score = -1e9

        for j in remaining:
            cand = scenario.candidate_xy[j]

            access_gain = sum(
                1
                for p_idx in uncovered
                if distance(scenario.poi_xy[p_idx], cand) <= access_radius_km
            )

            gbs_score = 0.0
            for gbs in scenario.gbs_xy:
                d = distance(cand, gbs)
                if d <= backhaul_radius_km:
                    los_bonus = 1.0 if synthetic_fresnel_clearance(cand, gbs, True) else 0.3
                    gbs_score = max(gbs_score, los_bonus * (1.0 - d / backhaul_radius_km))

            relay_score = 0.0
            for s in selected:
                ps = scenario.candidate_xy[s]
                d = distance(cand, ps)
                if d <= backhaul_radius_km:
                    relay_score = max(relay_score, 0.4 * (1.0 - d / backhaul_radius_km))

            score = access_gain + 3.0 * gbs_score + relay_score

            if score > best_score:
                best = j
                best_score = score

        if best is None:
            break

        selected.append(best)
        remaining.remove(best)

        cand = scenario.candidate_xy[best]
        newly_covered = {
            p_idx
            for p_idx in uncovered
            if distance(scenario.poi_xy[p_idx], cand) <= access_radius_km
        }
        uncovered -= newly_covered

    return sorted(selected)


def run_experiment() -> List[EvalResult]:
    seeds = [7, 13, 21, 34, 55]
    capacities = [2, 3, 4, 5, 6, 8]
    budget = 7
    access_radius_km = 1.8
    backhaul_radius_km = 4.0

    results: List[EvalResult] = []

    for seed in seeds:
        scenario = build_fragmentation_scenario(seed)
        rng = random.Random(seed)

        deployments = {
            "random": random_baseline(scenario, budget, rng),
            "coverage_first": coverage_first_baseline(
                scenario,
                budget,
                access_radius_km=access_radius_km,
            ),
            "backhaul_aware": backhaul_aware_baseline(
                scenario,
                budget,
                access_radius_km=access_radius_km,
                backhaul_radius_km=backhaul_radius_km,
            ),
        }

        for capacity in capacities:
            for method, selected in deployments.items():
                results.append(
                    evaluate_deployment(
                        scenario=scenario,
                        selected=selected,
                        method=method,
                        seed=seed,
                        capacity=capacity,
                        access_radius_km=access_radius_km,
                        backhaul_radius_km=backhaul_radius_km,
                        max_relay_hops=2,
                    )
                )

    return results


def save_results(results: Sequence[EvalResult]) -> None:
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    result_path = output_dir / "fragmentation_sensitivity_results.csv"

    with result_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "seed",
                "capacity",
                "selected",
                "poi_coverage",
                "binary_joint_serviceability",
                "capacity_constrained_jswpr",
                "fragmentation_ratio",
                "backhaul_reachable_uav_ratio",
            ]
        )

        for r in results:
            writer.writerow(
                [
                    r.method,
                    r.seed,
                    r.capacity,
                    " ".join(map(str, r.selected)),
                    f"{r.poi_coverage:.6f}",
                    f"{r.binary_joint_serviceability:.6f}",
                    f"{r.capacity_constrained_jswpr:.6f}",
                    f"{r.fragmentation_ratio:.6f}",
                    f"{r.backhaul_reachable_uav_ratio:.6f}",
                ]
            )

    summary = summarize_results(results)
    summary_path = output_dir / "fragmentation_sensitivity_summary.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "capacity",
                "mean_poi_coverage",
                "mean_binary_joint_serviceability",
                "mean_capacity_constrained_jswpr",
                "mean_fragmentation_ratio",
                "mean_backhaul_reachable_uav_ratio",
            ]
        )

        for row in summary:
            writer.writerow(row)

    print(f"Saved detailed results to: {result_path}")
    print(f"Saved summary results to: {summary_path}")


def summarize_results(results: Sequence[EvalResult]) -> List[List[object]]:
    grouped: Dict[Tuple[str, int], List[EvalResult]] = {}

    for r in results:
        grouped.setdefault((r.method, r.capacity), []).append(r)

    rows: List[List[object]] = []

    for method, capacity in sorted(grouped.keys()):
        items = grouped[(method, capacity)]

        def mean(values: Iterable[float]) -> float:
            values = list(values)
            return sum(values) / max(len(values), 1)

        rows.append(
            [
                method,
                capacity,
                f"{mean(r.poi_coverage for r in items):.6f}",
                f"{mean(r.binary_joint_serviceability for r in items):.6f}",
                f"{mean(r.capacity_constrained_jswpr for r in items):.6f}",
                f"{mean(r.fragmentation_ratio for r in items):.6f}",
                f"{mean(r.backhaul_reachable_uav_ratio for r in items):.6f}",
            ]
        )

    return rows


def plot_summary(results: Sequence[EvalResult]) -> None:
    figures_dir = ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)

    methods = ["random", "coverage_first", "backhaul_aware"]
    capacities = sorted({r.capacity for r in results})

    grouped: Dict[Tuple[str, int], List[EvalResult]] = {}
    for r in results:
        grouped.setdefault((r.method, r.capacity), []).append(r)

    def mean_metric(method: str, capacity: int, attr: str) -> float:
        items = grouped.get((method, capacity), [])
        if not items:
            return 0.0
        return sum(getattr(r, attr) for r in items) / len(items)

    plt.figure(figsize=(7.2, 4.2))
    for method in methods:
        y = [
            mean_metric(method, capacity, "capacity_constrained_jswpr")
            for capacity in capacities
        ]
        plt.plot(capacities, y, marker="o", label=method)

    plt.xlabel("Backhaul capacity per reachable UAV")
    plt.ylabel("Capacity-constrained JSWPR")
    plt.title("Capacity sensitivity of joint serviceability")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    jswpr_path = figures_dir / "fragmentation_sensitivity_jswpr.png"
    plt.savefig(jswpr_path, dpi=200)
    plt.close()

    plt.figure(figsize=(7.2, 4.2))
    for method in methods:
        y = [
            mean_metric(method, capacity, "fragmentation_ratio")
            for capacity in capacities
        ]
        plt.plot(capacities, y, marker="o", label=method)

    plt.xlabel("Backhaul capacity per reachable UAV")
    plt.ylabel("Access-backhaul fragmentation ratio")
    plt.title("Fragmentation decreases as backhaul capacity increases")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    frag_path = figures_dir / "fragmentation_sensitivity_fragmentation.png"
    plt.savefig(frag_path, dpi=200)
    plt.close()

    print(f"Saved JSWPR figure to: {jswpr_path}")
    print(f"Saved fragmentation figure to: {frag_path}")


def print_console_summary(results: Sequence[EvalResult]) -> None:
    print("\n=== Access-Backhaul Fragmentation Sensitivity Summary ===")
    print("method              capacity   coverage   binary_JS   cap_JSWPR   fragmentation")

    summary = summarize_results(results)
    for row in summary:
        method = str(row[0])
        capacity = int(row[1])
        coverage = float(row[2])
        binary_js = float(row[3])
        cap_jswpr = float(row[4])
        frag = float(row[5])
        print(
            f"{method:<18} {capacity:<8d} "
            f"{coverage:>8.3f}   {binary_js:>8.3f}   "
            f"{cap_jswpr:>8.3f}   {frag:>8.3f}"
        )


def main() -> None:
    results = run_experiment()
    save_results(results)
    plot_summary(results)
    print_console_summary(results)


if __name__ == "__main__":
    main()
EOF

python scripts/run_fragmentation_sensitivity.py

git add scripts/run_fragmentation_sensitivity.py outputs/fragmentation_sensitivity_results.csv outputs/fragmentation_sensitivity_summary.csv figures/fragmentation_sensitivity_jswpr.png figures/fragmentation_sensitivity_fragmentation.png
git commit -m "Add access-backhaul fragmentation sensitivity experiment"
git push
