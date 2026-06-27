"""Synthetic DEM/Fresnel-aware graph construction utilities.

This module provides a lightweight demo graph builder for UAV-BS deployment
experiments. It does not require external DEM files and is intended as a
minimal reproducible example for the repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, hypot
from typing import Dict, List, Tuple


Point = Tuple[float, float]


@dataclass
class Scenario:
    poi_xy: List[Point]
    candidate_xy: List[Point]
    gbs_xy: List[Point]
    elevation: Dict[str, float]


def terrain_height(x: float, y: float) -> float:
    """A small synthetic mountainous terrain surface."""
    ridge = 180.0 * exp(-((x - 4.8) ** 2) / 5.0)
    valley = -60.0 * exp(-((y - 5.2) ** 2) / 3.0)
    undulation = 35.0 * exp(-((x - 7.5) ** 2 + (y - 2.0) ** 2) / 4.0)
    return 500.0 + ridge + valley + undulation


def distance(a: Point, b: Point) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def build_demo_scenario() -> Scenario:
    """Build a compact synthetic scenario for quick-start experiments."""
    poi_xy = [
        (1.0, 1.2), (1.5, 2.0), (2.2, 3.0), (3.2, 4.2),
        (4.0, 5.2), (5.0, 6.0), (6.0, 6.8), (7.2, 7.5),
        (8.0, 6.2), (8.5, 4.5), (7.0, 3.2), (5.5, 2.4),
    ]

    candidate_xy = [
        (1.0, 1.0), (2.0, 2.5), (3.0, 4.0), (4.2, 5.5),
        (5.0, 4.0), (6.0, 6.0), (7.5, 7.0), (8.5, 5.0),
        (7.0, 2.5), (5.5, 1.5), (3.5, 1.5), (6.5, 4.5),
    ]

    gbs_xy = [(0.2, 0.5), (9.5, 8.8)]

    elevation = {}
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


def has_synthetic_los(a: Point, b: Point, clearance_m: float = 25.0) -> bool:
    """Approximate terrain line-of-sight check on the synthetic terrain.

    The endpoints are assumed to be UAV/GBS locations with additional antenna
    height. The sampled terrain between the endpoints should remain below the
    interpolated link height minus a clearance margin.
    """
    h_a = terrain_height(*a) + 300.0
    h_b = terrain_height(*b) + 30.0

    for k in range(1, 10):
        t = k / 10.0
        x = a[0] * (1 - t) + b[0] * t
        y = a[1] * (1 - t) + b[1] * t
        link_h = h_a * (1 - t) + h_b * t
        if terrain_height(x, y) + clearance_m > link_h:
            return False
    return True
