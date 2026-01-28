"""
Lebedev Angular Integration Grids

Provides Lebedev-Laikov quadrature rules for integration over the unit sphere.
These grids are optimal for integrating spherical harmonics up to a given order.

Reference:
- Lebedev & Laikov, Doklady Mathematics 59, 477 (1999)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class LebedevGrid:
    """
    Lebedev angular quadrature grid on the unit sphere.

    Attributes:
        points: Unit sphere points (n, 3)
        weights: Quadrature weights (sum to 1, multiply by 4π for sphere)
        order: Grid order (determines accuracy)
        n_points: Number of angular points
        max_l: Maximum angular momentum exactly integrated
    """
    points: np.ndarray
    weights: np.ndarray
    order: int
    n_points: int
    max_l: int

    @classmethod
    def get(cls, order: int) -> 'LebedevGrid':
        """
        Get Lebedev grid of specified order.

        Args:
            order: Number of points (6, 14, 26, 38, 50, 74, 86, 110,
                   146, 170, 194, 230, 266, 302, 350, 434, 590, 770, 974, 1202)

        Returns:
            LebedevGrid with specified number of points
        """
        return get_lebedev_grid(order)


def get_lebedev_grid(n_points: int) -> LebedevGrid:
    """
    Get Lebedev grid with specified number of points.

    Args:
        n_points: Target number of points

    Returns:
        LebedevGrid closest to requested size
    """
    # Find closest available grid
    available = sorted(LEBEDEV_GRIDS.keys())
    if n_points in available:
        order = n_points
    else:
        # Find closest
        order = min(available, key=lambda x: abs(x - n_points))

    points, weights = _generate_lebedev_grid(order)

    # Maximum angular momentum exactly integrated
    # Lebedev-n integrates spherical harmonics exactly up to l_max
    l_max = {
        6: 3, 14: 5, 26: 7, 38: 9, 50: 11, 74: 13, 86: 15,
        110: 17, 146: 19, 170: 21, 194: 23, 230: 25, 266: 27,
        302: 29, 350: 31, 434: 35, 590: 41, 770: 47, 974: 53, 1202: 59
    }.get(order, (order // 6) * 2 - 1)

    return LebedevGrid(
        points=points,
        weights=weights,
        order=order,
        n_points=len(weights),
        max_l=l_max
    )


def _generate_lebedev_grid(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Lebedev grid from stored coefficients.

    The grids are constructed from symmetry-equivalent point sets
    with associated weights.
    """
    if order not in LEBEDEV_GRIDS:
        raise ValueError(f"Lebedev grid of order {order} not available. "
                        f"Available: {sorted(LEBEDEV_GRIDS.keys())}")

    grid_data = LEBEDEV_GRIDS[order]
    points = []
    weights = []

    for entry in grid_data:
        if entry['type'] == 'a1':
            # 6 points along axes: (±1, 0, 0), (0, ±1, 0), (0, 0, ±1)
            w = entry['weight']
            pts = [
                [1, 0, 0], [-1, 0, 0],
                [0, 1, 0], [0, -1, 0],
                [0, 0, 1], [0, 0, -1]
            ]
            points.extend(pts)
            weights.extend([w] * 6)

        elif entry['type'] == 'a2':
            # 12 points on edges: (±1/√2, ±1/√2, 0) and permutations
            w = entry['weight']
            c = 1.0 / np.sqrt(2)
            pts = []
            for i in range(3):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        pt = [0, 0, 0]
                        pt[i] = 0
                        pt[(i+1) % 3] = s1 * c
                        pt[(i+2) % 3] = s2 * c
                        pts.append(pt)
            points.extend(pts)
            weights.extend([w] * 12)

        elif entry['type'] == 'a3':
            # 8 points at cube vertices: (±1/√3, ±1/√3, ±1/√3)
            w = entry['weight']
            c = 1.0 / np.sqrt(3)
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    for s3 in [1, -1]:
                        points.append([s1 * c, s2 * c, s3 * c])
                        weights.append(w)

        elif entry['type'] == 'b':
            # 24 points: (±a, ±a, ±b) and permutations, a² + a² + b² = 1
            w = entry['weight']
            a = entry['a']
            b = np.sqrt(1 - 2 * a * a)
            _add_b_points(points, weights, a, b, w)

        elif entry['type'] == 'c':
            # 24 points: (±a, ±b, 0) and permutations, a² + b² = 1
            w = entry['weight']
            a = entry['a']
            b = np.sqrt(1 - a * a)
            _add_c_points(points, weights, a, b, w)

        elif entry['type'] == 'd':
            # 48 points: (±a, ±b, ±c) and permutations, a² + b² + c² = 1
            w = entry['weight']
            a = entry['a']
            b = entry['b']
            c = np.sqrt(1 - a * a - b * b)
            _add_d_points(points, weights, a, b, c, w)

    return np.array(points), np.array(weights)


def _add_b_points(points, weights, a, b, w):
    """Add 24 points of type B: (±a, ±a, ±b) and permutations."""
    vals = [a, a, b]
    for perm in [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]:
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    pt = [s1 * vals[perm[0]], s2 * vals[perm[1]], s3 * vals[perm[2]]]
                    if pt not in points:  # Avoid duplicates
                        points.append(pt)
                        weights.append(w)


def _add_c_points(points, weights, a, b, w):
    """Add 24 points of type C: (±a, ±b, 0) and permutations."""
    for i in range(3):  # Which coordinate is zero
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                pt = [0, 0, 0]
                pt[(i + 1) % 3] = s1 * a
                pt[(i + 2) % 3] = s2 * b
                points.append(pt)
                weights.append(w)
                if abs(a - b) > 1e-10:  # a != b, add swapped version
                    pt2 = [0, 0, 0]
                    pt2[(i + 1) % 3] = s1 * b
                    pt2[(i + 2) % 3] = s2 * a
                    points.append(pt2)
                    weights.append(w)


def _add_d_points(points, weights, a, b, c, w):
    """Add 48 points of type D: all permutations of (±a, ±b, ±c)."""
    from itertools import permutations
    vals = [a, b, c]
    seen = set()
    for perm in permutations([0, 1, 2]):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    pt = (s1 * vals[perm[0]], s2 * vals[perm[1]], s3 * vals[perm[2]])
                    if pt not in seen:
                        seen.add(pt)
                        points.append(list(pt))
                        weights.append(w)


# Lebedev grid data - coefficients for generating grids
# Format: list of {'type': 'a1'|'a2'|'a3'|'b'|'c'|'d', 'weight': w, 'a': a, 'b': b}
LEBEDEV_GRIDS: Dict[int, list] = {
    6: [
        {'type': 'a1', 'weight': 1/6}
    ],
    14: [
        {'type': 'a1', 'weight': 0.06666666666666667},
        {'type': 'a3', 'weight': 0.07500000000000000},
    ],
    26: [
        {'type': 'a1', 'weight': 0.04761904761904762},
        {'type': 'a2', 'weight': 0.03809523809523810},
        {'type': 'a3', 'weight': 0.03214285714285714},
    ],
    38: [
        {'type': 'a1', 'weight': 0.00952380952380952},
        {'type': 'a3', 'weight': 0.03214285714285714},
        {'type': 'c', 'weight': 0.02857142857142857, 'a': 0.4597008433809831},
    ],
    50: [
        {'type': 'a1', 'weight': 0.01269841269841270},
        {'type': 'a2', 'weight': 0.02257495590828924},
        {'type': 'a3', 'weight': 0.02109375000000000},
        {'type': 'c', 'weight': 0.02017333553791887, 'a': 0.3015113445777636},
    ],
    74: [
        {'type': 'a1', 'weight': 0.00579970134279684},
        {'type': 'a2', 'weight': 0.01436969559544024},
        {'type': 'a3', 'weight': 0.01375423871500000},
        {'type': 'c', 'weight': 0.01353066735500000, 'a': 0.4292963545341347},
        {'type': 'c', 'weight': 0.01213091990500000, 'a': 0.2355187894242326},
    ],
    86: [
        {'type': 'a1', 'weight': 0.01154401154401154},
        {'type': 'a3', 'weight': 0.01194390908585628},
        {'type': 'b', 'weight': 0.01111055571060340, 'a': 0.3696028464541502},
        {'type': 'c', 'weight': 0.01187650129453714, 'a': 0.6943540066026664},
        {'type': 'c', 'weight': 0.01181236421399854, 'a': 0.3742430390903412},
    ],
    110: [
        {'type': 'a1', 'weight': 0.00380952380952381},
        {'type': 'a3', 'weight': 0.00985028080856611},
        {'type': 'b', 'weight': 0.00918276471546936, 'a': 0.1851156353447362},
        {'type': 'b', 'weight': 0.00933885860800063, 'a': 0.6904210483822922},
        {'type': 'c', 'weight': 0.00909874167963458, 'a': 0.3956894730559419},
        {'type': 'c', 'weight': 0.00878227277688580, 'a': 0.4783690288121502},
    ],
    146: [
        {'type': 'a1', 'weight': 0.00536424908018576},
        {'type': 'a2', 'weight': 0.00698754108967792},
        {'type': 'a3', 'weight': 0.00671044842912880},
        {'type': 'b', 'weight': 0.00686153978993208, 'a': 0.2613931360335988},
        {'type': 'b', 'weight': 0.00686926883955792, 'a': 0.4691028277171857},
        {'type': 'c', 'weight': 0.00687006256077080, 'a': 0.6283974668416820},
        {'type': 'c', 'weight': 0.00654224927831368, 'a': 0.1882881429870860},
    ],
    170: [
        {'type': 'a1', 'weight': 0.00505462158347460},
        {'type': 'a2', 'weight': 0.00584979968761620},
        {'type': 'a3', 'weight': 0.00580345377858920},
        {'type': 'b', 'weight': 0.00590175479792880, 'a': 0.2551252621114134},
        {'type': 'b', 'weight': 0.00581845992420160, 'a': 0.6743601460362766},
        {'type': 'c', 'weight': 0.00573379180660880, 'a': 0.4318910696719410},
        {'type': 'c', 'weight': 0.00590883166767988, 'a': 0.2613931360335988},
        {'type': 'd', 'weight': 0.00574395329176120, 'a': 0.4990453161796037, 'b': 0.1446630744325115},
    ],
    194: [
        {'type': 'a1', 'weight': 0.00178234044724699},
        {'type': 'a2', 'weight': 0.00571467217168900},
        {'type': 'a3', 'weight': 0.00557549095447320},
        {'type': 'b', 'weight': 0.00515188316027640, 'a': 0.6712973442695226},
        {'type': 'b', 'weight': 0.00560258728628420, 'a': 0.2892465627575439},
        {'type': 'c', 'weight': 0.00560531964261720, 'a': 0.4989433179925530},
        {'type': 'c', 'weight': 0.00521825609152340, 'a': 0.6950514505463520},
        {'type': 'd', 'weight': 0.00521476632629760, 'a': 0.3698073787285940, 'b': 0.1059719355907480},
    ],
    302: [
        {'type': 'a1', 'weight': 0.00084095664672380},
        {'type': 'a3', 'weight': 0.00326554328814540},
        {'type': 'b', 'weight': 0.00326305772440120, 'a': 0.7039373391585475},
        {'type': 'b', 'weight': 0.00328233238016800, 'a': 0.1012526248572414},
        {'type': 'b', 'weight': 0.00303535361784640, 'a': 0.4647448726420539},
        {'type': 'c', 'weight': 0.00330263759468620, 'a': 0.3277841814844940},
        {'type': 'c', 'weight': 0.00321831006122380, 'a': 0.6620338663699974},
        {'type': 'c', 'weight': 0.00331226327027680, 'a': 0.4346575516141163},
        {'type': 'c', 'weight': 0.00316284460019640, 'a': 0.1757597956207540},
        {'type': 'd', 'weight': 0.00330000267100660, 'a': 0.2177055520503280, 'b': 0.6427665877145260},
        {'type': 'd', 'weight': 0.00330512618620660, 'a': 0.5765113834776660, 'b': 0.1818135037440880},
        {'type': 'd', 'weight': 0.00334633359989300, 'a': 0.4784765536291460, 'b': 0.3791920169680340},
    ],
    434: [
        {'type': 'a1', 'weight': 0.00053937095655330},
        {'type': 'a2', 'weight': 0.00223069570104100},
        {'type': 'a3', 'weight': 0.00223082796770860},
        {'type': 'b', 'weight': 0.00220097652150360, 'a': 0.0717480604924560},
        {'type': 'b', 'weight': 0.00227079842935100, 'a': 0.2439595712529290},
        {'type': 'b', 'weight': 0.00229859436939960, 'a': 0.4212286680682460},
        {'type': 'b', 'weight': 0.00226922920632700, 'a': 0.5755289211630500},
        {'type': 'b', 'weight': 0.00228972313400500, 'a': 0.6981269949204000},
        {'type': 'c', 'weight': 0.00231047596252000, 'a': 0.1721420832906236},
        {'type': 'c', 'weight': 0.00230691175760040, 'a': 0.3586067974412447},
        {'type': 'c', 'weight': 0.00231050505555400, 'a': 0.5259054679112630},
        {'type': 'c', 'weight': 0.00227891646880980, 'a': 0.6596711812777400},
        {'type': 'd', 'weight': 0.00232209614927200, 'a': 0.2281538668143580, 'b': 0.4390350497559280},
        {'type': 'd', 'weight': 0.00231629240890120, 'a': 0.2280199395998220, 'b': 0.6299207919209840},
        {'type': 'd', 'weight': 0.00229949015519740, 'a': 0.3864787505128230, 'b': 0.5665217126424680},
        {'type': 'd', 'weight': 0.00232049192906460, 'a': 0.4804458051621620, 'b': 0.2079569146920200},
        {'type': 'd', 'weight': 0.00231555875980340, 'a': 0.0911424990268880, 'b': 0.6006206612893860},
    ],
    590: [
        {'type': 'a1', 'weight': 0.00032038212776320},
        {'type': 'a3', 'weight': 0.00167691245328900},
        {'type': 'b', 'weight': 0.00167570714750400, 'a': 0.0524025318718230},
        {'type': 'b', 'weight': 0.00173948820400040, 'a': 0.1859037813766460},
        {'type': 'b', 'weight': 0.00175983795276980, 'a': 0.3237839905017080},
        {'type': 'b', 'weight': 0.00175895399398940, 'a': 0.4597553831559500},
        {'type': 'b', 'weight': 0.00173917260098600, 'a': 0.5824053678398620},
        {'type': 'b', 'weight': 0.00168878953667160, 'a': 0.6821652605135720},
        {'type': 'b', 'weight': 0.00161689228210060, 'a': 0.7555801013773660},
        {'type': 'c', 'weight': 0.00171570970879200, 'a': 0.1271655513575900},
        {'type': 'c', 'weight': 0.00175282917866460, 'a': 0.2697155605869580},
        {'type': 'c', 'weight': 0.00176279103915260, 'a': 0.4049481791584100},
        {'type': 'c', 'weight': 0.00175598044612240, 'a': 0.5285829063636020},
        {'type': 'c', 'weight': 0.00173103767561080, 'a': 0.6369070346063380},
        {'type': 'c', 'weight': 0.00168667063534580, 'a': 0.7250486314063040},
        {'type': 'd', 'weight': 0.00175916688166200, 'a': 0.1738828954934040, 'b': 0.5040032611780940},
        {'type': 'd', 'weight': 0.00173866685291220, 'a': 0.1669686308796140, 'b': 0.6684037259052120},
        {'type': 'd', 'weight': 0.00175689951780640, 'a': 0.3015858595783260, 'b': 0.4229209234674300},
        {'type': 'd', 'weight': 0.00175535155093880, 'a': 0.2968612680545760, 'b': 0.5877927285213920},
        {'type': 'd', 'weight': 0.00175505629720040, 'a': 0.4093893416122720, 'b': 0.3555282178594880},
        {'type': 'd', 'weight': 0.00175436891605740, 'a': 0.4051086458975200, 'b': 0.5195698305285700},
        {'type': 'd', 'weight': 0.00175222946951800, 'a': 0.5027308395892380, 'b': 0.2981099509929540},
        {'type': 'd', 'weight': 0.00175133422734120, 'a': 0.4985614019619420, 'b': 0.4555017313498840},
        {'type': 'd', 'weight': 0.00174898946247020, 'a': 0.5849265509067260, 'b': 0.2505917035893680},
        {'type': 'd', 'weight': 0.00174796831839740, 'a': 0.5806780936093940, 'b': 0.3995768100746120},
        {'type': 'd', 'weight': 0.00174541619480200, 'a': 0.6530767397198680, 'b': 0.2105665073509680},
        {'type': 'd', 'weight': 0.00174324627971740, 'a': 0.6484696298628060, 'b': 0.3504584858813840},
    ],
}
