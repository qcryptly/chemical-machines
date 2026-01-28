"""
6-311G Basis Set Data

Triple-split valence basis set by Pople and coworkers.
Higher accuracy than 6-31G.

Reference:
- Krishnan, Binkley, Seeger, Pople, J. Chem. Phys. 72, 650 (1980)
"""

from typing import Dict
from copy import deepcopy


# Base 6-311G data for H-Ar
BASIS_6_311G_DATA = {
    'H': {
        '1s': [
            (0.0336866, 33.8650000),
            (0.1610429, 5.0947900),
            (0.4695300, 1.1587900),
        ],
        '2s': [
            (1.0, 0.3258400),
        ],
        '3s': [
            (1.0, 0.1027410),
        ],
    },
    'He': {
        '1s': [
            (0.0237660, 98.0783200),
            (0.1546790, 14.7644100),
            (0.4696300, 3.3185800),
        ],
        '2s': [
            (1.0, 0.8740470),
        ],
        '3s': [
            (1.0, 0.2445640),
        ],
    },
    'C': {
        '1s': [
            (0.0018347, 4563.2400000),
            (0.0140373, 682.0240000),
            (0.0688426, 154.9730000),
            (0.2321844, 44.4553000),
            (0.4679413, 13.0290000),
            (0.3623120, 1.8277300),
        ],
        '2s': [
            (1.0, 20.9642000),
        ],
        '3s': [
            (1.0, 4.8033100),
        ],
        '4s': [
            (1.0, 1.4593300),
        ],
        '2s_outer': [
            (-0.1193324, 0.4834560),
            (1.1434564, 0.1455850),
        ],
        '2s_outer2': [
            (1.0, 0.0438000),
        ],
        '2p': [
            (0.0689991, 20.9642000),
            (0.3164240, 4.8033100),
            (0.7443083, 1.4593300),
        ],
        '2p_outer': [
            (1.0, 0.4834560),
        ],
        '2p_outer2': [
            (1.0, 0.1455850),
        ],
        '2p_outer3': [
            (1.0, 0.0438000),
        ],
    },
    'N': {
        '1s': [
            (0.0019548, 6293.4800000),
            (0.0149209, 949.0440000),
            (0.0735882, 218.7760000),
            (0.2489020, 63.6916000),
            (0.4825720, 18.8282000),
            (0.3380840, 2.7202300),
        ],
        '2s': [
            (1.0, 30.6331000),
        ],
        '3s': [
            (1.0, 7.0261400),
        ],
        '4s': [
            (1.0, 2.1120500),
        ],
        '2s_outer': [
            (-0.1149610, 0.6840090),
            (1.1458520, 0.2008780),
        ],
        '2s_outer2': [
            (1.0, 0.0639000),
        ],
        '2p': [
            (0.0675800, 30.6331000),
            (0.3239070, 7.0261400),
            (0.7408950, 2.1120500),
        ],
        '2p_outer': [
            (1.0, 0.6840090),
        ],
        '2p_outer2': [
            (1.0, 0.2008780),
        ],
        '2p_outer3': [
            (1.0, 0.0639000),
        ],
    },
    'O': {
        '1s': [
            (0.0020300, 8588.5000000),
            (0.0154360, 1297.2300000),
            (0.0754740, 299.2960000),
            (0.2513620, 87.3771000),
            (0.4817530, 25.6789000),
            (0.3330910, 3.7400400),
        ],
        '2s': [
            (1.0, 42.1175000),
        ],
        '3s': [
            (1.0, 9.6283700),
        ],
        '4s': [
            (1.0, 2.8533200),
        ],
        '2s_outer': [
            (-0.1107775, 0.9056610),
            (1.1307670, 0.2556110),
        ],
        '2s_outer2': [
            (1.0, 0.0845000),
        ],
        '2p': [
            (0.0708743, 42.1175000),
            (0.3397528, 9.6283700),
            (0.7271586, 2.8533200),
        ],
        '2p_outer': [
            (1.0, 0.9056610),
        ],
        '2p_outer2': [
            (1.0, 0.2556110),
        ],
        '2p_outer3': [
            (1.0, 0.0845000),
        ],
    },
    'F': {
        '1s': [
            (0.0020960, 11427.1000000),
            (0.0159290, 1722.3500000),
            (0.0779050, 395.7460000),
            (0.2570960, 115.1390000),
            (0.4854010, 33.6026000),
            (0.3231700, 4.9190100),
        ],
        '2s': [
            (1.0, 55.4441000),
        ],
        '3s': [
            (1.0, 12.6323000),
        ],
        '4s': [
            (1.0, 3.7175600),
        ],
        '2s_outer': [
            (-0.1082040, 1.1654500),
            (1.1287400, 0.3218920),
        ],
        '2s_outer2': [
            (1.0, 0.1076000),
        ],
        '2p': [
            (0.0716167, 55.4441000),
            (0.3459121, 12.6323000),
            (0.7224699, 3.7175600),
        ],
        '2p_outer': [
            (1.0, 1.1654500),
        ],
        '2p_outer2': [
            (1.0, 0.3218920),
        ],
        '2p_outer3': [
            (1.0, 0.1076000),
        ],
    },
    'Ne': {
        '1s': [
            (0.0021830, 13995.7000000),
            (0.0165530, 2117.1000000),
            (0.0807990, 490.4250000),
            (0.2624380, 143.8330000),
            (0.4879570, 41.9265000),
            (0.3143530, 6.1560600),
        ],
        '2s': [
            (1.0, 69.1211000),
        ],
        '3s': [
            (1.0, 15.8351000),
        ],
        '4s': [
            (1.0, 4.6732600),
        ],
        '2s_outer': [
            (-0.1071830, 1.4575900),
            (1.1277740, 0.3978720),
        ],
        '2s_outer2': [
            (1.0, 0.1300000),
        ],
        '2p': [
            (0.0719095, 69.1211000),
            (0.3495133, 15.8351000),
            (0.7199405, 4.6732600),
        ],
        '2p_outer': [
            (1.0, 1.4575900),
        ],
        '2p_outer2': [
            (1.0, 0.3978720),
        ],
        '2p_outer3': [
            (1.0, 0.1300000),
        ],
    },
}


# Polarization d functions
POLARIZATION_D_6311G = {
    'C':  {'d': [(1.0, 0.626)]},
    'N':  {'d': [(1.0, 0.913)]},
    'O':  {'d': [(1.0, 1.292)]},
    'F':  {'d': [(1.0, 1.750)]},
    'Ne': {'d': [(1.0, 2.304)]},
}

# Polarization p functions for H
POLARIZATION_P_6311G = {
    'H':  {'p': [(1.0, 0.750)]},
    'He': {'p': [(1.0, 0.750)]},
}

# Diffuse functions
DIFFUSE_6311G = {
    'C':  {'s_diff': [(1.0, 0.0438)], 'p_diff': [(1.0, 0.0438)]},
    'N':  {'s_diff': [(1.0, 0.0639)], 'p_diff': [(1.0, 0.0639)]},
    'O':  {'s_diff': [(1.0, 0.0845)], 'p_diff': [(1.0, 0.0845)]},
    'F':  {'s_diff': [(1.0, 0.1076)], 'p_diff': [(1.0, 0.1076)]},
    'Ne': {'s_diff': [(1.0, 0.1300)], 'p_diff': [(1.0, 0.1300)]},
}

DIFFUSE_H_6311G = {
    'H':  {'s_diff': [(1.0, 0.0360)]},
    'He': {'s_diff': [(1.0, 0.0514)]},
}


def get_6_311g_basis(variant: str = '6-311G') -> Dict:
    """
    Get 6-311G basis set data with specified variant.

    Args:
        variant: One of '6-311G', '6-311G*', '6-311G**', '6-311+G*',
                 '6-311++G**', '6-311G(2d,2p)', '6-311+G(2d,2p)'

    Returns:
        Dictionary mapping elements to shell data
    """
    result = deepcopy(BASIS_6_311G_DATA)

    has_d_polar = '*' in variant or '(d' in variant.lower() or '(2d' in variant.lower()
    has_p_polar = '**' in variant or ',p)' in variant.lower() or '(2d,2p)' in variant.lower()
    has_diffuse_heavy = '+' in variant
    has_diffuse_h = '++' in variant
    has_2d = '(2d' in variant.lower()
    has_2p = '(2d,2p)' in variant.lower()

    # Add polarization d functions
    if has_d_polar:
        for elem, shells in POLARIZATION_D_6311G.items():
            if elem in result:
                if has_2d:
                    # Add two sets of d functions
                    d_exp = shells['d'][0][1]
                    result[elem]['d1'] = [(1.0, d_exp)]
                    result[elem]['d2'] = [(1.0, d_exp * 0.25)]
                else:
                    result[elem].update(shells)

    # Add polarization p functions on H
    if has_p_polar:
        for elem, shells in POLARIZATION_P_6311G.items():
            if elem in result:
                if has_2p:
                    p_exp = shells['p'][0][1]
                    result[elem]['p1'] = [(1.0, p_exp)]
                    result[elem]['p2'] = [(1.0, p_exp * 0.25)]
                else:
                    result[elem].update(shells)

    # Add diffuse functions on heavy atoms
    if has_diffuse_heavy:
        for elem, shells in DIFFUSE_6311G.items():
            if elem in result:
                result[elem].update(shells)

    # Add diffuse functions on H
    if has_diffuse_h:
        for elem, shells in DIFFUSE_H_6311G.items():
            if elem in result:
                result[elem].update(shells)

    return result
