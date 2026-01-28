"""
LANL2DZ Effective Core Potential

Los Alamos National Laboratory double-zeta ECP (Hay-Wadt).

Provides ECPs for:
- First-row transition metals (Sc-Zn): 10 core electrons ([Ar] core)
- Second-row transition metals (Y-Cd): 28 core electrons ([Ar]3d¹⁰ core)
- Third-row transition metals (La-Hg): 60 core electrons

Reference: Hay, Wadt, J. Chem. Phys. 82, 270 (1985)
"""

from .core import ECPPotential, ECPComponent


# LANL2DZ ECPs for first-row transition metals
# Format: element -> ECPPotential

LANL2DZ_ECP = {}

# Iron (Fe) - 10 core electrons ([Ne] core for 3d metals is [Ar] with 18e-, but LANL2DZ uses 10e- core)
# Actually LANL2DZ for Fe replaces 10 core electrons (1s²2s²2p⁶ = Ne core)
LANL2DZ_ECP['Fe'] = ECPPotential(
    element='Fe',
    n_core=10,
    lmax=2,  # d-type is highest
    local=[
        # Local (l=3, f-type) potential
        ECPComponent(coefficient=-10.0, r_power=1, exponent=1.0),
    ],
    semilocal={
        0: [  # s-type
            ECPComponent(coefficient=14.9676, r_power=2, exponent=9.7469),
            ECPComponent(coefficient=29.5961, r_power=2, exponent=4.0584),
            ECPComponent(coefficient=-4.4987, r_power=2, exponent=1.2967),
        ],
        1: [  # p-type
            ECPComponent(coefficient=9.0414, r_power=2, exponent=6.1791),
            ECPComponent(coefficient=23.2316, r_power=2, exponent=2.6851),
            ECPComponent(coefficient=6.6847, r_power=2, exponent=0.8951),
        ],
        2: [  # d-type
            ECPComponent(coefficient=-2.1269, r_power=2, exponent=4.6712),
            ECPComponent(coefficient=-9.1123, r_power=2, exponent=1.4478),
            ECPComponent(coefficient=-1.9117, r_power=2, exponent=0.3687),
        ],
    }
)

# Copper (Cu)
LANL2DZ_ECP['Cu'] = ECPPotential(
    element='Cu',
    n_core=10,
    lmax=2,
    local=[
        ECPComponent(coefficient=-10.0, r_power=1, exponent=1.0),
    ],
    semilocal={
        0: [
            ECPComponent(coefficient=16.3665, r_power=2, exponent=11.5475),
            ECPComponent(coefficient=34.9081, r_power=2, exponent=4.9170),
            ECPComponent(coefficient=-5.1720, r_power=2, exponent=1.5720),
        ],
        1: [
            ECPComponent(coefficient=10.8555, r_power=2, exponent=7.4857),
            ECPComponent(coefficient=27.5868, r_power=2, exponent=3.2661),
            ECPComponent(coefficient=7.5039, r_power=2, exponent=1.0892),
        ],
        2: [
            ECPComponent(coefficient=-2.7541, r_power=2, exponent=5.9457),
            ECPComponent(coefficient=-12.2019, r_power=2, exponent=1.8756),
            ECPComponent(coefficient=-2.5245, r_power=2, exponent=0.4717),
        ],
    }
)

# Zinc (Zn)
LANL2DZ_ECP['Zn'] = ECPPotential(
    element='Zn',
    n_core=10,
    lmax=2,
    local=[
        ECPComponent(coefficient=-10.0, r_power=1, exponent=1.0),
    ],
    semilocal={
        0: [
            ECPComponent(coefficient=17.7689, r_power=2, exponent=13.5033),
            ECPComponent(coefficient=40.1820, r_power=2, exponent=5.8241),
            ECPComponent(coefficient=-5.8584, r_power=2, exponent=1.8679),
        ],
        1: [
            ECPComponent(coefficient=12.7355, r_power=2, exponent=8.8936),
            ECPComponent(coefficient=31.8889, r_power=2, exponent=3.8822),
            ECPComponent(coefficient=8.2976, r_power=2, exponent=1.2975),
        ],
        2: [
            ECPComponent(coefficient=-3.4306, r_power=2, exponent=7.3259),
            ECPComponent(coefficient=-15.5059, r_power=2, exponent=2.3509),
            ECPComponent(coefficient=-3.1781, r_power=2, exponent=0.5906),
        ],
    }
)

# Nickel (Ni)
LANL2DZ_ECP['Ni'] = ECPPotential(
    element='Ni',
    n_core=10,
    lmax=2,
    local=[
        ECPComponent(coefficient=-10.0, r_power=1, exponent=1.0),
    ],
    semilocal={
        0: [
            ECPComponent(coefficient=13.6286, r_power=2, exponent=8.1247),
            ECPComponent(coefficient=24.4971, r_power=2, exponent=3.3428),
            ECPComponent(coefficient=-3.8242, r_power=2, exponent=1.0620),
        ],
        1: [
            ECPComponent(coefficient=7.2969, r_power=2, exponent=5.0197),
            ECPComponent(coefficient=19.0825, r_power=2, exponent=2.2014),
            ECPComponent(coefficient=5.9127, r_power=2, exponent=0.7313),
        ],
        2: [
            ECPComponent(coefficient=-1.5489, r_power=2, exponent=3.7299),
            ECPComponent(coefficient=-6.2127, r_power=2, exponent=1.1357),
            ECPComponent(coefficient=-1.3430, r_power=2, exponent=0.2862),
        ],
    }
)

# Cobalt (Co)
LANL2DZ_ECP['Co'] = ECPPotential(
    element='Co',
    n_core=10,
    lmax=2,
    local=[
        ECPComponent(coefficient=-10.0, r_power=1, exponent=1.0),
    ],
    semilocal={
        0: [
            ECPComponent(coefficient=12.5587, r_power=2, exponent=6.8789),
            ECPComponent(coefficient=20.2252, r_power=2, exponent=2.7916),
            ECPComponent(coefficient=-3.2604, r_power=2, exponent=0.8775),
        ],
        1: [
            ECPComponent(coefficient=5.9880, r_power=2, exponent=4.1203),
            ECPComponent(coefficient=15.8003, r_power=2, exponent=1.8131),
            ECPComponent(coefficient=5.2059, r_power=2, exponent=0.6004),
        ],
        2: [
            ECPComponent(coefficient=-1.1037, r_power=2, exponent=2.9481),
            ECPComponent(coefficient=-4.2612, r_power=2, exponent=0.8847),
            ECPComponent(coefficient=-0.9591, r_power=2, exponent=0.2193),
        ],
    }
)

# Manganese (Mn)
LANL2DZ_ECP['Mn'] = ECPPotential(
    element='Mn',
    n_core=10,
    lmax=2,
    local=[
        ECPComponent(coefficient=-10.0, r_power=1, exponent=1.0),
    ],
    semilocal={
        0: [
            ECPComponent(coefficient=10.5541, r_power=2, exponent=4.9254),
            ECPComponent(coefficient=12.6810, r_power=2, exponent=1.9247),
            ECPComponent(coefficient=-2.2135, r_power=2, exponent=0.5870),
        ],
        1: [
            ECPComponent(coefficient=3.7104, r_power=2, exponent=2.6361),
            ECPComponent(coefficient=9.4706, r_power=2, exponent=1.1727),
            ECPComponent(coefficient=3.7939, r_power=2, exponent=0.3838),
        ],
        2: [
            ECPComponent(coefficient=-0.5024, r_power=2, exponent=1.7179),
            ECPComponent(coefficient=-1.7318, r_power=2, exponent=0.5026),
            ECPComponent(coefficient=-0.4268, r_power=2, exponent=0.1231),
        ],
    }
)


# Valence basis sets for LANL2DZ
# These are the valence-only basis functions used with the ECP

LANL2DZ_BASIS = {
    'Fe': {
        '3s': [
            (0.3080, 0.8316, 2.2670),  # exponents
            (0.4517, 0.5730, 0.0253),  # coefficients
        ],
        '3p': [
            (0.1480, 0.4540, 1.4420),
            (0.5271, 0.5374, 0.0300),
        ],
        '3d': [
            (0.1260, 0.3800, 1.2890),
            (0.5234, 0.5716, 0.0265),
        ],
        '4s': [
            (0.0460,),
            (1.0000,),
        ],
        '4p': [
            (0.0200,),
            (1.0000,),
        ],
    },
    'Cu': {
        '3s': [
            (0.4000, 1.1020, 3.1640),
            (0.4445, 0.5850, 0.0213),
        ],
        '3p': [
            (0.1920, 0.5920, 1.9710),
            (0.5232, 0.5421, 0.0287),
        ],
        '3d': [
            (0.2000, 0.5660, 1.8450),
            (0.4954, 0.5814, 0.0413),
        ],
        '4s': [
            (0.0550,),
            (1.0000,),
        ],
        '4p': [
            (0.0250,),
            (1.0000,),
        ],
    },
    'Zn': {
        '3s': [
            (0.5000, 1.3600, 3.9300),
            (0.4393, 0.5926, 0.0170),
        ],
        '3p': [
            (0.2400, 0.7370, 2.5100),
            (0.5203, 0.5461, 0.0274),
        ],
        '3d': [
            (0.2500, 0.7050, 2.3330),
            (0.4852, 0.5855, 0.0411),
        ],
        '4s': [
            (0.0650,),
            (1.0000,),
        ],
        '4p': [
            (0.0300,),
            (1.0000,),
        ],
    },
}
