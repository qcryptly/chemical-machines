# %% Introduction
from cm.views import html

html("""
<h2>Tutorial 7: Electric Dipole Moments</h2>
<p>The electric dipole moment measures the separation of positive and negative
charge in a molecule. It is one of the most fundamental molecular properties,
governing intermolecular interactions, solubility, and spectroscopic selection
rules.</p>
<p>The dipole moment has nuclear and electronic contributions:</p>
<p style="text-align:center">&mu; = &mu;<sub>nuc</sub> + &mu;<sub>elec</sub>
= &Sigma;<sub>A</sub> Z<sub>A</sub> R<sub>A</sub> &minus; Tr[P D]</p>
<p>where P is the density matrix and D contains the dipole integrals.</p>
<p>In this tutorial we will:</p>
<ul>
  <li>Visualize water and ammonia</li>
  <li>Compute dipole moments from HF density matrices</li>
  <li>Compare with experimental dipole moments</li>
</ul>
""")

# %% Visualize Molecules
from cm.views import html, molecule

html("<h3>Molecules</h3>")

html("<p><strong>Water (H<sub>2</sub>O):</strong></p>")
water_vis = [
    ('O', 0.000,  0.000, 0.000),
    ('H', 0.000,  0.757, 0.587),
    ('H', 0.000, -0.757, 0.587),
]
molecule(water_vis, bonds=[(0, 1), (0, 2)])

html("<p><strong>Ammonia (NH<sub>3</sub>):</strong></p>")
nh3_vis = [
    ('N', 0.000, 0.000, 0.000),
    ('H', 0.000, 0.942, 0.380),
    ('H', 0.816, -0.471, 0.380),
    ('H', -0.816, -0.471, 0.380),
]
molecule(nh3_vis, bonds=[(0, 1), (0, 2), (0, 3)])

# %% Compute Dipole Moments
from cm.views import html, log, table
from cm.qm.integrals import hartree_fock, dipole_moment

html("<h3>Dipole Moment Calculations</h3>")

# Water
water = [
    ('O', (0.000,  0.000, 0.000)),
    ('H', (0.000,  0.757, 0.587)),
    ('H', (0.000, -0.757, 0.587)),
]
log("Computing HF/STO-3G for water...", level="info")
hf_water = hartree_fock(water, basis='STO-3G')
dip_water = dipole_moment(hf_water)

log(f"Water dipole: {dip_water.magnitude:.4f} Debye", level="success")
log(f"  Components (Debye): x={dip_water.dipole[0]:.4f}, y={dip_water.dipole[1]:.4f}, z={dip_water.dipole[2]:.4f}", level="info")

# Ammonia
nh3 = [
    ('N', (0.000, 0.000, 0.000)),
    ('H', (0.000, 0.942, 0.380)),
    ('H', (0.816, -0.471, 0.380)),
    ('H', (-0.816, -0.471, 0.380)),
]
log("Computing HF/STO-3G for ammonia...", level="info")
hf_nh3 = hartree_fock(nh3, basis='STO-3G')
dip_nh3 = dipole_moment(hf_nh3)

log(f"Ammonia dipole: {dip_nh3.magnitude:.4f} Debye", level="success")
log(f"  Components (Debye): x={dip_nh3.dipole[0]:.4f}, y={dip_nh3.dipole[1]:.4f}, z={dip_nh3.dipole[2]:.4f}", level="info")

# %% Dipole Breakdown
from cm.views import html, table
import numpy as np

html("<h3>Dipole Moment Components</h3>")

dipole_data = [
    ["H2O", f"{dip_water.magnitude:.4f}",
     f"({dip_water.dipole[0]:.3f}, {dip_water.dipole[1]:.3f}, {dip_water.dipole[2]:.3f})",
     f"({dip_water.dipole_nuclear[0]:.3f}, {dip_water.dipole_nuclear[1]:.3f}, {dip_water.dipole_nuclear[2]:.3f})" if dip_water.dipole_nuclear is not None else "N/A",
     f"({dip_water.dipole_electronic[0]:.3f}, {dip_water.dipole_electronic[1]:.3f}, {dip_water.dipole_electronic[2]:.3f})" if dip_water.dipole_electronic is not None else "N/A"],
    ["NH3", f"{dip_nh3.magnitude:.4f}",
     f"({dip_nh3.dipole[0]:.3f}, {dip_nh3.dipole[1]:.3f}, {dip_nh3.dipole[2]:.3f})",
     f"({dip_nh3.dipole_nuclear[0]:.3f}, {dip_nh3.dipole_nuclear[1]:.3f}, {dip_nh3.dipole_nuclear[2]:.3f})" if dip_nh3.dipole_nuclear is not None else "N/A",
     f"({dip_nh3.dipole_electronic[0]:.3f}, {dip_nh3.dipole_electronic[1]:.3f}, {dip_nh3.dipole_electronic[2]:.3f})" if dip_nh3.dipole_electronic is not None else "N/A"],
]
table(dipole_data, headers=["Molecule", "|mu| (D)", "Total (D)", "Nuclear (au)", "Electronic (au)"])

# %% Comparison with Experiment
from cm.views import html, table
from cm.data import get

html("<h3>Comparison with Experimental Dipole Moments</h3>")

mol_data = get("water")
log(f"Data for: {mol_data.name} ({mol_data.formula})", level="info")

comparison = [
    ["H2O", f"{dip_water.magnitude:.2f}", "1.85", "NIST"],
    ["NH3", f"{dip_nh3.magnitude:.2f}", "1.47", "NIST"],
]
table(comparison, headers=["Molecule", "Computed (D)", "Experimental (D)", "Source"])

html("""
<p><strong>Key insight:</strong> Dipole moments are sensitive to the quality of
the electron density, which depends on both the method (HF, DFT, correlated) and
the basis set. STO-3G is a minimal basis set that provides qualitative trends but
may differ from experiment. Larger basis sets (6-31G*, cc-pVTZ) with polarization
functions dramatically improve dipole moment predictions.</p>
<p>The direction of the dipole indicates the charge separation: in water, the
oxygen end carries partial negative charge, and the hydrogen end carries partial
positive charge. In ammonia, the nitrogen lone pair creates a dipole pointing
away from the hydrogen atoms.</p>
""")
