# %% Introduction
from cm.views import html, log, table

html("""
<h2>Tutorial 1: Hartree-Fock Theory</h2>
<p>The Hartree-Fock (HF) method is the foundation of ab initio quantum chemistry.
It approximates the many-electron wavefunction as a single Slater determinant
and solves the Roothaan-Hall equations self-consistently (SCF).</p>
<p>In this tutorial we will:</p>
<ul>
  <li>Define a water molecule and visualize it in 3D</li>
  <li>Run a Restricted Hartree-Fock calculation with the STO-3G basis set</li>
  <li>Examine the energy components and molecular orbital energies</li>
  <li>Compare our computed properties with experimental data</li>
</ul>
""")

# %% Define and Visualize the Water Molecule
from cm.views import html, molecule

# Water geometry (near experimental equilibrium, in Angstroms)
water = [
    ('O', 0.000,  0.000, 0.000),
    ('H', 0.000,  0.757, 0.587),
    ('H', 0.000, -0.757, 0.587),
]

html("<h3>Water (H<sub>2</sub>O) Molecular Structure</h3>")
molecule(water, bonds=[(0, 1), (0, 2)])

# %% Run Hartree-Fock Calculation
from cm.views import html, log
from cm.qm.integrals import hartree_fock

html("<h3>Hartree-Fock SCF Calculation</h3>")
log("Running RHF/STO-3G on water...", level="info")

water_nested = [
    ('O', (0.000,  0.000, 0.000)),
    ('H', (0.000,  0.757, 0.587)),
    ('H', (0.000, -0.757, 0.587)),
]

hf_result = hartree_fock(water_nested, basis='STO-3G', verbose=True)

log(f"Converged: {hf_result.converged}", level="success")
log(f"Total HF Energy: {hf_result.energy:.10f} Hartree", level="success")

# %% Energy Breakdown
from cm.views import html, table, log
import numpy as np

html("<h3>Energy Components</h3>")

energy_data = [
    ["Kinetic Energy (T)", f"{hf_result.E_kinetic:.6f}"],
    ["Nuclear Attraction (V_ne)", f"{hf_result.E_nuclear_attraction:.6f}"],
    ["Coulomb Repulsion (J)", f"{hf_result.E_coulomb:.6f}"],
    ["Exchange Energy (K)", f"{hf_result.E_exchange:.6f}"],
    ["Nuclear Repulsion (V_nn)", f"{hf_result.E_nuclear_repulsion:.6f}"],
    ["Total Electronic", f"{hf_result.energy - hf_result.E_nuclear_repulsion:.6f}"],
    ["Total Energy", f"{hf_result.energy:.6f}"],
]
table(energy_data, headers=["Component", "Energy (Hartree)"])

# %% Molecular Orbital Energies
from cm.views import html, table, log
import numpy as np

html("<h3>Molecular Orbital Energies</h3>")

n_occ = hf_result.n_electrons // 2
orb_data = []
for i, eps in enumerate(hf_result.orbital_energies):
    occ = "occupied" if i < n_occ else "virtual"
    label = ""
    if i == n_occ - 1:
        label = " (HOMO)"
    elif i == n_occ:
        label = " (LUMO)"
    orb_data.append([f"MO {i+1}", f"{eps:.6f}", f"{eps * 27.2114:.2f}", f"{occ}{label}"])

table(orb_data, headers=["Orbital", "Energy (Hartree)", "Energy (eV)", "Occupation"])

homo = hf_result.orbital_energies[n_occ - 1]
lumo = hf_result.orbital_energies[n_occ]
gap = (lumo - homo) * 27.2114
log(f"HOMO-LUMO gap: {gap:.2f} eV", level="info")

# %% Comparison with Experimental Data
from cm.views import html, table, log
from cm.data import get

html("<h3>Comparison with Reference Data</h3>")

mol_data = get("water")
log(f"Molecule: {mol_data.name} ({mol_data.formula})", level="info")
log(f"PubChem CID: {mol_data.cid}", level="info")

# Known reference values for water
# Literature HF/STO-3G energy: -74.9659 Hartree
# Experimental total energy (est.): -76.48 Hartree
comparison = [
    ["HF/STO-3G Energy", f"{hf_result.energy:.6f} Ha", "-74.9659 Ha", "Literature HF/STO-3G"],
    ["Exact non-rel. Energy", f"{hf_result.energy:.6f} Ha", "-76.438 Ha", "Experimental estimate"],
    ["Correlation Energy", "0.000 (by definition)", "-1.472 Ha", "Difference (exact - HF/CBS)"],
]
table(comparison, headers=["Property", "Computed", "Reference", "Source"])

html("""
<p><strong>Key insight:</strong> HF captures ~98.1% of the total energy but misses
electron correlation entirely. The correlation energy (~1.5 Ha for water) is small
in absolute terms but chemically significant &mdash; it is essential for accurate
bond energies, reaction barriers, and molecular properties.</p>
""")
