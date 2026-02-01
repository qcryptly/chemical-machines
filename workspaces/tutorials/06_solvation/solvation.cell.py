# %% Introduction
from cm.views import html

html("""
<h2>Tutorial 6: Implicit Solvation &mdash; PCM</h2>
<p>Most chemistry occurs in solution, not in vacuum. The Polarizable Continuum
Model (PCM) accounts for solvent effects by surrounding the molecule with a
dielectric continuum characterized by the solvent's dielectric constant.</p>
<p>The molecule is placed inside a cavity carved from the continuum, and the
interaction between the molecular charge distribution and the induced surface
charges on the cavity provides the electrostatic solvation free energy.</p>
<p>In this tutorial we will:</p>
<ul>
  <li>Compute the gas-phase energy of water using HF</li>
  <li>Apply PCM to compute the solvation energy in water</li>
  <li>Examine the molecular cavity and surface charges</li>
  <li>Compare with experimental hydration free energy</li>
</ul>
""")

# %% Visualize Water
from cm.views import html, molecule

water_vis = [
    ('O', 0.000,  0.000, 0.000),
    ('H', 0.000,  0.757, 0.587),
    ('H', 0.000, -0.757, 0.587),
]

html("<h3>Water (H<sub>2</sub>O)</h3>")
molecule(water_vis, bonds=[(0, 1), (0, 2)])

# %% Gas Phase HF
from cm.views import html, log
from cm.qm.integrals import hartree_fock

water = [
    ('O', (0.000,  0.000, 0.000)),
    ('H', (0.000,  0.757, 0.587)),
    ('H', (0.000, -0.757, 0.587)),
]

html("<h3>Step 1: Gas Phase Calculation</h3>")
hf_result = hartree_fock(water, basis='STO-3G')
log(f"Gas phase HF energy: {hf_result.energy:.6f} Hartree", level="success")

# %% PCM Solvation
from cm.views import html, log, table
from cm.qm.integrals import compute_solvation_energy

html("<h3>Step 2: PCM Solvation in Water</h3>")
log("Computing solvation energy using PCM...", level="info")

pcm_result = compute_solvation_energy(hf_result, solvent='water', verbose=True)

log(f"Solvation Energy: {pcm_result.solvation_energy:.6f} Hartree", level="success")
log(f"Solvation Energy: {pcm_result.solvation_energy_kcal:.2f} kcal/mol", level="success")
log(f"Dielectric constant: {pcm_result.epsilon}", level="info")

# %% Cavity Details
from cm.views import html, table, log

html("<h3>Molecular Cavity Details</h3>")

cavity = pcm_result.cavity
log(f"Number of surface points (tesserae): {cavity.n_points}", level="info")
log(f"Total cavity surface area: {cavity.total_area:.2f} Angstrom^2", level="info")

cavity_data = [
    ["Surface points", str(cavity.n_points)],
    ["Surface area", f"{cavity.total_area:.2f} A^2"],
    ["Dielectric constant", f"{pcm_result.epsilon:.2f}"],
    ["Solvent", pcm_result.solvent or "water"],
    ["Solvation energy", f"{pcm_result.solvation_energy_kcal:.2f} kcal/mol"],
]
table(cavity_data, headers=["Property", "Value"])

# %% Comparison with Experiment
from cm.views import html, table
from cm.data import get

html("<h3>Comparison with Experimental Data</h3>")

mol_data = get("water")
log(f"Molecule: {mol_data.name} ({mol_data.formula})", level="info")

comparison = [
    ["PCM/HF/STO-3G", f"{pcm_result.solvation_energy_kcal:.2f} kcal/mol", "This calculation"],
    ["PCM/HF/6-31G*", "~-8.0 kcal/mol", "Literature"],
    ["Experimental (hydration)", "-6.3 kcal/mol", "Ben-Naim & Marcus, 1984"],
]
table(comparison, headers=["Method", "Solvation Energy", "Source"])

html("""
<p><strong>Key insight:</strong> The PCM model captures the dominant electrostatic
contribution to solvation. The computed solvation energy depends strongly on the
basis set and the quality of the electronic density. Additional contributions from
cavitation, dispersion, and repulsion are needed for quantitative accuracy.</p>
<p>With a minimal basis set, the electrostatic solvation energy may be overestimated.
Larger basis sets and inclusion of non-electrostatic terms bring the result closer
to the experimental hydration free energy of &minus;6.3 kcal/mol.</p>
""")
