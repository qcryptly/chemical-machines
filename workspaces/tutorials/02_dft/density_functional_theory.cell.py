# %% Introduction
from cm.views import html

html("""
<h2>Tutorial 2: Density Functional Theory</h2>
<p>Density Functional Theory (DFT) reformulates the quantum many-body problem
in terms of the electron density rather than the wavefunction. The Kohn-Sham
approach maps the interacting system onto a non-interacting reference with the
same density, capturing exchange-correlation effects through an approximate
functional.</p>
<p>In this tutorial we will:</p>
<ul>
  <li>Visualize the water molecule</li>
  <li>Run Kohn-Sham DFT with the B3LYP hybrid functional</li>
  <li>Compare HF vs DFT energy components</li>
  <li>Examine the role of exchange-correlation energy</li>
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

# %% Run DFT Calculation
from cm.views import html, log
from cm.qm.integrals import kohn_sham

html("<h3>Kohn-Sham DFT Calculation (B3LYP/STO-3G)</h3>")

water = [
    ('O', (0.000,  0.000, 0.000)),
    ('H', (0.000,  0.757, 0.587)),
    ('H', (0.000, -0.757, 0.587)),
]

log("Running B3LYP/STO-3G...", level="info")
dft_result = kohn_sham(water, functional='B3LYP', basis='STO-3G', verbose=True)

log(f"Converged: {dft_result.converged}", level="success")
log(f"Total DFT Energy: {dft_result.energy:.10f} Hartree", level="success")
log(f"Functional: {dft_result.functional_name}", level="info")

# %% Compare HF vs DFT
from cm.views import html, table, log
from cm.qm.integrals import hartree_fock

html("<h3>HF vs DFT Energy Comparison</h3>")
log("Running HF/STO-3G for comparison...", level="info")
hf_result = hartree_fock(water, basis='STO-3G')

comparison = [
    ["Total Energy", f"{hf_result.energy:.6f}", f"{dft_result.energy:.6f}"],
    ["Kinetic Energy", f"{hf_result.E_kinetic:.6f}", f"{dft_result.E_kinetic:.6f}"],
    ["Nuclear Attraction", f"{hf_result.E_nuclear_attraction:.6f}", f"{dft_result.E_nuclear_attraction:.6f}"],
    ["Coulomb (J)", f"{hf_result.E_coulomb:.6f}", f"{dft_result.E_coulomb:.6f}"],
    ["Exchange (K)", f"{hf_result.E_exchange:.6f}", "---"],
    ["XC Energy", "---", f"{dft_result.E_xc:.6f}"],
    ["Exact Exchange Fraction", "100%", f"{dft_result.exact_exchange_fraction*100:.0f}%"],
    ["Nuclear Repulsion", f"{hf_result.E_nuclear_repulsion:.6f}", f"{dft_result.E_nuclear_repulsion:.6f}"],
]
table(comparison, headers=["Component", "HF/STO-3G (Ha)", "B3LYP/STO-3G (Ha)"])

# %% DFT Orbital Energies
from cm.views import html, table

html("<h3>Kohn-Sham Orbital Energies</h3>")

n_occ = dft_result.n_electrons // 2
orb_data = []
for i, eps in enumerate(dft_result.orbital_energies):
    occ = "occupied" if i < n_occ else "virtual"
    label = ""
    if i == n_occ - 1:
        label = " (HOMO)"
    elif i == n_occ:
        label = " (LUMO)"
    orb_data.append([f"KS-MO {i+1}", f"{eps:.6f}", f"{eps * 27.2114:.2f}", f"{occ}{label}"])

table(orb_data, headers=["Orbital", "Energy (Hartree)", "Energy (eV)", "Occupation"])

# %% Reference Comparison
from cm.views import html, table
from cm.data import get

html("<h3>Comparison with Reference Data</h3>")

mol_data = get("water")

# Known values
comparison = [
    ["HF/STO-3G", f"{hf_result.energy:.6f} Ha", "Hartree-Fock (this tutorial)"],
    ["B3LYP/STO-3G", f"{dft_result.energy:.6f} Ha", "DFT (this tutorial)"],
    ["B3LYP/6-311++G(2d,2p)", "-76.4600 Ha", "Literature (large basis)"],
    ["Experimental", "-76.438 Ha", "Estimated exact energy"],
]
table(comparison, headers=["Method", "Energy", "Source"])

html("""
<p><strong>Key insight:</strong> DFT with hybrid functionals like B3LYP provides
a good balance between accuracy and computational cost. The exchange-correlation
functional captures electron correlation effects that pure HF misses, typically
recovering 80-95% of the correlation energy at a cost similar to HF.</p>
<p>Note: With a minimal basis set (STO-3G), both HF and DFT results have large
basis set errors. Larger basis sets (6-31G*, cc-pVTZ) would significantly improve
accuracy.</p>
""")
