# %% Introduction
from cm.views import html

html("""
<h2>Tutorial 4: Correlated Methods &mdash; MP2</h2>
<p>Hartree-Fock theory treats electron-electron repulsion in a mean-field
approximation, missing the instantaneous correlation between electrons.
Post-HF methods systematically recover this <em>correlation energy</em>.</p>
<p>M&oslash;ller-Plesset perturbation theory at second order (MP2) is the simplest
and most widely used correlated method. It adds a perturbative correction to the
HF energy using double excitations.</p>
<p>In this tutorial we will:</p>
<ul>
  <li>Run HF on water as the reference calculation</li>
  <li>Apply MP2 to recover correlation energy</li>
  <li>Compare HF and MP2 total energies</li>
  <li>Discuss the importance of electron correlation</li>
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

# %% Run HF Reference
from cm.views import html, log
from cm.qm.integrals import hartree_fock

water = [
    ('O', (0.000,  0.000, 0.000)),
    ('H', (0.000,  0.757, 0.587)),
    ('H', (0.000, -0.757, 0.587)),
]

html("<h3>Step 1: Hartree-Fock Reference</h3>")
hf_result = hartree_fock(water, basis='STO-3G', verbose=True)
log(f"HF Energy: {hf_result.energy:.10f} Hartree", level="success")

# %% Run MP2
from cm.views import html, log, table
from cm.qm.integrals import mp2

html("<h3>Step 2: MP2 Correlation Energy</h3>")
log("Computing MP2 correlation energy...", level="info")
log("This involves transforming the 2-electron integrals to the MO basis (O(N^5) step)", level="info")

mp2_result = mp2(hf_result=hf_result, verbose=True)

log(f"HF Reference Energy: {mp2_result.energy_hf:.10f} Hartree", level="info")
log(f"MP2 Correlation Energy: {mp2_result.energy_mp2:.10f} Hartree", level="info")
log(f"MP2 Total Energy: {mp2_result.energy_total:.10f} Hartree", level="success")

# %% Energy Summary
from cm.views import html, table

html("<h3>Energy Summary</h3>")

pct_corr = abs(mp2_result.energy_mp2 / (mp2_result.energy_total - mp2_result.energy_hf)) * 100 if mp2_result.energy_total != mp2_result.energy_hf else 0

summary = [
    ["HF Energy", f"{mp2_result.energy_hf:.8f}"],
    ["MP2 Correlation", f"{mp2_result.energy_mp2:.8f}"],
    ["MP2 Total", f"{mp2_result.energy_total:.8f}"],
    ["Correlation (kcal/mol)", f"{mp2_result.energy_mp2 * 627.509:.2f}"],
]
table(summary, headers=["Component", "Energy (Hartree)"])

# %% Comparison with Experiment
from cm.views import html, table
from cm.data import get

html("<h3>Comparison with Reference Data</h3>")

mol_data = get("water")
log(f"Data source: {mol_data.name} from {mol_data.sources}", level="info")

comparison = [
    ["HF/STO-3G", f"{hf_result.energy:.6f}"],
    ["MP2/STO-3G", f"{mp2_result.energy_total:.6f}"],
    ["HF/CBS limit", "-76.067"],
    ["MP2/CBS limit", "-76.370"],
    ["CCSD(T)/CBS limit", "-76.437"],
    ["Exact (non-rel.)", "-76.438"],
]
table(comparison, headers=["Method", "Energy (Hartree)"])

html("""
<p><strong>Key insight:</strong> MP2 recovers a significant portion of the electron
correlation energy at a computational cost of O(N<sup>5</sup>), compared to HF's
O(N<sup>4</sup>). The correlation energy of water is about &minus;1.5 eV at the
complete basis set limit. With STO-3G, both the HF and MP2 energies are far from
the CBS limit due to basis set incompleteness, but the relative correlation
energy recovery is physically meaningful.</p>
<p>For high accuracy, larger basis sets (cc-pVTZ, cc-pVQZ) and higher-level
methods (CCSD(T)) are needed.</p>
""")
