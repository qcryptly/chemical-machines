# %% Introduction
from cm.views import html

html("""
<h2>Tutorial 5: Excited States &mdash; CIS/TDA</h2>
<p>Electronic excitations occur when photons promote electrons from occupied to
virtual molecular orbitals. Computing excited states is essential for predicting
UV-Vis absorption spectra, photochemistry, and fluorescence.</p>
<p>The Tamm-Dancoff Approximation (TDA), also known as Configuration Interaction
Singles (CIS) when applied to HF, diagonalizes the response matrix built from
single excitations to find excited state energies and transition properties.</p>
<p>In this tutorial we will:</p>
<ul>
  <li>Run HF on water to get the ground state</li>
  <li>Apply TDA to compute excited state energies</li>
  <li>Analyze oscillator strengths and transition characters</li>
  <li>Compare with experimental UV absorption data</li>
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

# %% Ground State HF
from cm.views import html, log
from cm.qm.integrals import hartree_fock

water = [
    ('O', (0.000,  0.000, 0.000)),
    ('H', (0.000,  0.757, 0.587)),
    ('H', (0.000, -0.757, 0.587)),
]

html("<h3>Step 1: HF Ground State</h3>")
hf_result = hartree_fock(water, basis='STO-3G')
log(f"Ground state energy: {hf_result.energy:.6f} Hartree", level="success")

# %% Compute Excited States
from cm.views import html, log, table
from cm.qm.integrals import tda

html("<h3>Step 2: TDA Excited States</h3>")
log("Computing excited states via Tamm-Dancoff Approximation...", level="info")

td_result = tda(hf_result, n_states=5, verbose=True)

# %% Excitation Summary
from cm.views import html, table

html("<h3>Excited State Summary</h3>")

exc_data = []
for i in range(td_result.n_states):
    E_ev = td_result.excitation_energies[i]
    E_nm = 1239.84 / E_ev if E_ev > 0 else float('inf')
    f_osc = td_result.oscillator_strengths[i]
    exc_data.append([
        f"S{i+1}",
        f"{E_ev:.4f}",
        f"{E_nm:.1f}",
        f"{f_osc:.6f}",
        "Bright" if f_osc > 0.01 else "Dark"
    ])

table(exc_data, headers=["State", "Energy (eV)", "Wavelength (nm)", "Osc. Strength", "Character"])

# %% Comparison with Experiment
from cm.views import html, table
from cm.data import get

html("<h3>Comparison with Experimental UV Absorption</h3>")

mol_data = get("water")
log(f"Molecule: {mol_data.name} ({mol_data.formula})", level="info")

first_exc = td_result.excitation_energies[0]
first_nm = 1239.84 / first_exc if first_exc > 0 else 0

comparison = [
    ["First excitation (CIS/STO-3G)", f"{first_exc:.2f} eV ({first_nm:.0f} nm)"],
    ["First excitation (CIS/large basis)", "~8.5 eV (~146 nm)"],
    ["Experimental 1st absorption", "7.4 eV (167 nm)"],
    ["Experimental onset", "~6.5 eV (~190 nm)"],
]
table(comparison, headers=["Method", "Energy"])

html("""
<p><strong>Key insight:</strong> Water absorbs strongly in the vacuum UV region
(&lt;200 nm), which is why it is transparent in the visible spectrum.
CIS/TDA with a minimal basis set overestimates excitation energies due to
basis set incompleteness. With larger basis sets and correlated methods (EOM-CCSD,
ADC(2)), better agreement with the experimental 7.4 eV first absorption is
achieved.</p>
<p>Oscillator strengths near zero indicate <em>dark</em> (symmetry-forbidden)
transitions, while larger values indicate <em>bright</em> (allowed) transitions
visible in experimental spectra.</p>
""")
