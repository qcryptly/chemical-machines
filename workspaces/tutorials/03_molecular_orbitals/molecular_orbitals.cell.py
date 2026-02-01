# %% Introduction
from cm.views import html

html("""
<h2>Tutorial 3: Molecular Orbital Visualization</h2>
<p>Molecular orbitals (MOs) describe the spatial distribution of electrons in a
molecule. They are obtained as eigenvectors of the Fock (or Kohn-Sham) matrix
during the SCF procedure. Each MO is a linear combination of atomic orbital
basis functions (LCAO).</p>
<p>In this tutorial we will:</p>
<ul>
  <li>Compute the electronic structure of water using Hartree-Fock</li>
  <li>Visualize the HOMO (highest occupied MO) and LUMO (lowest unoccupied MO)</li>
  <li>Display the full orbital energy level diagram</li>
  <li>Discuss chemical bonding from the MO perspective</li>
</ul>
""")

# %% Visualize Water
from cm.views import html, molecule

water_vis = [
    ('O', 0.000,  0.000, 0.000),
    ('H', 0.000,  0.757, 0.587),
    ('H', 0.000, -0.757, 0.587),
]

html("<h3>Water Molecule</h3>")
molecule(water_vis, bonds=[(0, 1), (0, 2)])

# %% Run HF Calculation
from cm.views import html, log
from cm.qm.integrals import hartree_fock

water = [
    ('O', (0.000,  0.000, 0.000)),
    ('H', (0.000,  0.757, 0.587)),
    ('H', (0.000, -0.757, 0.587)),
]

html("<h3>Hartree-Fock Calculation</h3>")
hf_result = hartree_fock(water, basis='STO-3G')
log(f"HF Energy: {hf_result.energy:.6f} Hartree", level="success")
log(f"Number of MOs: {len(hf_result.orbital_energies)}", level="info")

n_occ = hf_result.n_electrons // 2
log(f"Occupied MOs: {n_occ}", level="info")
log(f"Virtual MOs: {len(hf_result.orbital_energies) - n_occ}", level="info")

# %% Visualize HOMO
from cm.views import html, orbital

html("<h3>HOMO (Highest Occupied Molecular Orbital)</h3>")
html("<p>The HOMO of water is a lone pair orbital on oxygen, oriented perpendicular to the molecular plane.</p>")
orbital(hf_result, mo_index='HOMO', isovalue=0.02, resolution=30)

# %% Visualize LUMO
from cm.views import html, orbital

html("<h3>LUMO (Lowest Unoccupied Molecular Orbital)</h3>")
html("<p>The LUMO of water is an antibonding orbital with a node between the O and H atoms.</p>")
orbital(hf_result, mo_index='LUMO', isovalue=0.02, resolution=30)

# %% Orbital Energy Diagram
from cm.views import html, table, log
from cm.data import get

html("<h3>Orbital Energy Level Diagram</h3>")

n_occ = hf_result.n_electrons // 2
orb_data = []
labels = {0: "1a1 (O 1s core)", 1: "2a1 (O 2s + H)", 2: "1b2 (O-H bonding)",
          3: "3a1 (lone pair)", 4: "1b1 (lone pair, HOMO)"}

for i, eps in enumerate(hf_result.orbital_energies):
    occ_str = "**" if i < n_occ else "  "
    label = labels.get(i, "")
    marker = ""
    if i == n_occ - 1:
        marker = " <-- HOMO"
    elif i == n_occ:
        marker = " <-- LUMO"
    orb_data.append([f"{occ_str} MO {i+1}", f"{eps:.4f}", f"{eps*27.2114:.2f}", label + marker])

table(orb_data, headers=["Orbital", "Energy (Ha)", "Energy (eV)", "Character"])

# %% Comparison
from cm.views import html, table
from cm.data import get

html("<h3>Comparison with Reference Data</h3>")

mol_data = get("water")
log(f"Molecule: {mol_data.name} ({mol_data.formula})", level="info")

homo = hf_result.orbital_energies[n_occ - 1]
lumo = hf_result.orbital_energies[n_occ]
gap = (lumo - homo) * 27.2114

comparison = [
    ["HOMO energy", f"{homo:.4f} Ha ({homo*27.2114:.2f} eV)", "-0.500 Ha (-13.61 eV)", "HF/large basis"],
    ["HOMO-LUMO gap", f"{gap:.2f} eV", "~13.5 eV", "HF/large basis"],
    ["1st Ionization Energy", f"{-homo*27.2114:.2f} eV (Koopmans)", "12.62 eV", "Experimental"],
]
table(comparison, headers=["Property", "Computed (STO-3G)", "Reference", "Source"])

html("""
<p><strong>Key insight:</strong> Koopmans' theorem relates the HOMO energy to the
ionization potential. With a minimal basis set, quantitative agreement is limited,
but the qualitative orbital picture is correct: water has two lone pairs and two
bonding orbitals, consistent with its bent geometry and chemical behavior.</p>
""")
