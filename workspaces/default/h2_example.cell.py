# %%
from cm import qm
from cm.views import html

html("<h2>H\u2082 Molecule - Slater Determinant Basis</h2>")

# Define atomic orbitals: 1s on atom A (n=1) and atom B (n=2)
orbital_1sA_up = qm.basis_sh_element(spin=1, L=0, m=0, n=1)
orbital_1sA_down = qm.basis_sh_element(spin=-1, L=0, m=0, n=1)
orbital_1sB_up = qm.basis_sh_element(spin=1, L=0, m=0, n=2)
orbital_1sB_down = qm.basis_sh_element(spin=-1, L=0, m=0, n=2)

# Ionic configurations (both electrons on one atom)
html("<h3>Ionic Configurations</h3>")
config_ionic_A = qm.slater([orbital_1sA_up, orbital_1sA_down])
config_ionic_B = qm.slater([orbital_1sB_up, orbital_1sB_down])
html("<p>H\u207b\u2090 H\u207a\u1d47:</p>")
config_ionic_A.render()
html("<p>H\u207a\u2090 H\u207b\u1d47:</p>")
config_ionic_B.render()

# Covalent configurations (one electron on each atom)
html("<h3>Covalent Configurations</h3>")
config_cov_1 = qm.slater([orbital_1sA_up, orbital_1sB_down])
config_cov_2 = qm.slater([orbital_1sA_down, orbital_1sB_up])
config_cov_1.render()
config_cov_2.render()

# Overlaps
html("<h3>Overlap Matrix Elements</h3>")
html(f"<p>\u27e8ionic_A|ionic_A\u27e9 = {(config_ionic_A @ config_ionic_A).value}</p>")
html(f"<p>\u27e8ionic_A|ionic_B\u27e9 = {(config_ionic_A @ config_ionic_B).value}</p>")
html(f"<p>\u27e8ionic_A|covalent\u27e9 = {(config_ionic_A @ config_cov_1).value}</p>")

# Hamiltonian matrix elements
html("<h3>Hamiltonian Matrix Elements</h3>")
H = qm.hamiltonian()

html("<p>Diagonal \u27e8\u03a8|H|\u03a8\u27e9:</p>")
(config_ionic_A @ H @ config_ionic_A).render()

html("<p>Single excitation \u27e8ionic|H|covalent\u27e9:</p>")
(config_ionic_A @ H @ config_cov_1).render()

html("<p>Double excitation \u27e8ionic_A|H|ionic_B\u27e9:</p>")
(config_ionic_A @ H @ config_ionic_B).render()
