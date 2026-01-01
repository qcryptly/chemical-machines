from cm import qm
from cm.views import html

html("<h2>Relativistic Quantum Mechanics - Dirac Spinors</h2>")

html("<h3>Heavy Atom: Gold (Au) Core Electrons</h3>")
html("<p>In heavy atoms, relativistic effects are crucial. The 1s electrons move at ~60% the speed of light!</p>")

# Gold 1s shell - relativistic (using κ notation)
# 1s₁/₂: n=1, κ=-1, mⱼ=±1/2
spinor_1s_up = qm.dirac_spinor(n=1, kappa=-1, mj=0.5)
spinor_1s_down = qm.dirac_spinor(n=1, kappa=-1, mj=-0.5)

html("<p>1s₁/₂ shell spinors:</p>")
html(f"<p>κ = {spinor_1s_up.kappa}, j = {spinor_1s_up.j}, l = {spinor_1s_up.l}</p>")

# Create Dirac determinant for 1s² configuration
config_1s2 = qm.dirac_slater([spinor_1s_up, spinor_1s_down])
html("<p>1s² configuration:</p>")
config_1s2.render()

# Also show in spectroscopic notation
html("<p>Spectroscopic notation:</p>")
config_1s2.render(notation="spectroscopic")

html("<h3>Spin-Orbit Splitting in 2p Shell</h3>")
html("<p>Unlike non-relativistic QM, the 2p shell splits into 2p₁/₂ and 2p₃/₂:</p>")

# 2p₁/₂: n=2, κ=+1 (j = l - 1/2 = 0.5)
spinor_2p12_up = qm.dirac_spinor(n=2, kappa=1, mj=0.5)
spinor_2p12_down = qm.dirac_spinor(n=2, kappa=1, mj=-0.5)

# 2p₃/₂: n=2, κ=-2 (j = l + 1/2 = 1.5)
spinor_2p32_m32 = qm.dirac_spinor(n=2, kappa=-2, mj=-1.5)
spinor_2p32_m12 = qm.dirac_spinor(n=2, kappa=-2, mj=-0.5)
spinor_2p32_p12 = qm.dirac_spinor(n=2, kappa=-2, mj=0.5)
spinor_2p32_p32 = qm.dirac_spinor(n=2, kappa=-2, mj=1.5)

html("<p>2p₁/₂ has 2 states (mⱼ = ±1/2):</p>")
html(f"<p>  {spinor_2p12_up.label}</p>")
html(f"<p>  {spinor_2p12_down.label}</p>")

html("<p>2p₃/₂ has 4 states (mⱼ = ±1/2, ±3/2):</p>")
html(f"<p>  {spinor_2p32_m32.label}</p>")
html(f"<p>  {spinor_2p32_m12.label}</p>")
html(f"<p>  {spinor_2p32_p12.label}</p>")
html(f"<p>  {spinor_2p32_p32.label}</p>")

html("<h3>Alternative: Using (n, l, j, mⱼ) Notation</h3>")
html("<p>For convenience, you can specify orbitals using l and j instead of κ:</p>")

# Same 2p₃/₂ spinor using l, j notation
spinor_alt = qm.dirac_spinor_lj(n=2, l=1, j=1.5, mj=0.5)
html(f"<p>dirac_spinor_lj(n=2, l=1, j=1.5, mj=0.5) → κ = {spinor_alt.kappa}</p>")

# Or use basis_dirac with 4-tuples
spinors = qm.basis_dirac([
    (2, 1, 1.5, 0.5),   # 2p₃/₂, mⱼ=+1/2 using (n, l, j, mⱼ)
    (2, 1, 1.5, -0.5),  # 2p₃/₂, mⱼ=-1/2
])
html("<p>Created via basis_dirac with (n, l, j, mⱼ) tuples:</p>")
qm.dirac_slater(spinors).render(notation="spectroscopic")

# %%
html("<h3>Dirac-Coulomb Hamiltonian Matrix Elements</h3>")

# Create two configurations
config_1s2 = qm.dirac_slater([spinor_1s_up, spinor_1s_down])
config_2p12 = qm.dirac_slater([spinor_2p12_up, spinor_2p12_down])

# Dirac-Coulomb Hamiltonian
H_DC = qm.dirac_hamiltonian("coulomb")

html("<p>Diagonal element ⟨1s²|Ĥ_DC|1s²⟩:</p>")
(config_1s2 @ H_DC @ config_1s2).render()

html("<p>Off-diagonal ⟨1s²|Ĥ_DC|2p₁/₂²⟩ (double excitation):</p>")
(config_1s2 @ H_DC @ config_2p12).render()

html("<h3>Dirac-Coulomb-Breit Hamiltonian</h3>")
html("<p>For highest accuracy, include Breit interaction (magnetic + retardation):</p>")

H_DCB = qm.dirac_hamiltonian("coulomb_breit")

html("<p>⟨1s²|Ĥ_DCB|1s²⟩:</p>")
(config_1s2 @ H_DCB @ config_1s2).render()

html("<h3>Overlaps</h3>")

# Same configuration
overlap_same = config_1s2 @ config_1s2
html(f"<p>⟨1s²|1s²⟩ = {overlap_same.value}</p>")

# Different configurations (orthogonal)
overlap_diff = config_1s2 @ config_2p12
html(f"<p>⟨1s²|2p₁/₂²⟩ = {overlap_diff.value}</p>")

html("<h3>κ Quantum Number Reference</h3>")
html("""
<table>
<tr><th>Orbital</th><th>l</th><th>j</th><th>κ</th><th>States</th></tr>
<tr><td>s₁/₂</td><td>0</td><td>1/2</td><td>-1</td><td>2</td></tr>
<tr><td>p₁/₂</td><td>1</td><td>1/2</td><td>+1</td><td>2</td></tr>
<tr><td>p₃/₂</td><td>1</td><td>3/2</td><td>-2</td><td>4</td></tr>
<tr><td>d₃/₂</td><td>2</td><td>3/2</td><td>+2</td><td>4</td></tr>
<tr><td>d₅/₂</td><td>2</td><td>5/2</td><td>-3</td><td>6</td></tr>
<tr><td>f₅/₂</td><td>3</td><td>5/2</td><td>+3</td><td>6</td></tr>
<tr><td>f₇/₂</td><td>3</td><td>7/2</td><td>-4</td><td>8</td></tr>
</table>
<p>Formula: κ = -(l+1) for j = l+1/2, κ = +l for j = l-1/2</p>
""")
