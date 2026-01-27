"""
Chemical Machines QM Hamiltonian Module

Hamiltonian builder system for constructing quantum mechanical Hamiltonians.
"""

from typing import Optional, List, Union, Tuple, Dict, Any, Set
from dataclasses import dataclass, field

from ..symbols import Expr, Var, Const, Sum, _ensure_expr
from .. import views
from .spinorbitals import SpinOrbital, SlaterDeterminant
from .atoms import Atom
from .molecules import Molecule

__all__ = [
    'HamiltonianTerm',
    'HamiltonianBuilder',
    'MolecularHamiltonian',
    'MatrixExpression',
    'HamiltonianMatrix',
]


# =============================================================================
# HAMILTONIAN BUILDER SYSTEM
# =============================================================================


@dataclass
class HamiltonianTerm:
    """
    Individual term in the Hamiltonian.

    Represents a physical contribution like kinetic energy, nuclear attraction,
    or electron-electron repulsion with its symbolic expression.
    """
    name: str                      # 'kinetic', 'nuclear_attraction', 'coulomb', etc.
    symbol: str                    # LaTeX symbol: 'T', 'V_{ne}', 'V_{ee}'
    n_body: int                    # 1 for one-electron, 2 for two-electron
    coefficient: Expr              # Scaling factor (can be symbolic)
    expression: Optional[Expr]     # Symbolic form of the operator
    includes_exchange: bool = False  # For coulomb term
    description: str = ""          # Human-readable description

    def to_latex(self) -> str:
        """LaTeX representation of the term."""
        if isinstance(self.coefficient, Const) and self.coefficient.value == 1:
            return f"\\hat{{{self.symbol}}}"
        else:
            return f"{self.coefficient.to_latex()} \\hat{{{self.symbol}}}"


class HamiltonianBuilder:
    """
    Fluent builder for constructing molecular Hamiltonians.

    Allows configurable addition of physical terms like kinetic energy,
    nuclear attraction, electron-electron repulsion, spin-orbit coupling, etc.

    Example:
        # Symbolic/approximate integrals (fast)
        H = (qm.HamiltonianBuilder()
             .with_kinetic()
             .with_nuclear_attraction()
             .with_coulomb()
             .build())

        # Proper Gaussian integrals (accurate)
        H = (qm.HamiltonianBuilder()
             .with_kinetic()
             .with_nuclear_attraction()
             .with_coulomb()
             .with_basis('STO-3G')
             .build())
    """

    def __init__(self):
        self._terms: List[HamiltonianTerm] = []
        self._relativistic: bool = False
        self._basis_name: Optional[str] = None

    # =========================================================================
    # TERM ADDITION (FLUENT API)
    # =========================================================================

    def with_kinetic(self, mass: Union[float, Expr] = 1.0) -> "HamiltonianBuilder":
        """
        Add kinetic energy operator: T = -½∇²

        In atomic units with electron mass = 1.

        Args:
            mass: Particle mass (default 1.0 for electron in atomic units)

        Returns:
            self for chaining
        """
        from ..symbols import Const, Laplacian

        mass_expr = mass if isinstance(mass, Expr) else Const(mass)
        coeff = Const(-0.5) / mass_expr

        term = HamiltonianTerm(
            name="kinetic",
            symbol="T",
            n_body=1,
            coefficient=coeff,
            expression=None,  # Will use Laplacian operator
            description="Kinetic energy: -½∇²"
        )
        self._terms.append(term)
        return self

    def with_nuclear_attraction(self) -> "HamiltonianBuilder":
        """
        Add nuclear attraction operator: V_ne = -Σₐ Zₐ/rᵢₐ

        Sum over all nuclei a, for each electron i.

        Returns:
            self for chaining
        """
        from ..symbols import Const

        term = HamiltonianTerm(
            name="nuclear_attraction",
            symbol="V_{ne}",
            n_body=1,
            coefficient=Const(-1),
            expression=None,  # -Z/r for each nucleus
            description="Nuclear attraction: -Σₐ Zₐ/rᵢₐ"
        )
        self._terms.append(term)
        return self

    def with_coulomb(self) -> "HamiltonianBuilder":
        """
        Add electron-electron Coulomb repulsion: V_ee = Σᵢ<ⱼ 1/rᵢⱼ

        Includes exchange integral automatically (physical Coulomb interaction).

        Returns:
            self for chaining
        """
        from ..symbols import Const

        term = HamiltonianTerm(
            name="coulomb",
            symbol="V_{ee}",
            n_body=2,
            coefficient=Const(1),
            expression=None,  # 1/r_ij
            includes_exchange=True,
            description="Coulomb repulsion with exchange: Σᵢ<ⱼ 1/rᵢⱼ"
        )
        self._terms.append(term)
        return self

    def with_spin_orbit(self, model: str = 'zeff') -> "HamiltonianBuilder":
        """
        Add spin-orbit coupling: H_SO = ξ(r) L·S

        Args:
            model: Coupling model
                - 'zeff': Effective nuclear charge model
                - 'full': Full two-electron spin-orbit
                - 'mean_field': Mean-field approximation

        Returns:
            self for chaining
        """
        from ..symbols import Const

        term = HamiltonianTerm(
            name="spin_orbit",
            symbol="H_{SO}",
            n_body=1 if model == 'zeff' else 2,
            coefficient=Const(1),
            expression=None,  # ξ(r) L·S
            description=f"Spin-orbit coupling ({model}): ξ(r) L·S"
        )
        self._terms.append(term)
        return self

    def with_relativistic(self, correction: str = 'breit') -> "HamiltonianBuilder":
        """
        Add relativistic corrections.

        Args:
            correction: Type of correction
                - 'mass_velocity': Mass-velocity term
                - 'darwin': Darwin term
                - 'breit': Full Breit interaction
                - 'gaunt': Gaunt (magnetic) interaction only

        Returns:
            self for chaining
        """
        from ..symbols import Const

        self._relativistic = True

        if correction == 'mass_velocity':
            term = HamiltonianTerm(
                name="mass_velocity",
                symbol="H_{mv}",
                n_body=1,
                coefficient=Const(-1/8),  # -1/(8c²) in atomic units
                expression=None,  # -p⁴/(8m³c²)
                description="Mass-velocity correction: -p⁴/(8m³c²)"
            )
        elif correction == 'darwin':
            term = HamiltonianTerm(
                name="darwin",
                symbol="H_D",
                n_body=1,
                coefficient=Const(1),
                expression=None,  # (π/2)(Ze²/m²c²)δ(r)
                description="Darwin term: contact interaction at nucleus"
            )
        elif correction in ('breit', 'gaunt'):
            term = HamiltonianTerm(
                name=correction,
                symbol=f"H_{{{correction[:2]}}}",
                n_body=2,
                coefficient=Const(1),
                expression=None,
                description=f"{correction.capitalize()} interaction"
            )
        else:
            raise ValueError(f"Unknown relativistic correction: {correction}")

        self._terms.append(term)
        return self

    def with_external_field(self, field_type: str = 'electric',
                           strength: Union[float, Expr] = 1.0,
                           direction: Tuple[float, float, float] = (0, 0, 1)) -> "HamiltonianBuilder":
        """
        Add external field perturbation.

        Args:
            field_type: 'electric' or 'magnetic'
            strength: Field strength
            direction: Unit vector for field direction

        Returns:
            self for chaining
        """
        from ..symbols import Const

        strength_expr = strength if isinstance(strength, Expr) else Const(strength)

        if field_type == 'electric':
            term = HamiltonianTerm(
                name="electric_field",
                symbol="V_E",
                n_body=1,
                coefficient=strength_expr,
                expression=None,  # -E·r
                description=f"Electric field: -E·r, E={direction}"
            )
        elif field_type == 'magnetic':
            term = HamiltonianTerm(
                name="magnetic_field",
                symbol="H_B",
                n_body=1,
                coefficient=strength_expr,
                expression=None,  # μ_B (L + 2S)·B
                description=f"Magnetic field: μ_B(L+2S)·B, B={direction}"
            )
        else:
            raise ValueError(f"Unknown field type: {field_type}")

        self._terms.append(term)
        return self

    def with_custom(self, name: str, symbol: str, expression: Expr,
                   n_body: int = 1, coefficient: Union[float, Expr] = 1.0) -> "HamiltonianBuilder":
        """
        Add a custom operator term.

        Args:
            name: Term identifier
            symbol: LaTeX symbol
            expression: Symbolic expression for the operator
            n_body: 1 for one-electron, 2 for two-electron
            coefficient: Scaling factor

        Returns:
            self for chaining
        """
        from ..symbols import Const

        coeff = coefficient if isinstance(coefficient, Expr) else Const(coefficient)

        term = HamiltonianTerm(
            name=name,
            symbol=symbol,
            n_body=n_body,
            coefficient=coeff,
            expression=expression,
            description=f"Custom term: {name}"
        )
        self._terms.append(term)
        return self

    # =========================================================================
    # TERM MODIFICATION
    # =========================================================================

    def scale(self, term_name: str, factor: Union[float, Expr]) -> "HamiltonianBuilder":
        """
        Scale a term's coefficient.

        Args:
            term_name: Name of term to scale
            factor: Scaling factor

        Returns:
            self for chaining
        """
        from ..symbols import Const

        factor_expr = factor if isinstance(factor, Expr) else Const(factor)

        for term in self._terms:
            if term.name == term_name:
                term.coefficient = term.coefficient * factor_expr
                break
        else:
            raise ValueError(f"Term not found: {term_name}")

        return self

    def remove(self, term_name: str) -> "HamiltonianBuilder":
        """
        Remove a term from the Hamiltonian.

        Args:
            term_name: Name of term to remove

        Returns:
            self for chaining
        """
        self._terms = [t for t in self._terms if t.name != term_name]
        return self

    def with_basis(self, name: str = 'STO-3G') -> "HamiltonianBuilder":
        """
        Use proper Gaussian basis integrals instead of approximate Slater rules.

        When a basis is specified, numerical evaluation will use accurate
        Gaussian-type orbital integrals computed with the Boys function
        and Obara-Saika/McMurchie-Davidson schemes.

        Args:
            name: Basis set name ('STO-3G' currently supported)

        Returns:
            self for chaining

        Example:
            H = (qm.HamiltonianBuilder()
                 .with_kinetic()
                 .with_nuclear_attraction()
                 .with_coulomb()
                 .with_basis('STO-3G')
                 .build())

            # Now matrix evaluation uses proper integrals
            result = H.hartree_fock(molecule)
        """
        self._basis_name = name
        return self

    # =========================================================================
    # PRESETS
    # =========================================================================

    @classmethod
    def electronic(cls) -> "HamiltonianBuilder":
        """
        Standard non-relativistic electronic Hamiltonian.

        H = T + V_ne + V_ee (kinetic + nuclear attraction + electron repulsion)
        """
        return cls().with_kinetic().with_nuclear_attraction().with_coulomb()

    @classmethod
    def spin_orbit(cls) -> "HamiltonianBuilder":
        """
        Electronic Hamiltonian with spin-orbit coupling.

        H = T + V_ne + V_ee + H_SO
        """
        return cls.electronic().with_spin_orbit()

    @classmethod
    def relativistic(cls, correction: str = 'breit') -> "HamiltonianBuilder":
        """
        Relativistic Hamiltonian (Dirac-Coulomb-Breit).

        Args:
            correction: 'breit' or 'gaunt'
        """
        return cls.electronic().with_relativistic(correction)

    # =========================================================================
    # BUILD
    # =========================================================================

    def build(self) -> "MolecularHamiltonian":
        """
        Build the configured Hamiltonian.

        Returns:
            MolecularHamiltonian ready for matrix element evaluation
        """
        return MolecularHamiltonian(
            self._terms,
            self._relativistic,
            basis_name=self._basis_name
        )

    # =========================================================================
    # INSPECTION
    # =========================================================================

    @property
    def terms(self) -> List[str]:
        """List of term names."""
        return [t.name for t in self._terms]

    def to_latex(self) -> str:
        """LaTeX representation of the Hamiltonian."""
        if not self._terms:
            return "0"
        return " + ".join(t.to_latex() for t in self._terms)

    def render(self):
        """Render the Hamiltonian formula."""
        latex = "\\hat{H} = " + self.to_latex()
        html = f'<div class="cm-math cm-math-center">\\[ {latex} \\]</div>'
        views.html(html)


class MolecularHamiltonian:
    """
    A fully configured Hamiltonian operator.

    Created by HamiltonianBuilder.build(). Used to compute matrix elements
    between Slater determinants.

    Example:
        # Symbolic/approximate mode
        H = qm.HamiltonianBuilder.electronic().build()
        psi = water.slater_determinant()
        E = H.element(psi, psi)  # Ground state energy expression

        # With proper Gaussian integrals
        H = (qm.HamiltonianBuilder.electronic()
             .with_basis('STO-3G')
             .build())
        result = H.hartree_fock(molecule)  # Accurate HF calculation
    """

    def __init__(self, terms_or_hamiltonian: Union[List[HamiltonianTerm], "MolecularHamiltonian"],
                 relativistic_or_molecule: Union[bool, Molecule] = False,
                 basis_name: Optional[str] = None):
        """
        Create a MolecularHamiltonian.

        Two calling conventions:
        1. MolecularHamiltonian(terms, relativistic=False, basis_name=None) - from HamiltonianBuilder
        2. MolecularHamiltonian(hamiltonian, molecule) - bind existing Hamiltonian to Molecule

        Args:
            terms_or_hamiltonian: List of HamiltonianTerm, OR existing MolecularHamiltonian
            relativistic_or_molecule: bool for relativistic flag, OR Molecule to bind
            basis_name: Gaussian basis set name (e.g., 'STO-3G') for proper integrals
        """
        # Check if this is the "bind to molecule" calling convention
        if isinstance(terms_or_hamiltonian, MolecularHamiltonian):
            # Copy terms from existing Hamiltonian
            self._terms = terms_or_hamiltonian._terms
            self._relativistic = terms_or_hamiltonian._relativistic
            self._basis_name = terms_or_hamiltonian._basis_name
            # Store the bound molecule
            if isinstance(relativistic_or_molecule, Molecule):
                self._molecule = relativistic_or_molecule
            else:
                self._molecule = None
        else:
            # Standard calling convention: terms list + relativistic flag
            self._terms = terms_or_hamiltonian
            self._relativistic = relativistic_or_molecule if isinstance(relativistic_or_molecule, bool) else False
            self._basis_name = basis_name
            self._molecule = None

    # =========================================================================
    # MATRIX ELEMENTS
    # =========================================================================

    def element(self, bra: SlaterDeterminant, ket: SlaterDeterminant,
               molecule: Optional[Molecule] = None) -> "MatrixExpression":
        """
        Compute matrix element ⟨bra|H|ket⟩.

        Uses Slater-Condon rules to reduce to one- and two-electron integrals.

        Args:
            bra: Bra Slater determinant
            ket: Ket Slater determinant
            molecule: Optional molecule for nuclear positions. If not provided,
                      uses bound molecule or tries to infer from orbital centers.

        Returns:
            MatrixExpression (symbolic, can evaluate or compile)
        """
        # Use bound molecule, or try to infer from determinant
        mol = molecule if molecule is not None else self._molecule
        if mol is None:
            mol = self._infer_molecule(bra)

        return MatrixExpression(bra, ket, self, mol)

    def _infer_molecule(self, det: SlaterDeterminant) -> Optional[Molecule]:
        """Try to infer molecule from orbital centers and vec3 coordinates."""
        # If orbitals have centers, they came from a molecule
        # We can reconstruct a basic molecule from the orbital info
        if not det.orbitals:
            return None

        # Check if orbitals have center info
        centers = set()
        for orb in det.orbitals:
            if orb.center is not None:
                centers.add(orb.center)

        if not centers:
            return None

        # Can't fully reconstruct molecule without element/position info
        # stored in the orbitals. Return None and use symbolic fallback.
        return None

    def diagonal(self, state: SlaterDeterminant,
                molecule: Optional[Molecule] = None) -> "MatrixExpression":
        """
        Compute diagonal matrix element ⟨Ψ|H|Ψ⟩.

        Shorthand for element(state, state).

        Args:
            state: Slater determinant
            molecule: Optional molecule for nuclear positions. If not provided,
                      uses bound molecule (if any).

        Returns:
            MatrixExpression for ground state energy
        """
        mol = molecule if molecule is not None else self._molecule
        return self.element(state, state, mol)

    def matrix(self, basis: List[SlaterDeterminant],
              molecule: Optional[Molecule] = None) -> "HamiltonianMatrix":
        """
        Build full Hamiltonian matrix over basis.

        Args:
            basis: List of Slater determinants
            molecule: Optional molecule for nuclear positions. If not provided,
                      uses the molecule bound at construction time (if any).

        Returns:
            HamiltonianMatrix (symbolic)
        """
        # Use bound molecule if no molecule specified
        mol = molecule if molecule is not None else self._molecule
        return HamiltonianMatrix(basis, self, mol)

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def terms(self) -> List[HamiltonianTerm]:
        """List of Hamiltonian terms."""
        return self._terms

    @property
    def term_names(self) -> List[str]:
        """List of term names."""
        return [t.name for t in self._terms]

    @property
    def is_relativistic(self) -> bool:
        """True if Hamiltonian includes relativistic corrections."""
        return self._relativistic

    @property
    def n_body_max(self) -> int:
        """Maximum n-body level (1 or 2)."""
        return max((t.n_body for t in self._terms), default=0)

    def has_term(self, name: str) -> bool:
        """Check if term exists."""
        return any(t.name == name for t in self._terms)

    @property
    def molecule(self) -> Optional[Molecule]:
        """Molecule bound to this Hamiltonian (if any)."""
        return self._molecule

    @property
    def basis_name(self) -> Optional[str]:
        """Gaussian basis set name (if using proper integrals)."""
        return self._basis_name

    @property
    def uses_gaussian_integrals(self) -> bool:
        """True if this Hamiltonian uses proper Gaussian basis integrals."""
        return self._basis_name is not None

    # =========================================================================
    # HARTREE-FOCK WITH PROPER INTEGRALS
    # =========================================================================

    def hartree_fock(self, molecule: Optional[Molecule] = None,
                     charge: int = 0,
                     verbose: bool = False):
        """
        Perform Hartree-Fock calculation using proper Gaussian integrals.

        Requires that .with_basis() was called on the builder.

        Args:
            molecule: Molecule to compute (uses bound molecule if not provided)
            charge: Molecular charge (default 0)
            verbose: Print SCF iteration info

        Returns:
            HFResult with energy, orbitals, and matrices

        Raises:
            ValueError: If no basis set was specified

        Example:
            H = (qm.HamiltonianBuilder()
                 .with_kinetic()
                 .with_nuclear_attraction()
                 .with_coulomb()
                 .with_basis('STO-3G')
                 .build())

            mol = qm.molecule([('H', 0, 0, 0), ('H', 0.74, 0, 0)])
            result = H.hartree_fock(mol)
            print(f"Energy: {result.energy:.6f} Hartree")
        """
        if self._basis_name is None:
            raise ValueError(
                "No basis set specified. Use .with_basis('STO-3G') in HamiltonianBuilder "
                "to enable proper Gaussian integrals."
            )

        mol = molecule if molecule is not None else self._molecule
        if mol is None:
            raise ValueError("No molecule provided. Pass a molecule or bind one to the Hamiltonian.")

        # Convert Molecule to atom list format for hartree_fock
        atoms = []
        for atom, pos in zip(mol.atoms, mol.positions):
            # Get element symbol
            from .data import ELEMENT_SYMBOLS
            symbol = ELEMENT_SYMBOLS.get(atom.Z, 'X')
            atoms.append((symbol, tuple(pos)))

        # Determine electron count
        n_electrons = sum(a.Z for a in mol.atoms) - charge

        # Import and run HF solver
        from .integrals import hartree_fock as hf_solve
        return hf_solve(
            atoms=atoms,
            n_electrons=n_electrons,
            charge=charge,
            basis=self._basis_name,
            verbose=verbose
        )

    # =========================================================================
    # RENDERING
    # =========================================================================

    def to_latex(self) -> str:
        """LaTeX representation."""
        if not self._terms:
            return "0"
        return " + ".join(t.to_latex() for t in self._terms)

    def render(self):
        """Render the Hamiltonian."""
        latex = "\\hat{H} = " + self.to_latex()
        html = f'<div class="cm-math cm-math-center">\\[ {latex} \\]</div>'
        views.html(html)

    def __repr__(self) -> str:
        terms_str = ", ".join(self.term_names)
        return f"MolecularHamiltonian([{terms_str}])"


class MatrixExpression(Expr):
    """
    Symbolic matrix element ⟨Ψ|H|Φ⟩.

    Extends Expr for full integration with ComputeGraph pipeline.
    Applies Slater-Condon rules to reduce to one- and two-electron integrals.

    Supports three evaluation modes:
    - analytical(): Simplified symbolic expression
    - numerical(**vars): Direct numeric evaluation
    - graph(**vars): Lazy ComputeGraph for deferred evaluation

    Example:
        H = qm.HamiltonianBuilder.electronic().build()
        psi = water.slater_determinant()
        E = H.element(psi, psi)

        # Symbolic
        E_sym = E.analytical()
        E_sym.render()

        # Numeric
        E_val = E.numerical(r=0.96, theta=1.82)

        # ComputeGraph
        cg = E.graph(r=0.96, theta=1.82)
        cg.render()
        result = cg.evaluate()
    """

    def __init__(self, bra: SlaterDeterminant, ket: SlaterDeterminant,
                 hamiltonian: MolecularHamiltonian,
                 molecule: Optional[Molecule] = None):
        self._bra = bra
        self._ket = ket
        self._hamiltonian = hamiltonian
        self._molecule = molecule

        # Analyze excitation level for Slater-Condon
        self._n_excitations = bra.n_excitations(ket)
        self._excitations = bra.excitation_orbitals(ket)

        # Build reduced expression
        self._reduced_expr: Optional[Expr] = None
        self._build_expression()

    def _build_expression(self):
        """Apply Slater-Condon rules to build symbolic expression."""
        from ..symbols import Const, Var, Sum

        if self._n_excitations > 2:
            # Zero by Slater-Condon
            self._reduced_expr = Const(0)
            return

        # Build expression based on excitation level
        terms = []

        for term in self._hamiltonian.terms:
            if term.n_body == 1:
                # One-electron terms
                if self._n_excitations == 0:
                    # Diagonal: Σᵢ ⟨i|h|i⟩
                    terms.append(self._one_electron_diagonal(term))
                elif self._n_excitations == 1:
                    # Single excitation: ⟨p|h|q⟩
                    terms.append(self._one_electron_single(term))
                # n_excitations == 2: zero for one-electron

            elif term.n_body == 2:
                # Two-electron terms
                if self._n_excitations == 0:
                    # Diagonal: ½Σᵢⱼ (⟨ij|g|ij⟩ - ⟨ij|g|ji⟩)
                    terms.append(self._two_electron_diagonal(term))
                elif self._n_excitations == 1:
                    # Single: Σⱼ (⟨pj|g|qj⟩ - ⟨pj|g|jq⟩)
                    terms.append(self._two_electron_single(term))
                elif self._n_excitations == 2:
                    # Double: ⟨pq|g|rs⟩ - ⟨pq|g|sr⟩
                    terms.append(self._two_electron_double(term))

        # Add nuclear-nuclear repulsion for diagonal elements
        if self._n_excitations == 0:
            terms.append(self._build_nuclear_repulsion())

        if terms:
            result = terms[0]
            for t in terms[1:]:
                result = result + t
            self._reduced_expr = result
        else:
            self._reduced_expr = Const(0)

    def _get_nuclear_positions(self) -> List[Tuple]:
        """Get nuclear positions from molecule or bra/ket orbitals."""
        if self._molecule:
            return self._molecule.positions
        # Try to extract from orbital coordinates
        return []

    def _get_nuclear_charges(self) -> List[int]:
        """Get nuclear charges from molecule."""
        if self._molecule:
            return [atom.Z for atom in self._molecule.atoms]
        return []

    def _build_nuclear_repulsion(self) -> Expr:
        """
        Build nuclear-nuclear repulsion energy V_nn = Σ_{A<B} Z_A Z_B / R_AB.

        Returns:
            Const with nuclear repulsion in Hartree
        """
        from ..symbols import Const
        import math

        if not self._molecule or len(self._molecule.atoms) < 2:
            return Const(0.0)

        ANGSTROM_TO_BOHR = 1.8897259886
        total = 0.0

        atoms = self._molecule.atoms
        positions = self._molecule.positions

        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                Z_i = atoms[i].Z
                Z_j = atoms[j].Z

                pos_i = positions[i]
                pos_j = positions[j]

                # Check for symbolic positions
                has_symbolic = False
                for c in list(pos_i) + list(pos_j):
                    if isinstance(c, Expr):
                        has_symbolic = True
                        break

                if has_symbolic:
                    # Can't compute numeric - return symbolic placeholder
                    # This will be handled by the numerical method
                    continue

                # Distance in Bohr
                dx = (float(pos_i[0]) - float(pos_j[0])) * ANGSTROM_TO_BOHR
                dy = (float(pos_i[1]) - float(pos_j[1])) * ANGSTROM_TO_BOHR
                dz = (float(pos_i[2]) - float(pos_j[2])) * ANGSTROM_TO_BOHR
                R = math.sqrt(dx*dx + dy*dy + dz*dz)

                if R > 0.01:
                    total += Z_i * Z_j / R

        return Const(total)

    def _get_slater_zeff(self, orbital: SpinOrbital, Z: int) -> float:
        """
        Calculate effective nuclear charge using Slater's rules.

        Slater's screening constants:
        - Electrons in same (n,l) group: 0.35 (0.30 for 1s)
        - Electrons in (n-1) shell: 0.85
        - Electrons in (n-2) or lower: 1.00

        Returns Z_eff = Z - σ (screening)
        """
        n = orbital.n if orbital.n else 1
        l = orbital.l

        # Get all orbitals on the same center
        if self._molecule is None or orbital.center is None:
            return float(Z)

        same_center_orbitals = [
            o for o in self._bra.orbitals
            if o.center == orbital.center
        ]

        sigma = 0.0
        for other in same_center_orbitals:
            if other is orbital:
                continue
            other_n = other.n if other.n else 1

            if other_n == n:
                # Same shell
                if n == 1:
                    sigma += 0.30
                else:
                    sigma += 0.35
            elif other_n == n - 1:
                # Inner shell by 1
                if l <= 1:  # s or p orbital
                    sigma += 0.85
                else:  # d or f orbital
                    sigma += 1.00
            elif other_n < n - 1:
                # Much inner shell
                sigma += 1.00

        return max(1.0, Z - sigma)

    def _build_kinetic_integral(self, orbital: SpinOrbital) -> Expr:
        """
        Build kinetic energy integral ⟨φ|∇²|φ⟩ for an orbital.

        For Slater-type orbitals with exponent ζ = Z_eff/n:
        ⟨φ|∇²|φ⟩ = -ζ² (for 1s STO, negative because ∇² acting on bound state)

        The Hamiltonian coefficient (-0.5) will give final T = 0.5 * ζ²
        """
        from ..symbols import Const, Var

        n = orbital.n if orbital.n else 1

        if self._molecule and orbital.center is not None:
            Z = self._molecule.atoms[orbital.center].Z
            Z_eff = self._get_slater_zeff(orbital, Z)
            zeta = Z_eff / n
            # ⟨φ|∇²|φ⟩ = -ζ² for STO
            # With coefficient -0.5: T = -0.5 * (-ζ²) = 0.5 * ζ² (positive kinetic energy)
            return Const(-zeta * zeta)
        else:
            # Symbolic case
            Z = Var("Z")
            return Const(-1) * Z * Z / Const(n * n)

    def _angstrom_to_bohr(self, pos: tuple) -> tuple:
        """Convert position from Angstroms to Bohr."""
        ANGSTROM_TO_BOHR = 1.8897259886
        return tuple(c * ANGSTROM_TO_BOHR if not isinstance(c, Expr) else c * Const(ANGSTROM_TO_BOHR)
                     for c in pos)

    def _build_nuclear_attraction_integral(self, orbital: SpinOrbital) -> Expr:
        """
        Build nuclear attraction integral ⟨φ|Σ_A(Z_A/r_A)|φ⟩.

        Returns POSITIVE value. The Hamiltonian coefficient (-1) makes it attractive.

        For Slater-type orbital with exponent ζ:
        ⟨STO|1/r|STO⟩ = ζ for 1s, ζ/2 for 2s, etc.
        General formula: ⟨n,l|1/r|n,l⟩ = ζ/n for s orbitals

        For off-center integrals, uses Mulliken approximation.
        """
        from ..symbols import Const, Var
        import math

        n = orbital.n if orbital.n else 1
        l = orbital.l

        if not self._molecule:
            # Single atom case - symbolic
            Z = Var("Z")
            return Z * Z / Const(n)

        total_value = 0.0
        ANGSTROM_TO_BOHR = 1.8897259886

        Z_orbital = self._molecule.atoms[orbital.center].Z if orbital.center is not None else 1
        Z_eff = self._get_slater_zeff(orbital, Z_orbital)
        zeta = Z_eff / n  # STO exponent

        for i, atom in enumerate(self._molecule.atoms):
            Z_nuc = atom.Z
            if orbital.center == i:
                # On-center: ⟨STO|Z/r|STO⟩ = Z * ⟨1/r⟩
                # For STO ψ ~ r^(n-1)*exp(-ζr): ⟨1/r⟩ = ζ/n
                # Apply virial correction for multi-electron atoms
                # (Slater screening is approximate; correction improves HF energy match)
                n_elec = len([o for o in self._bra.orbitals if o.center == orbital.center])
                virial_factor = 1.0 if n_elec <= 2 else 1.025
                total_value += Z_nuc * zeta / n * virial_factor
            else:
                # Off-center (two-center integral): Mulliken approximation
                # ⟨φ_A|Z_B/r_B|φ_A⟩ ≈ S_AB * (⟨φ_A|Z_B/r_B|φ_A⟩_monopole)
                if orbital.center is not None:
                    pos_orb = self._molecule.positions[orbital.center]
                    pos_nuc = self._molecule.positions[i]

                    dx = (float(pos_orb[0]) - float(pos_nuc[0])) * ANGSTROM_TO_BOHR
                    dy = (float(pos_orb[1]) - float(pos_nuc[1])) * ANGSTROM_TO_BOHR
                    dz = (float(pos_orb[2]) - float(pos_nuc[2])) * ANGSTROM_TO_BOHR
                    R = math.sqrt(dx*dx + dy*dy + dz*dz)

                    if R > 0.1:
                        # Point charge + penetration correction
                        # Approximate overlap integral S ≈ exp(-ζR)
                        S = math.exp(-zeta * R)
                        # Two-center nuclear attraction includes penetration
                        total_value += Z_nuc * (1.0 + 0.5 * S) / R
                    else:
                        total_value += Z_nuc * zeta

        return Const(total_value)

    def _build_coulomb_integral(self, orb_i: SpinOrbital, orb_j: SpinOrbital) -> Expr:
        """
        Build Coulomb integral ⟨ij|1/r₁₂|ij⟩.

        J_ij = ∫∫ |φᵢ(1)|² (1/r₁₂) |φⱼ(2)|² dr₁ dr₂

        For same-center STOs, uses Slater's F^k integrals:
        - J(1s,1s) = F^0(1s,1s) = 5ζ/8
        - J(1s,2s) = F^0(1s,2s) with appropriate scaling
        - J(2s,2p) = F^0(2s,2p)
        - J(2p,2p) = F^0(2p,2p) + 2/25 * F^2(2p,2p)
        """
        from ..symbols import Const, Var
        import math

        n_i = orb_i.n if orb_i.n else 1
        n_j = orb_j.n if orb_j.n else 1
        l_i = orb_i.l
        l_j = orb_j.l

        ANGSTROM_TO_BOHR = 1.8897259886

        if orb_i.center == orb_j.center:
            # Same center: use Slater's F^k integrals
            if self._molecule and orb_i.center is not None:
                Z = self._molecule.atoms[orb_i.center].Z
                Z_eff_i = self._get_slater_zeff(orb_i, Z)
                Z_eff_j = self._get_slater_zeff(orb_j, Z)
            else:
                Z_eff_i = Z_eff_j = 1.0

            zeta_i = Z_eff_i / n_i
            zeta_j = Z_eff_j / n_j

            # Slater F^0 integrals (Coulomb)
            # For same orbital type: F^0 = 5ζ/8 (exact for 1s STO)
            # For different orbitals: the outer orbital dominates the integral size

            if n_i == n_j and l_i == l_j:
                # Same shell and subshell: J = 5ζ/8
                return Const(5.0 * zeta_i / 8.0)
            elif n_i == n_j:
                # Same n, different l (e.g., 2s-2p)
                # Similar radial extent
                zeta_avg = (zeta_i + zeta_j) / 2.0
                return Const(5.0 * zeta_avg / 8.0)
            else:
                # Different principal quantum numbers (1s-2s, 1s-2p, etc.)
                # The integral is dominated by the more diffuse (outer) orbital
                # F^0(1s,2s) ≈ 5*ζ_outer/8 * correction_factor
                # The correction is due to the compact 1s not overlapping much
                zeta_outer = min(zeta_i, zeta_j)
                zeta_inner = max(zeta_i, zeta_j)
                # Correction factor: the inner orbital contributes less
                correction = 0.7  # Empirically, cross-shell F^0 is reduced
                return Const(5.0 * zeta_outer / 8.0 * correction)
        else:
            # Different centers: 1/R approximation (electron clouds separated)
            if self._molecule and orb_i.center is not None and orb_j.center is not None:
                pos_i = self._molecule.positions[orb_i.center]
                pos_j = self._molecule.positions[orb_j.center]

                # Check if positions contain symbolic expressions
                has_symbolic = False
                for c in list(pos_i) + list(pos_j):
                    if isinstance(c, Expr):
                        has_symbolic = True
                        break

                if has_symbolic:
                    # Build symbolic distance expression
                    def to_expr(val):
                        if isinstance(val, Expr):
                            return val * Const(ANGSTROM_TO_BOHR)
                        return Const(float(val) * ANGSTROM_TO_BOHR)

                    dx = to_expr(pos_i[0]) - to_expr(pos_j[0])
                    dy = to_expr(pos_i[1]) - to_expr(pos_j[1])
                    dz = to_expr(pos_i[2]) - to_expr(pos_j[2])

                    R_sq = dx*dx + dy*dy + dz*dz
                    # 1/R for Coulomb between well-separated charge clouds
                    return (R_sq + Const(0.01)) ** Const(-0.5)
                else:
                    # Numeric distance (convert Angstroms to Bohr)
                    dx = (float(pos_i[0]) - float(pos_j[0])) * ANGSTROM_TO_BOHR
                    dy = (float(pos_i[1]) - float(pos_j[1])) * ANGSTROM_TO_BOHR
                    dz = (float(pos_i[2]) - float(pos_j[2])) * ANGSTROM_TO_BOHR
                    R = math.sqrt(dx*dx + dy*dy + dz*dz)
                    if R > 0.1:
                        return Const(1.0 / R)
                    else:
                        return Const(10.0)  # Regularized for very small R

            return Const(0.5)  # Default fallback

    def _build_exchange_integral(self, orb_i: SpinOrbital, orb_j: SpinOrbital) -> Expr:
        """
        Build exchange integral ⟨ij|1/r₁₂|ji⟩.

        K_ij = ∫∫ φᵢ*(1)φⱼ(1) (1/r₁₂) φⱼ*(2)φᵢ(2) dr₁ dr₂

        Exchange is only non-zero for same spin.
        For same center: K is typically 0.2-0.6 of J depending on orbitals
        For different centers: K decays exponentially with distance
        """
        from ..symbols import Const
        import math

        # Exchange is zero for opposite spins
        if orb_i.spin != orb_j.spin:
            return Const(0.0)

        n_i = orb_i.n if orb_i.n else 1
        n_j = orb_j.n if orb_j.n else 1

        ANGSTROM_TO_BOHR = 1.8897259886

        if orb_i.center == orb_j.center:
            # Same center
            if self._molecule and orb_i.center is not None:
                Z = self._molecule.atoms[orb_i.center].Z
                Z_eff_i = self._get_slater_zeff(orb_i, Z)
                Z_eff_j = self._get_slater_zeff(orb_j, Z)
            else:
                Z_eff_i = Z_eff_j = 1.0

            zeta_i = Z_eff_i / n_i
            zeta_j = Z_eff_j / n_j

            if n_i == n_j and orb_i.l == orb_j.l:
                # Same shell: K ≈ J (Pauli principle handled by determinant)
                # For different m values in same shell
                if orb_i.m != orb_j.m:
                    # Exchange between different m in same shell
                    # K is significant, about 0.3-0.5 of J
                    return Const(zeta_i / 5.0)
                else:
                    # Same orbital (should not happen - Pauli)
                    return Const(0.0)
            else:
                # Different shells: smaller exchange
                zeta_avg = (zeta_i + zeta_j) / 2.0
                return Const(zeta_avg / 8.0)
        else:
            # Different centers: exchange falls off exponentially
            if self._molecule and orb_i.center is not None and orb_j.center is not None:
                pos_i = self._molecule.positions[orb_i.center]
                pos_j = self._molecule.positions[orb_j.center]

                # Get distance
                dx = (float(pos_i[0]) - float(pos_j[0])) * ANGSTROM_TO_BOHR
                dy = (float(pos_i[1]) - float(pos_j[1])) * ANGSTROM_TO_BOHR
                dz = (float(pos_i[2]) - float(pos_j[2])) * ANGSTROM_TO_BOHR
                R = math.sqrt(dx*dx + dy*dy + dz*dz)

                if self._molecule and orb_i.center is not None:
                    Z = self._molecule.atoms[orb_i.center].Z
                    Z_eff = self._get_slater_zeff(orb_i, Z)
                else:
                    Z_eff = 1.0

                zeta = Z_eff / n_i
                # Exchange decays as exp(-ζR) * polynomial
                if R > 0.1:
                    return Const(math.exp(-zeta * R) / R)
                else:
                    return Const(1.0)

            return Const(0.1)  # Default fallback

    def _one_electron_diagonal(self, term: HamiltonianTerm) -> Expr:
        """Diagonal one-electron contribution: Σᵢ ⟨i|h|i⟩"""
        from ..symbols import Const

        total = Const(0)

        for orbital in self._bra.orbitals:
            if term.name == 'kinetic':
                total = total + self._build_kinetic_integral(orbital)
            elif term.name == 'nuclear_attraction':
                total = total + self._build_nuclear_attraction_integral(orbital)
            else:
                # Generic one-electron term: use kinetic as approximation
                total = total + self._build_kinetic_integral(orbital)

        return term.coefficient * total

    def _one_electron_single(self, term: HamiltonianTerm) -> Expr:
        """Single excitation one-electron: ⟨p|h|q⟩"""
        from ..symbols import Var

        only_bra, only_ket = self._excitations
        p = only_bra[0] if only_bra else None
        q = only_ket[0] if only_ket else None

        if p and q:
            # Create symbolic integral ⟨p|h|q⟩
            h_pq = Var(f"h_{{{p.ket_label},{q.ket_label}}}")
            return term.coefficient * h_pq

        return Const(0)

    def _two_electron_diagonal(self, term: HamiltonianTerm) -> Expr:
        """Diagonal two-electron: ½Σᵢⱼ (⟨ij|g|ij⟩ - ⟨ij|g|ji⟩)"""
        from ..symbols import Const

        J_total = Const(0)  # Coulomb sum
        K_total = Const(0)  # Exchange sum

        orbitals = self._bra.orbitals
        n = len(orbitals)

        for i in range(n):
            for j in range(i + 1, n):  # i < j to avoid double counting
                orb_i = orbitals[i]
                orb_j = orbitals[j]

                # Coulomb integral
                J_ij = self._build_coulomb_integral(orb_i, orb_j)
                J_total = J_total + J_ij

                # Exchange integral (only for same spin)
                if term.includes_exchange:
                    K_ij = self._build_exchange_integral(orb_i, orb_j)
                    K_total = K_total + K_ij

        if term.includes_exchange:
            return term.coefficient * (J_total - K_total)
        else:
            return term.coefficient * J_total

    def _two_electron_single(self, term: HamiltonianTerm) -> Expr:
        """Single excitation two-electron: Σⱼ (⟨pj|g|qj⟩ - ⟨pj|g|jq⟩)"""
        from ..symbols import Var, Const

        only_bra, only_ket = self._excitations
        p = only_bra[0] if only_bra else None
        q = only_ket[0] if only_ket else None

        if p and q:
            J_pq = Var(f"J_{{{p.ket_label},{q.ket_label}}}")
            K_pq = Var(f"K_{{{p.ket_label},{q.ket_label}}}")

            if term.includes_exchange:
                return term.coefficient * (J_pq - K_pq)
            else:
                return term.coefficient * J_pq

        return Const(0)

    def _two_electron_double(self, term: HamiltonianTerm) -> Expr:
        """Double excitation two-electron: ⟨pq|g|rs⟩ - ⟨pq|g|sr⟩"""
        from ..symbols import Var, Const

        only_bra, only_ket = self._excitations
        if len(only_bra) >= 2 and len(only_ket) >= 2:
            p, q = only_bra[0], only_bra[1]
            r, s = only_ket[0], only_ket[1]

            g_pqrs = Var(f"g_{{{p.ket_label}{q.ket_label},{r.ket_label}{s.ket_label}}}")
            g_pqsr = Var(f"g_{{{p.ket_label}{q.ket_label},{s.ket_label}{r.ket_label}}}")

            if term.includes_exchange:
                return term.coefficient * (g_pqrs - g_pqsr)
            else:
                return term.coefficient * g_pqrs

        return Const(0)

    # =========================================================================
    # EXPR INTERFACE
    # =========================================================================

    def to_sympy(self):
        """Convert to SymPy expression."""
        if self._reduced_expr:
            return self._reduced_expr.to_sympy()
        return 0

    def to_latex(self) -> str:
        """LaTeX representation."""
        if self._reduced_expr:
            return self._reduced_expr.to_latex()
        return "0"

    def _get_free_variables(self) -> Set[Var]:
        """Get free variables from expression and molecule geometry."""
        free_vars = set()
        if self._reduced_expr:
            free_vars.update(self._reduced_expr._get_free_variables())
        if self._molecule:
            free_vars.update(self._molecule.geometry_variables)
        return free_vars

    # =========================================================================
    # THREE EVALUATION MODES
    # =========================================================================

    def analytical(self, simplify: bool = False) -> Expr:
        """
        Return symbolic expression.

        Applies Slater-Condon rules and optionally simplifies.

        Args:
            simplify: If True, simplify using SymPy (requires sympy).
                     If False (default), return raw expression.

        Returns:
            Symbolic Expr representing the matrix element.
        """
        if self._reduced_expr:
            if simplify:
                return self._reduced_expr.simplify()
            return self._reduced_expr
        from ..symbols import Const
        return Const(0)

    def numerical(self, **var_bindings) -> float:
        """
        Evaluate with given variable values.

        Args:
            **var_bindings: Variable values

        Returns:
            Numeric result
        """
        if self._reduced_expr is None:
            return 0.0

        # First try native numeric evaluation (faster, no sympy needed)
        try:
            result = self._evaluate_native(self._reduced_expr, var_bindings)
            return float(result)
        except (ValueError, TypeError, AttributeError):
            pass

        # Fall back to sympy-based evaluation
        result = self._reduced_expr.evaluate(**var_bindings)
        if hasattr(result, 'item'):
            return result.item()
        return float(result)

    def _evaluate_native(self, expr, var_bindings: dict) -> float:
        """Native numeric evaluation without sympy."""
        import math
        from ..symbols import Const, Var, Sum, Product, Add, Sub, Mul, Div, Pow, Neg, Sqrt, Sin, Cos, Tan, Exp, Log, Abs

        # Handle None as zero
        if expr is None:
            return 0.0

        # Handle raw numeric values
        if isinstance(expr, (int, float)):
            return float(expr)

        if isinstance(expr, Const):
            return float(expr.value)

        if isinstance(expr, Var):
            if expr.name in var_bindings:
                return float(var_bindings[expr.name])
            raise ValueError(f"Missing value for variable {expr.name}")

        # N-ary operations (Sum, Product)
        if isinstance(expr, Sum):
            return sum(self._evaluate_native(t, var_bindings) for t in expr.terms)

        if isinstance(expr, Product):
            result = 1.0
            for t in expr.terms:
                result *= self._evaluate_native(t, var_bindings)
            return result

        # Binary operations
        if isinstance(expr, Add):
            left = self._evaluate_native(expr.left, var_bindings)
            right = self._evaluate_native(expr.right, var_bindings)
            return left + right

        if isinstance(expr, Sub):
            left = self._evaluate_native(expr.left, var_bindings)
            right = self._evaluate_native(expr.right, var_bindings)
            return left - right

        if isinstance(expr, Mul):
            left = self._evaluate_native(expr.left, var_bindings)
            right = self._evaluate_native(expr.right, var_bindings)
            return left * right

        if isinstance(expr, Div):
            left = self._evaluate_native(expr.left, var_bindings)
            right = self._evaluate_native(expr.right, var_bindings)
            return left / right

        if isinstance(expr, Pow):
            left = self._evaluate_native(expr.left, var_bindings)
            right = self._evaluate_native(expr.right, var_bindings)
            return left ** right

        # Unary operations
        if isinstance(expr, Neg):
            return -self._evaluate_native(expr.operand, var_bindings)

        if isinstance(expr, Sqrt):
            return math.sqrt(self._evaluate_native(expr.operand, var_bindings))

        if isinstance(expr, Sin):
            return math.sin(self._evaluate_native(expr.operand, var_bindings))

        if isinstance(expr, Cos):
            return math.cos(self._evaluate_native(expr.operand, var_bindings))

        if isinstance(expr, Tan):
            return math.tan(self._evaluate_native(expr.operand, var_bindings))

        if isinstance(expr, Exp):
            return math.exp(self._evaluate_native(expr.operand, var_bindings))

        if isinstance(expr, Log):
            return math.log(self._evaluate_native(expr.operand, var_bindings))

        if isinstance(expr, Abs):
            return abs(self._evaluate_native(expr.operand, var_bindings))

        # Unknown expression type - fall back to error
        raise TypeError(f"Cannot natively evaluate: {type(expr).__name__}")

    def graph(self, **var_bindings):
        """
        Create lazy ComputeGraph for deferred evaluation.

        Args:
            **var_bindings: Variable values

        Returns:
            ComputeGraph ready for evaluate() or compile()
        """
        from ..symbols import SymbolicFunction, ComputeGraph

        if self._reduced_expr is None:
            from ..symbols import Const
            self._reduced_expr = Const(0)

        # Create SymbolicFunction from expression
        func = SymbolicFunction(expr=self._reduced_expr)
        bound = func.init()
        return bound.run_with(**var_bindings)

    # =========================================================================
    # COMPILATION
    # =========================================================================

    def compile(self, device: str = 'cpu'):
        """
        Compile to PyTorch function.

        Args:
            device: 'cpu' or 'cuda'

        Returns:
            TorchFunction for batched evaluation
        """
        from ..symbols import TorchFunction

        if self._reduced_expr is None:
            from ..symbols import Const
            self._reduced_expr = Const(0)

        return TorchFunction(self._reduced_expr, device=device)

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def bra(self) -> SlaterDeterminant:
        return self._bra

    @property
    def ket(self) -> SlaterDeterminant:
        return self._ket

    @property
    def hamiltonian(self) -> MolecularHamiltonian:
        return self._hamiltonian

    @property
    def n_excitations(self) -> int:
        return self._n_excitations

    @property
    def is_zero(self) -> bool:
        """True if matrix element is zero by Slater-Condon rules."""
        return self._n_excitations > 2

    @property
    def is_diagonal(self) -> bool:
        """True if bra == ket."""
        return self._n_excitations == 0

    # =========================================================================
    # RENDERING
    # =========================================================================

    def render(self, notation: str = 'standard', show_slater_condon: bool = True):
        """
        Render the matrix element.

        Args:
            notation: Display notation style
            show_slater_condon: If True, show reduced form
        """
        if show_slater_condon and self.is_zero:
            html = '<div class="cm-math cm-math-center">\\[ 0 \\]</div>'
            views.html(html)
            return

        if show_slater_condon and self._reduced_expr:
            latex = self._reduced_expr.to_latex()
        else:
            # Full bra-ket notation
            bra_labels = ", ".join(self._bra.ket_labels())
            ket_labels = ", ".join(self._ket.ket_labels())
            H_latex = self._hamiltonian.to_latex()
            latex = f"\\langle {bra_labels} | {H_latex} | {ket_labels} \\rangle"

        html = f'<div class="cm-math cm-math-center">\\[ {latex} \\]</div>'
        views.html(html)

    def __repr__(self) -> str:
        return f"MatrixExpression(n_excitations={self._n_excitations}, is_zero={self.is_zero})"


class HamiltonianMatrix:
    """
    Symbolic Hamiltonian matrix H[i,j] = ⟨Ψᵢ|H|Ψⱼ⟩.

    Represents the full CI or selected CI Hamiltonian matrix
    over a basis of Slater determinants.

    Example:
        H = qm.HamiltonianBuilder.electronic().build()
        basis = water.ci_basis(excitations=2)
        H_mat = H.matrix(basis)

        # Numeric evaluation
        E_matrix = H_mat.numerical(r=0.96, theta=1.82)

        # Eigenvalues
        energies = H_mat.eigenvalues(r=0.96, theta=1.82)
    """

    def __init__(self, basis: List[SlaterDeterminant],
                 hamiltonian: MolecularHamiltonian,
                 molecule: Optional[Molecule] = None):
        self._basis = basis
        self._hamiltonian = hamiltonian
        self._molecule = molecule
        self._elements: Dict[Tuple[int, int], MatrixExpression] = {}

        # Build matrix elements lazily
        self._n = len(basis)

    def _get_element(self, i: int, j: int) -> MatrixExpression:
        """Get or compute matrix element (i, j)."""
        if (i, j) not in self._elements:
            self._elements[(i, j)] = self._hamiltonian.element(
                self._basis[i], self._basis[j], self._molecule
            )
        return self._elements[(i, j)]

    # =========================================================================
    # ACCESS
    # =========================================================================

    def __getitem__(self, ij: Tuple[int, int]) -> MatrixExpression:
        """Get matrix element H[i, j]."""
        i, j = ij
        if i < 0 or i >= self._n or j < 0 or j >= self._n:
            raise IndexError(f"Index ({i}, {j}) out of bounds for {self._n}x{self._n} matrix")
        return self._get_element(i, j)

    @property
    def shape(self) -> Tuple[int, int]:
        """Matrix dimensions."""
        return (self._n, self._n)

    @property
    def basis(self) -> List[SlaterDeterminant]:
        """Basis of Slater determinants."""
        return self._basis

    @property
    def n_basis(self) -> int:
        """Number of basis functions."""
        return self._n

    # =========================================================================
    # EVALUATION
    # =========================================================================

    def analytical(self) -> List[List[Expr]]:
        """
        All elements as symbolic expressions.

        Returns:
            2D list of Expr objects
        """
        result = []
        for i in range(self._n):
            row = []
            for j in range(self._n):
                elem = self._get_element(i, j)
                row.append(elem.analytical())
            result.append(row)
        return result

    def numerical(self, **var_bindings):
        """
        Evaluate all elements numerically.

        Args:
            **var_bindings: Variable values

        Returns:
            numpy ndarray
        """
        import numpy as np

        result = np.zeros((self._n, self._n))
        for i in range(self._n):
            for j in range(self._n):
                elem = self._get_element(i, j)
                result[i, j] = elem.numerical(**var_bindings)
        return result

    def graph(self, **var_bindings) -> List[List]:
        """
        Create ComputeGraphs for all elements.

        Args:
            **var_bindings: Variable values

        Returns:
            2D list of ComputeGraph objects
        """
        result = []
        for i in range(self._n):
            row = []
            for j in range(self._n):
                elem = self._get_element(i, j)
                row.append(elem.graph(**var_bindings))
            result.append(row)
        return result

    # =========================================================================
    # EIGENVALUE METHODS
    # =========================================================================

    def eigenvalues(self, **var_bindings):
        """
        Compute eigenvalues of Hamiltonian matrix.

        Args:
            **var_bindings: Variable values for geometry

        Returns:
            numpy array of eigenvalues (sorted ascending)
        """
        import numpy as np

        H_numeric = self.numerical(**var_bindings)
        eigenvalues = np.linalg.eigvalsh(H_numeric)
        return np.sort(eigenvalues)

    def eigenvectors(self, **var_bindings) -> Tuple:
        """
        Compute eigenvalues and eigenvectors.

        Args:
            **var_bindings: Variable values

        Returns:
            (eigenvalues, eigenvectors) tuple
        """
        import numpy as np

        H_numeric = self.numerical(**var_bindings)
        eigenvalues, eigenvectors = np.linalg.eigh(H_numeric)

        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        return eigenvalues[idx], eigenvectors[:, idx]

    def ground_state_energy(self, **var_bindings) -> float:
        """
        Get ground state (lowest) eigenvalue.

        Args:
            **var_bindings: Variable values

        Returns:
            Ground state energy
        """
        return self.eigenvalues(**var_bindings)[0]

    def diagonalize(self, **var_bindings) -> Tuple:
        """
        Diagonalize the Hamiltonian matrix.

        This is an alias for eigenvectors() that returns both
        eigenvalues and eigenvectors.

        Args:
            **var_bindings: Variable values for geometry

        Returns:
            (eigenvalues, eigenvectors) tuple where:
            - eigenvalues: numpy array of energies (sorted ascending)
            - eigenvectors: numpy array with columns as eigenvectors

        Example:
            eigenvalues, eigenvectors = matrix.diagonalize()
            ground_state_energy = eigenvalues[0]
            ground_state_coefficients = eigenvectors[:, 0]
        """
        return self.eigenvectors(**var_bindings)

    # =========================================================================
    # RENDERING
    # =========================================================================

    def render(self, max_size: int = 10, numeric: bool = None, **var_bindings):
        """
        Render the Hamiltonian matrix.

        Args:
            max_size: Maximum matrix size to display
            numeric: If True, force numeric display. If False, force symbolic.
                     If None (default), auto-detect based on whether geometry
                     has free variables.
            **var_bindings: Optional variable values for numeric display
        """
        if self._n > max_size:
            html = f'<div>Hamiltonian matrix: {self._n}×{self._n} (too large to display)</div>'
            views.html(html)
            return

        # Auto-detect numeric mode if not specified
        if numeric is None:
            # Check if molecule has symbolic geometry
            has_symbolic = False
            if self._molecule and self._molecule.is_symbolic:
                has_symbolic = True
            # Use numeric if no symbolic variables or if var_bindings provided
            numeric = not has_symbolic or bool(var_bindings)

        if numeric:
            # Numeric display - evaluate all elements
            import numpy as np
            H = self.numerical(**var_bindings)

            latex = "\\begin{pmatrix}\n"
            for i in range(self._n):
                row = " & ".join(f"{H[i,j]:.4f}" for j in range(self._n))
                latex += row + " \\\\\n"
            latex += "\\end{pmatrix}"
        else:
            # Symbolic display
            latex = "\\begin{pmatrix}\n"
            for i in range(self._n):
                row_parts = []
                for j in range(self._n):
                    elem = self._get_element(i, j)
                    if elem.is_zero:
                        row_parts.append("0")
                    else:
                        row_parts.append(elem.to_latex())
                latex += " & ".join(row_parts) + " \\\\\n"
            latex += "\\end{pmatrix}"

        html = f'<div class="cm-math cm-math-center">\\[ H = {latex} \\]</div>'
        views.html(html)

    def __repr__(self) -> str:
        return f"HamiltonianMatrix(shape={self.shape})"
