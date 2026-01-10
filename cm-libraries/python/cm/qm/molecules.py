"""
Chemical Machines QM Molecules Module

Molecule class for multi-atom systems.
"""

from typing import Optional, List, Union, Tuple, Set, Any

from ..symbols import Expr, Var
from .atoms import Atom, atom
from .spinorbitals import SpinOrbital, SlaterDeterminant

__all__ = [
    'Molecule',
    'molecule',
]


# =============================================================================
# MOLECULE CLASS
# =============================================================================


class Molecule:
    """
    Collection of atoms with geometric arrangement.

    Positions can be numeric or symbolic (Expr) for geometry optimization.
    The geometry variables become free variables in Hamiltonian evaluation.

    Example:
        # Numeric geometry
        water = qm.Molecule([
            (qm.atom('O'), 0, 0, 0),
            (qm.atom('H'), 0.96, 0, 0),
            (qm.atom('H'), -0.24, 0.93, 0),
        ])

        # Symbolic geometry for optimization
        from cm import Math
        r, theta = Math.var('r'), Math.var('theta')
        water = qm.Molecule([
            (qm.atom('O'), 0, 0, 0),
            (qm.atom('H'), r, 0, 0),
            (qm.atom('H'), r * Math.cos(theta), r * Math.sin(theta), 0),
        ])
    """

    def __init__(self, atoms_with_positions: List[Tuple]):
        """
        Create a molecule from atoms with positions.

        Args:
            atoms_with_positions: List of (Atom, x, y, z) tuples where
                                  x, y, z can be float or Expr
        """
        self._atoms: List[Atom] = []
        self._positions: List[Tuple] = []
        self._geometry_vars: Set[Var] = set()

        for item in atoms_with_positions:
            if len(item) != 4:
                raise ValueError(f"Expected (Atom, x, y, z), got {len(item)} elements")

            atom_obj, x, y, z = item

            # Handle tuple specs like ('O', 0, 0, 0)
            if isinstance(atom_obj, str):
                atom_obj = atom(atom_obj)
            elif isinstance(atom_obj, int):
                atom_obj = atom(atom_obj)

            self._atoms.append(atom_obj)

            # Convert positions to Expr if needed
            x_expr = x if isinstance(x, Expr) else x
            y_expr = y if isinstance(y, Expr) else y
            z_expr = z if isinstance(z, Expr) else z
            self._positions.append((x_expr, y_expr, z_expr))

            # Collect geometry variables
            for coord in (x, y, z):
                if isinstance(coord, Expr):
                    self._geometry_vars.update(coord._get_free_variables())

    # =========================================================================
    # ACCESS
    # =========================================================================

    def __getitem__(self, i: int) -> Atom:
        return self._atoms[i]

    def __len__(self) -> int:
        return len(self._atoms)

    def __iter__(self):
        return iter(self._atoms)

    @property
    def atoms(self) -> List[Atom]:
        """List of atoms in the molecule."""
        return self._atoms

    @property
    def positions(self) -> List[Tuple]:
        """List of (x, y, z) positions (may contain Expr)."""
        return self._positions

    @property
    def n_atoms(self) -> int:
        """Number of atoms."""
        return len(self._atoms)

    @property
    def n_electrons(self) -> int:
        """Total number of electrons."""
        return sum(a.n_electrons for a in self._atoms)

    @property
    def nuclear_charges(self) -> List[int]:
        """List of nuclear charges Z for each atom."""
        return [a.Z for a in self._atoms]

    @property
    def total_nuclear_charge(self) -> int:
        """Sum of nuclear charges."""
        return sum(self.nuclear_charges)

    @property
    def geometry_variables(self) -> Set[Var]:
        """Free variables in geometry (for optimization)."""
        return self._geometry_vars

    @property
    def is_symbolic(self) -> bool:
        """True if geometry contains symbolic variables."""
        return len(self._geometry_vars) > 0

    # =========================================================================
    # ATOM ACCESS WITH POSITIONS
    # =========================================================================

    def atom_at(self, i: int) -> Tuple[Atom, Tuple]:
        """Get atom and its position."""
        return self._atoms[i], self._positions[i]

    def atoms_with_positions(self) -> List[Tuple[Atom, Tuple]]:
        """List of (Atom, (x, y, z)) tuples."""
        return list(zip(self._atoms, self._positions))

    # =========================================================================
    # SLATER DETERMINANTS
    # =========================================================================

    def slater_determinant(self) -> SlaterDeterminant:
        """
        Combined ground state Slater determinant from all atoms.

        Collects orbitals from all atoms and forms single determinant.
        Each orbital is labeled with its atomic center index to distinguish
        orbitals on different atoms (e.g., 1s on H_0 vs 1s on H_1).
        """
        all_orbitals = []
        for i, atom_obj in enumerate(self._atoms):
            # Create orbitals with center label to distinguish atoms
            for n, l, m, spin in atom_obj.configuration.orbitals:
                orbital = SpinOrbital(
                    atom_obj.vec3, n=n, l=l, m=m, spin=spin,
                    center=i  # Label orbital with atom index
                )
                all_orbitals.append(orbital)
        return SlaterDeterminant(all_orbitals)

    def ci_basis(self, excitations: int = 2,
                 frozen_core: Optional[int] = None,
                 active_space: Optional[Tuple[int, int]] = None) -> List[SlaterDeterminant]:
        """
        Generate CI basis with up to N excitations from ground state.

        Args:
            excitations: Maximum excitation level (1=CIS, 2=CISD, etc.)
            frozen_core: Number of lowest orbitals to freeze (not excite from)
            active_space: (n_occupied, n_virtual) to limit active orbitals

        Returns:
            List of SlaterDeterminant objects
        """
        ground = self.slater_determinant()
        basis = [ground]

        # Get all occupied orbitals
        occupied = list(ground.orbitals)
        n_occ = len(occupied)

        # Apply frozen core
        if frozen_core:
            occupied = occupied[frozen_core:]

        # Generate virtual orbitals (placeholder - would need basis set info)
        # For now, generate excited states by promoting within occupied space
        # This is simplified - real implementation needs virtual orbital basis

        # Single excitations
        if excitations >= 1:
            for i, occ_i in enumerate(occupied):
                for j, occ_j in enumerate(occupied):
                    if i != j:
                        # Simple swap as placeholder for real virtual orbital
                        new_orbitals = list(ground.orbitals)
                        # In real impl: replace occ_i with virtual orbital
                        # For now just include ground state copies with marks
                        pass

        # For full CI implementation, would need:
        # 1. Virtual orbital basis (from basis set)
        # 2. Generate all N-choose-K excitations
        # 3. Symmetry filtering

        return basis

    # =========================================================================
    # GEOMETRY OPERATIONS
    # =========================================================================

    def with_geometry(self, **var_bindings) -> "Molecule":
        """
        Return new molecule with substituted geometry.

        Args:
            **var_bindings: Variable values (e.g., r=0.96, theta=1.82)

        Returns:
            New Molecule with numeric geometry
        """
        new_positions = []
        for x, y, z in self._positions:
            new_x = x.evaluate(**var_bindings) if isinstance(x, Expr) else x
            new_y = y.evaluate(**var_bindings) if isinstance(y, Expr) else y
            new_z = z.evaluate(**var_bindings) if isinstance(z, Expr) else z
            new_positions.append((new_x, new_y, new_z))

        new_items = [(a, *pos) for a, pos in zip(self._atoms, new_positions)]
        return Molecule(new_items)

    def bond_length(self, i: int, j: int) -> Union[float, Expr]:
        """
        Distance between atoms i and j.

        Args:
            i, j: Atom indices

        Returns:
            Distance (float if numeric, Expr if symbolic)
        """
        x1, y1, z1 = self._positions[i]
        x2, y2, z2 = self._positions[j]

        dx = x2 - x1 if isinstance(x2, Expr) or isinstance(x1, Expr) else x2 - x1
        dy = y2 - y1 if isinstance(y2, Expr) or isinstance(y1, Expr) else y2 - y1
        dz = z2 - z1 if isinstance(z2, Expr) or isinstance(z1, Expr) else z2 - z1

        if isinstance(dx, Expr) or isinstance(dy, Expr) or isinstance(dz, Expr):
            from ..symbols import Sqrt
            return Sqrt(dx*dx + dy*dy + dz*dz)
        else:
            import math
            return math.sqrt(dx*dx + dy*dy + dz*dz)

    def bond_angle(self, i: int, j: int, k: int) -> Union[float, Expr]:
        """
        Angle i-j-k in radians (j is the central atom).

        Args:
            i, j, k: Atom indices

        Returns:
            Angle in radians
        """
        # Vector from j to i
        xi, yi, zi = self._positions[i]
        xj, yj, zj = self._positions[j]
        xk, yk, zk = self._positions[k]

        # For symbolic case, would need arccos implementation
        # For now, numeric only
        if self.is_symbolic:
            raise NotImplementedError("Symbolic bond angles not yet supported")

        import math
        v1 = (xi - xj, yi - yj, zi - zj)
        v2 = (xk - xj, yk - yj, zk - zj)

        dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)

        return math.acos(dot / (mag1 * mag2))

    # =========================================================================
    # ENERGY CALCULATIONS
    # =========================================================================

    def energy(self, hamiltonian: "MolecularHamiltonian") -> "MatrixExpression":
        """
        Calculate ground state energy using the given Hamiltonian.

        Args:
            hamiltonian: MolecularHamiltonian to use

        Returns:
            MatrixExpression for ⟨Ψ|H|Ψ⟩

        Example:
            H = qm.HamiltonianBuilder.electronic().build()
            E = water.energy(H)
            E_val = E.numerical(r=0.96, theta=1.82)
        """
        psi = self.slater_determinant()
        return hamiltonian.element(psi, psi, molecule=self)

    # =========================================================================
    # RENDERING
    # =========================================================================

    def render(self, style: str = 'ball-stick'):
        """
        Render molecule using views.molecule().

        Args:
            style: Visualization style
        """
        if self.is_symbolic:
            raise ValueError("Cannot render molecule with symbolic geometry")

        mol_data = []
        for atom_obj, pos in zip(self._atoms, self._positions):
            mol_data.append((atom_obj.symbol, pos[0], pos[1], pos[2]))

        views.molecule(mol_data, style=style)

    def to_xyz(self) -> str:
        """
        Convert to XYZ format string.

        Returns:
            XYZ format string
        """
        if self.is_symbolic:
            raise ValueError("Cannot convert symbolic geometry to XYZ")

        lines = [str(len(self._atoms)), ""]
        for atom_obj, pos in zip(self._atoms, self._positions):
            lines.append(f"{atom_obj.symbol:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        formula = self._molecular_formula()
        return f"Molecule({formula}, n_atoms={len(self._atoms)}, n_electrons={self.n_electrons})"

    def _molecular_formula(self) -> str:
        """Generate molecular formula like H2O."""
        from collections import Counter
        counts = Counter(a.symbol for a in self._atoms)
        formula = ""
        # Standard order: C, H, then alphabetical
        for element in ['C', 'H']:
            if element in counts:
                n = counts.pop(element)
                formula += element + (str(n) if n > 1 else "")
        for element in sorted(counts.keys()):
            n = counts[element]
            formula += element + (str(n) if n > 1 else "")
        return formula


def molecule(atoms_with_positions: List[Tuple]) -> Molecule:
    """
    Create a Molecule from atoms with positions.

    Args:
        atoms_with_positions: List of (Atom or element, x, y, z) tuples

    Returns:
        Molecule object

    Example:
        water = qm.molecule([
            ('O', 0.0, 0.0, 0.0),
            ('H', 0.96, 0.0, 0.0),
            ('H', -0.24, 0.93, 0.0),
        ])
    """
    return Molecule(atoms_with_positions)


# =============================================================================
