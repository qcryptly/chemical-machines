"""
Chemical Machines QM Atoms Module

Electron configurations and Atom class for atomic calculations.
"""

from typing import Optional, List, Union, Tuple, Dict, Set, Any

from ..symbols import Expr, Var
from .data import ATOMIC_NUMBERS, ELEMENT_SYMBOLS, AUFBAU_ORDER, L_LABELS, NOBLE_GAS_CONFIGS, _max_electrons_in_subshell
from .coordinates import Coordinate3D, coord3d, spherical_coord
from .spinorbitals import SpinOrbital, SlaterDeterminant, spin_orbital, slater
from .relativistic import DiracSpinor, DiracDeterminant

__all__ = [
    'ElectronConfiguration',
    'Atom',
    'atom',
    'atoms',
    'ground_state',
    'config_from_string',
]


# =============================================================================
# ELECTRON CONFIGURATION
# =============================================================================

class ElectronConfiguration:
    """
    Represents an electron configuration for an atom.

    Supports automatic aufbau filling (ground state) and manual orbital
    specification for excited states or custom configurations.

    The configuration stores orbitals as (n, l, m, spin) tuples following
    Hund's rules for ground state filling.

    Example:
        # Ground state carbon (automatic aufbau)
        config = ElectronConfiguration.aufbau(6)
        print(config.label)  # "1s² 2s² 2p²"

        # Manual specification
        config = ElectronConfiguration.manual([
            (1, 0, 0, 1), (1, 0, 0, -1),  # 1s²
            (2, 0, 0, 1), (2, 0, 0, -1),  # 2s²
            (2, 1, 0, 1), (2, 1, 1, 1),   # 2p² (Hund's rule)
        ])

        # From string notation
        config = ElectronConfiguration.from_string("1s2 2s2 2p2")
        config = ElectronConfiguration.from_string("[He] 2s2 2p2")
    """

    def __init__(self, orbitals: List[Tuple[int, int, int, int]]):
        """
        Create an electron configuration from explicit orbital list.

        Args:
            orbitals: List of (n, l, m, spin) tuples where:
                - n: Principal quantum number (>= 1)
                - l: Angular momentum (0 to n-1)
                - m: Magnetic quantum number (-l to l)
                - spin: +1 (up/alpha) or -1 (down/beta)

        Raises:
            ValueError: If quantum numbers are invalid or Pauli exclusion violated
        """
        self._orbitals = list(orbitals)
        self._validate()

    def _validate(self):
        """Validate quantum numbers and Pauli exclusion."""
        seen = set()
        for n, l, m, spin in self._orbitals:
            if n < 1:
                raise ValueError(f"Principal quantum number n must be >= 1, got {n}")
            if l < 0 or l >= n:
                raise ValueError(f"Angular momentum l must be 0 <= l < n, got l={l}, n={n}")
            if abs(m) > l:
                raise ValueError(f"|m| must be <= l, got m={m}, l={l}")
            if spin not in (1, -1):
                raise ValueError(f"Spin must be +1 or -1, got {spin}")

            key = (n, l, m, spin)
            if key in seen:
                raise ValueError(f"Pauli exclusion violation: duplicate orbital {key}")
            seen.add(key)

    @classmethod
    def aufbau(cls, n_electrons: int) -> "ElectronConfiguration":
        """
        Create ground state configuration using aufbau principle.

        Fills orbitals following:
        1. Aufbau order (increasing n+l, then n)
        2. Hund's rules (maximize spin within subshell)
        3. Pauli exclusion principle

        Args:
            n_electrons: Number of electrons to place

        Returns:
            ElectronConfiguration with ground state filling

        Example:
            config = ElectronConfiguration.aufbau(6)  # Carbon
            print(config.label)  # "1s² 2s² 2p²"
        """
        if n_electrons < 0:
            raise ValueError(f"Number of electrons must be >= 0, got {n_electrons}")

        orbitals = []
        remaining = n_electrons

        for n, l in AUFBAU_ORDER:
            if remaining <= 0:
                break

            max_in_subshell = _max_electrons_in_subshell(l)
            electrons_to_add = min(remaining, max_in_subshell)

            # Fill following Hund's rules: spin up first, then spin down
            subshell_orbitals = cls._fill_subshell(n, l, electrons_to_add)
            orbitals.extend(subshell_orbitals)
            remaining -= electrons_to_add

        return cls(orbitals)

    @staticmethod
    def _fill_subshell(n: int, l: int, n_electrons: int) -> List[Tuple[int, int, int, int]]:
        """
        Fill a subshell following Hund's rules.

        First fills all m values with spin up, then spin down.
        This maximizes total spin (Hund's first rule).

        Args:
            n: Principal quantum number
            l: Angular momentum quantum number
            n_electrons: Number of electrons to place

        Returns:
            List of (n, l, m, spin) tuples
        """
        orbitals = []
        m_values = list(range(-l, l + 1))  # -l, -l+1, ..., l-1, l

        # First pass: fill with spin up (+1)
        for m in m_values:
            if len(orbitals) >= n_electrons:
                break
            orbitals.append((n, l, m, 1))

        # Second pass: fill with spin down (-1)
        for m in m_values:
            if len(orbitals) >= n_electrons:
                break
            orbitals.append((n, l, m, -1))

        return orbitals

    @classmethod
    def manual(cls, orbitals: List[Tuple[int, int, int, int]]) -> "ElectronConfiguration":
        """
        Create configuration from explicit orbital list.

        Args:
            orbitals: List of (n, l, m, spin) tuples

        Returns:
            ElectronConfiguration with specified orbitals

        Example:
            # Excited state with electron promoted
            config = ElectronConfiguration.manual([
                (1, 0, 0, 1), (1, 0, 0, -1),  # 1s²
                (2, 0, 0, 1),                  # 2s¹
                (2, 1, -1, 1), (2, 1, 0, 1), (2, 1, 1, 1),  # 2p³
            ])
        """
        return cls(orbitals)

    @classmethod
    def from_string(cls, notation: str) -> "ElectronConfiguration":
        """
        Parse electron configuration from string notation.

        Supports:
        - Standard notation: "1s2 2s2 2p6"
        - Noble gas core: "[He] 2s2 2p2", "[Ar] 3d10 4s2"
        - Superscript numbers: "1s² 2s² 2p²"

        Args:
            notation: Configuration string

        Returns:
            ElectronConfiguration object

        Example:
            config = ElectronConfiguration.from_string("1s2 2s2 2p2")
            config = ElectronConfiguration.from_string("[Ne] 3s2 3p4")
        """
        # Normalize: replace superscript digits with regular digits
        superscripts = {'⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
                        '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'}
        for sup, reg in superscripts.items():
            notation = notation.replace(sup, reg)

        orbitals = []

        # Check for noble gas core notation [He], [Ne], etc.
        noble_match = re.match(r'\[(\w+)\]\s*', notation)
        if noble_match:
            noble_gas = noble_match.group(1)
            if noble_gas not in NOBLE_GAS_CONFIGS:
                raise ValueError(f"Unknown noble gas: {noble_gas}")
            # Get the noble gas configuration
            n_core = NOBLE_GAS_CONFIGS[noble_gas]
            core_config = cls.aufbau(n_core)
            orbitals.extend(core_config._orbitals)
            notation = notation[noble_match.end():]

        # Parse remaining subshells: "1s2", "2p6", "3d10", etc.
        pattern = r'(\d+)([spdfghik])(\d+)'
        for match in re.finditer(pattern, notation.lower()):
            n = int(match.group(1))
            l_char = match.group(2)
            count = int(match.group(3))

            l = L_LABELS.index(l_char)
            max_e = _max_electrons_in_subshell(l)
            if count > max_e:
                raise ValueError(f"Too many electrons in {n}{l_char}: {count} > {max_e}")

            subshell = cls._fill_subshell(n, l, count)
            orbitals.extend(subshell)

        if not orbitals:
            raise ValueError(f"Could not parse configuration: {notation}")

        return cls(orbitals)

    @property
    def n_electrons(self) -> int:
        """Total number of electrons."""
        return len(self._orbitals)

    @property
    def orbitals(self) -> List[Tuple[int, int, int, int]]:
        """List of (n, l, m, spin) tuples."""
        return list(self._orbitals)

    @property
    def subshell_occupancy(self) -> Dict[str, int]:
        """
        Return occupancy by subshell.

        Returns:
            Dict mapping subshell labels to electron counts
            e.g., {'1s': 2, '2s': 2, '2p': 2}
        """
        occupancy: Dict[str, int] = {}
        for n, l, m, spin in self._orbitals:
            key = f"{n}{L_LABELS[l]}"
            occupancy[key] = occupancy.get(key, 0) + 1
        return occupancy

    @property
    def shell_occupancy(self) -> Dict[int, int]:
        """
        Return occupancy by shell (principal quantum number).

        Returns:
            Dict mapping n to electron counts
            e.g., {1: 2, 2: 4}
        """
        occupancy: Dict[int, int] = {}
        for n, l, m, spin in self._orbitals:
            occupancy[n] = occupancy.get(n, 0) + 1
        return occupancy

    @property
    def label(self) -> str:
        """
        Human-readable label using superscript notation.

        Returns:
            String like "1s² 2s² 2p²"
        """
        superscripts = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
                        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'}
        parts = []
        for subshell, count in self.subshell_occupancy.items():
            count_str = ''.join(superscripts[d] for d in str(count))
            parts.append(f"{subshell}{count_str}")
        return ' '.join(parts)

    @property
    def latex_label(self) -> str:
        """
        LaTeX formatted label.

        Returns:
            String like "1s^{2} 2s^{2} 2p^{2}"
        """
        parts = []
        for subshell, count in self.subshell_occupancy.items():
            parts.append(f"{subshell}^{{{count}}}")
        return ' '.join(parts)

    def to_latex(self) -> str:
        """LaTeX representation."""
        return self.latex_label

    def excite(self, from_orbital: Tuple[int, int, int, int],
               to_orbital: Tuple[int, int, int, int]) -> "ElectronConfiguration":
        """
        Create excited state by moving an electron.

        Args:
            from_orbital: (n, l, m, spin) of orbital to vacate
            to_orbital: (n, l, m, spin) of orbital to occupy

        Returns:
            New ElectronConfiguration with the excitation

        Raises:
            ValueError: If from_orbital not occupied or to_orbital already occupied
        """
        if from_orbital not in self._orbitals:
            raise ValueError(f"Cannot excite from {from_orbital}: not occupied")
        if to_orbital in self._orbitals:
            raise ValueError(f"Cannot excite to {to_orbital}: already occupied")

        new_orbitals = [o for o in self._orbitals if o != from_orbital]
        new_orbitals.append(to_orbital)
        return ElectronConfiguration(new_orbitals)

    def ionize(self, n: int = 1) -> "ElectronConfiguration":
        """
        Remove electrons (create cation).

        Removes electrons from highest energy orbitals first.

        Args:
            n: Number of electrons to remove (default 1)

        Returns:
            New ElectronConfiguration with fewer electrons

        Raises:
            ValueError: If n > number of electrons
        """
        if n > len(self._orbitals):
            raise ValueError(f"Cannot remove {n} electrons from {len(self._orbitals)}")
        if n <= 0:
            return ElectronConfiguration(self._orbitals)

        # Sort orbitals by energy (reverse aufbau order) and remove from highest
        def orbital_energy_key(orb):
            n, l, m, spin = orb
            # Find position in aufbau order
            try:
                idx = AUFBAU_ORDER.index((n, l))
            except ValueError:
                idx = 100  # High value for orbitals not in standard order
            return (idx, m, -spin)  # Higher index = higher energy

        sorted_orbitals = sorted(self._orbitals, key=orbital_energy_key, reverse=True)
        return ElectronConfiguration(sorted_orbitals[n:])

    def add_electron(self, orbital: Optional[Tuple[int, int, int, int]] = None) -> "ElectronConfiguration":
        """
        Add an electron (create anion).

        Args:
            orbital: Specific (n, l, m, spin) to add, or None for next aufbau position

        Returns:
            New ElectronConfiguration with additional electron
        """
        if orbital is not None:
            if orbital in self._orbitals:
                raise ValueError(f"Orbital {orbital} already occupied")
            return ElectronConfiguration(self._orbitals + [orbital])

        # Find next available orbital following aufbau
        next_config = ElectronConfiguration.aufbau(len(self._orbitals) + 1)
        # The last orbital in aufbau is the one to add
        new_orbital = next_config._orbitals[-1]
        if new_orbital in self._orbitals:
            raise ValueError("Cannot determine next orbital to add")
        return ElectronConfiguration(self._orbitals + [new_orbital])

    def to_spinors(self) -> List[Tuple[int, int, float]]:
        """
        Convert to relativistic spinor quantum numbers.

        Maps (n, l, m, spin) to (n, kappa, mj) for DiracSpinor.

        The mapping considers that for a given mj, we need to choose
        the appropriate j (and hence kappa) such that |mj| <= j.

        Returns:
            List of (n, kappa, mj) tuples
        """
        spinors = []
        for n, l, m, spin in self._orbitals:
            # mj = m + spin/2
            mj = m + spin * 0.5

            if l == 0:
                # s orbital: only j = 1/2, kappa = -1
                kappa = -1
            else:
                # For l > 0, we have two choices: j = l + 1/2 or j = l - 1/2
                # j = l + 1/2 -> kappa = -(l+1), valid mj range: [-l-0.5, l+0.5]
                # j = l - 1/2 -> kappa = l, valid mj range: [-l+0.5, l-0.5]

                j_high = l + 0.5  # j = l + 1/2
                j_low = l - 0.5   # j = l - 1/2

                # Check if mj is valid for j_low (smaller j)
                if j_low >= 0.5 and abs(mj) <= j_low:
                    # Can use either j, prefer j_low for spin-down, j_high for spin-up
                    if spin == -1:
                        kappa = l  # j = l - 1/2
                    else:
                        kappa = -(l + 1)  # j = l + 1/2
                elif abs(mj) <= j_high:
                    # Must use j_high
                    kappa = -(l + 1)  # j = l + 1/2
                else:
                    # This shouldn't happen with valid quantum numbers
                    # Fall back to j_high
                    kappa = -(l + 1)

            spinors.append((n, kappa, mj))
        return spinors

    @property
    def spinors(self) -> List[Tuple[int, int, float]]:
        """List of (n, kappa, mj) tuples for relativistic calculations."""
        return self.to_spinors()

    def __repr__(self) -> str:
        return f"ElectronConfiguration({self.label})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, ElectronConfiguration):
            return False
        return set(self._orbitals) == set(other._orbitals)

    def __hash__(self) -> int:
        return hash(frozenset(self._orbitals))


# =============================================================================
# ATOM CLASS
# =============================================================================

class Atom(Expr):
    """
    Represents an atom with nuclear charge, position, and electron configuration.

    The Atom class integrates with the existing qm module to provide:
    - Automatic or manual electron configurations
    - Non-relativistic (SpinOrbital) or relativistic (DiracSpinor) orbitals
    - Coordinate system support (numeric or symbolic Coordinate3D)
    - Visualization via views.molecule()
    - Slater/Dirac determinant generation for QM calculations
    - ComputeGraph support for symbolic computation and PyTorch compilation

    Example:
        # Simple carbon atom at origin
        C = qm.atom('C')
        psi = C.slater_determinant()

        # Carbon with explicit position
        C = qm.atom('C', position=(1.0, 0.0, 0.0))

        # Manual electron configuration
        C_excited = qm.atom('C', configuration=ElectronConfiguration.from_string("1s2 2s1 2p3"))

        # Relativistic uranium
        U = qm.atom('U', relativistic=True)
        psi = U.dirac_determinant()

        # ComputeGraph for Hamiltonian evaluation
        H = qm.atom('H')
        cg = H.to_compute_graph(r=1.0)
        energy = cg.evaluate()
    """

    def __init__(
        self,
        element: Union[str, int],
        position: Union[Tuple[float, float, float], Coordinate3D] = (0.0, 0.0, 0.0),
        configuration: Optional[ElectronConfiguration] = None,
        relativistic: bool = False,
        charge: int = 0,
        vec3: Optional[Coordinate3D] = None,
    ):
        """
        Create an Atom.

        Args:
            element: Element symbol ('C', 'Fe') or atomic number (6, 26)
            position: Atom center position as (x, y, z) tuple or Coordinate3D
            configuration: Electron configuration (None for automatic aufbau)
            relativistic: Use DiracSpinor (True) or SpinOrbital (False)
            charge: Ionic charge (positive for cations, negative for anions)
            vec3: Shared Coordinate3D for orbital wavefunctions (default: spherical)

        Raises:
            ValueError: If element is unknown or charge exceeds atomic number
        """
        super().__init__()

        # Resolve element symbol and atomic number
        if isinstance(element, str):
            if element not in ATOMIC_NUMBERS:
                raise ValueError(f"Unknown element symbol: {element}")
            self._symbol = element
            self._Z = ATOMIC_NUMBERS[element]
        else:
            if element not in ELEMENT_SYMBOLS:
                raise ValueError(f"Unknown atomic number: {element}")
            self._Z = element
            self._symbol = ELEMENT_SYMBOLS[element]

        # Validate charge
        if charge > self._Z:
            raise ValueError(f"Charge {charge} exceeds atomic number {self._Z}")

        self._charge = charge
        self._relativistic = relativistic

        # Set position
        if isinstance(position, Coordinate3D):
            self._position = position
        else:
            # Store as tuple for numeric positions
            self._position = tuple(position)

        # Set orbital coordinate system
        if vec3 is not None:
            self._vec3 = vec3
        else:
            # Default to spherical coordinates for orbital wavefunctions
            self._vec3 = spherical_coord()

        # Set electron configuration
        n_electrons = self._Z - charge
        if configuration is not None:
            if configuration.n_electrons != n_electrons:
                raise ValueError(
                    f"Configuration has {configuration.n_electrons} electrons, "
                    f"but atom {self._symbol} with charge {charge} should have {n_electrons}"
                )
            self._configuration = configuration
        else:
            # Automatic aufbau filling
            self._configuration = ElectronConfiguration.aufbau(n_electrons)

        # Cache for orbitals/spinors
        self._orbitals_cache: Optional[List[SpinOrbital]] = None
        self._spinors_cache: Optional[List[DiracSpinor]] = None

    # =========================================================================
    # ELEMENT PROPERTIES
    # =========================================================================

    @property
    def symbol(self) -> str:
        """Element symbol (e.g., 'C', 'Fe')."""
        return self._symbol

    @property
    def Z(self) -> int:
        """Atomic number (nuclear charge)."""
        return self._Z

    @property
    def name(self) -> str:
        """Element name (e.g., 'Carbon', 'Iron')."""
        # Get from views.ELEMENT_DATA if available
        if self._symbol in views.ELEMENT_DATA:
            return views.ELEMENT_DATA[self._symbol].get('name', self._symbol)
        return self._symbol

    @property
    def n_electrons(self) -> int:
        """Number of electrons (Z - charge)."""
        return self._configuration.n_electrons

    @property
    def charge(self) -> int:
        """Net ionic charge."""
        return self._charge

    @property
    def is_ion(self) -> bool:
        """True if charged (cation or anion)."""
        return self._charge != 0

    # =========================================================================
    # POSITION PROPERTIES
    # =========================================================================

    @property
    def position(self) -> Union[Tuple[float, float, float], Coordinate3D]:
        """Atom center position."""
        return self._position

    @property
    def x(self) -> Union[float, Expr]:
        """X coordinate of atom center."""
        if isinstance(self._position, Coordinate3D):
            if self._position.coord_type == CoordinateType.CARTESIAN:
                return self._position.x
            raise ValueError("Position is in spherical coordinates, use .position instead")
        return self._position[0]

    @property
    def y(self) -> Union[float, Expr]:
        """Y coordinate of atom center."""
        if isinstance(self._position, Coordinate3D):
            if self._position.coord_type == CoordinateType.CARTESIAN:
                return self._position.y
            raise ValueError("Position is in spherical coordinates, use .position instead")
        return self._position[1]

    @property
    def z(self) -> Union[float, Expr]:
        """Z coordinate of atom center."""
        if isinstance(self._position, Coordinate3D):
            if self._position.coord_type == CoordinateType.CARTESIAN:
                return self._position.z
            raise ValueError("Position is in spherical coordinates, use .position instead")
        return self._position[2]

    @property
    def is_symbolic(self) -> bool:
        """True if position uses symbolic Coordinate3D."""
        return isinstance(self._position, Coordinate3D)

    @property
    def numeric_position(self) -> Tuple[float, float, float]:
        """
        Get numeric (x, y, z) position.

        Raises:
            ValueError: If position is symbolic and cannot be evaluated
        """
        if isinstance(self._position, tuple):
            return self._position
        raise ValueError("Position is symbolic Coordinate3D, cannot get numeric position")

    # =========================================================================
    # CONFIGURATION PROPERTIES
    # =========================================================================

    @property
    def configuration(self) -> ElectronConfiguration:
        """Electron configuration object."""
        return self._configuration

    @property
    def relativistic(self) -> bool:
        """True if using relativistic (DiracSpinor) orbitals."""
        return self._relativistic

    @property
    def vec3(self) -> Coordinate3D:
        """Shared coordinate for orbital wavefunctions."""
        return self._vec3

    @property
    def orbitals(self) -> List[SpinOrbital]:
        """
        List of SpinOrbital objects for this atom.

        Creates SpinOrbital objects from the electron configuration.
        """
        if self._orbitals_cache is None:
            self._orbitals_cache = [
                SpinOrbital(self._vec3, n=n, l=l, m=m, spin=spin)
                for n, l, m, spin in self._configuration.orbitals
            ]
        return self._orbitals_cache

    @property
    def spinors(self) -> List[DiracSpinor]:
        """
        List of DiracSpinor objects for this atom.

        Creates DiracSpinor objects from the electron configuration.
        """
        if self._spinors_cache is None:
            self._spinors_cache = [
                DiracSpinor(n=n, kappa=kappa, mj=mj)
                for n, kappa, mj in self._configuration.spinors
            ]
        return self._spinors_cache

    # =========================================================================
    # SLATER DETERMINANT GENERATION
    # =========================================================================

    def slater_determinant(self) -> SlaterDeterminant:
        """
        Create a SlaterDeterminant for this atom's electrons.

        Returns:
            SlaterDeterminant containing all occupied orbitals

        Example:
            C = qm.atom('C')
            psi = C.slater_determinant()
            H = qm.hamiltonian()
            energy = psi @ H @ psi
        """
        return SlaterDeterminant(self.orbitals)

    def dirac_determinant(self) -> DiracDeterminant:
        """
        Create a DiracDeterminant for this atom's electrons.

        Returns:
            DiracDeterminant containing all occupied spinors

        Example:
            U = qm.atom('U', relativistic=True)
            psi = U.dirac_determinant()
            H_DC = qm.dirac_hamiltonian()
            energy = psi @ H_DC @ psi
        """
        return DiracDeterminant(self.spinors)

    def determinant(self) -> Union[SlaterDeterminant, DiracDeterminant]:
        """
        Create appropriate determinant based on relativistic setting.

        Returns:
            SlaterDeterminant if relativistic=False
            DiracDeterminant if relativistic=True
        """
        if self._relativistic:
            return self.dirac_determinant()
        return self.slater_determinant()

    # =========================================================================
    # EXPR INTERFACE (required for symbolic computation)
    # =========================================================================

    def to_sympy(self):
        """
        Convert to SymPy representation.

        Returns the nuclear charge Z as a SymPy number.
        """
        sp = _get_sympy()
        return sp.Integer(self._Z)

    def to_latex(self) -> str:
        """
        LaTeX representation of the atom.

        Returns:
            LaTeX string like "\\mathrm{C}" or "\\mathrm{Fe}^{2+}"
        """
        if self._charge == 0:
            return f"\\mathrm{{{self._symbol}}}"
        elif self._charge > 0:
            return f"\\mathrm{{{self._symbol}}}^{{{self._charge}+}}"
        else:
            return f"\\mathrm{{{self._symbol}}}^{{{abs(self._charge)}-}}"

    def _get_free_variables(self) -> Set[Var]:
        """
        Return free variables from position/coordinates.

        Returns:
            Set of Var objects from symbolic position or vec3
        """
        free_vars: Set[Var] = set()
        if isinstance(self._position, Coordinate3D):
            free_vars.update(self._position._get_free_variables())
        free_vars.update(self._vec3._get_free_variables())
        return free_vars

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    def to_molecule_tuple(self) -> Tuple[str, float, float, float]:
        """
        Convert to views.molecule() format.

        Returns:
            (symbol, x, y, z) tuple for use with views.molecule()

        Raises:
            ValueError: If position is symbolic and not evaluable
        """
        if isinstance(self._position, tuple):
            return (self._symbol, self._position[0], self._position[1], self._position[2])
        raise ValueError("Cannot convert symbolic position to molecule tuple")

    def render(self, style: str = 'ball-stick'):
        """
        Render atom using views.molecule().

        Args:
            style: Visualization style ('ball-stick', 'spacefill')
        """
        try:
            mol_tuple = self.to_molecule_tuple()
            views.molecule([mol_tuple], style=style)
        except ValueError:
            # Symbolic position - render as text instead
            html = f'<div class="cm-math cm-math-center">\\[ {self.to_latex()} \\]</div>'
            views.html(html)

    def render_configuration(self, style: str = 'text'):
        """
        Render electron configuration.

        Args:
            style: 'text' for simple label, 'latex' for LaTeX
        """
        if style == 'latex':
            latex = f"{self.to_latex()}: {self._configuration.latex_label}"
            html = f'<div class="cm-math cm-math-center">\\[ {latex} \\]</div>'
        else:
            html = f'<div>{self._symbol}: {self._configuration.label}</div>'
        views.html(html)

    # =========================================================================
    # MODIFICATION (return new atom)
    # =========================================================================

    def with_position(self, position: Union[Tuple[float, float, float], Coordinate3D]) -> "Atom":
        """Create new atom with different position."""
        return Atom(
            element=self._symbol,
            position=position,
            configuration=self._configuration,
            relativistic=self._relativistic,
            charge=self._charge,
            vec3=self._vec3,
        )

    def with_configuration(self, configuration: ElectronConfiguration) -> "Atom":
        """Create new atom with different configuration."""
        return Atom(
            element=self._symbol,
            position=self._position,
            configuration=configuration,
            relativistic=self._relativistic,
            charge=self._Z - configuration.n_electrons,
            vec3=self._vec3,
        )

    def ionize(self, charge_delta: int = 1) -> "Atom":
        """
        Create ion by changing charge.

        Args:
            charge_delta: Change in charge (+1 removes electron, -1 adds electron)

        Returns:
            New Atom with updated charge and configuration
        """
        new_charge = self._charge + charge_delta
        new_config = self._configuration.ionize(charge_delta) if charge_delta > 0 else \
                     self._configuration.add_electron() if charge_delta == -1 else \
                     self._configuration

        return Atom(
            element=self._symbol,
            position=self._position,
            configuration=new_config,
            relativistic=self._relativistic,
            charge=new_charge,
            vec3=self._vec3,
        )

    def excite(self, from_orbital: Tuple[int, int, int, int],
               to_orbital: Tuple[int, int, int, int]) -> "Atom":
        """
        Create excited state atom.

        Args:
            from_orbital: (n, l, m, spin) of orbital to vacate
            to_orbital: (n, l, m, spin) of orbital to occupy

        Returns:
            New Atom with excited configuration
        """
        new_config = self._configuration.excite(from_orbital, to_orbital)
        return Atom(
            element=self._symbol,
            position=self._position,
            configuration=new_config,
            relativistic=self._relativistic,
            charge=self._charge,
            vec3=self._vec3,
        )

    def to_relativistic(self) -> "Atom":
        """Convert to relativistic representation."""
        return Atom(
            element=self._symbol,
            position=self._position,
            configuration=self._configuration,
            relativistic=True,
            charge=self._charge,
            vec3=self._vec3,
        )

    def to_nonrelativistic(self) -> "Atom":
        """Convert to non-relativistic representation."""
        return Atom(
            element=self._symbol,
            position=self._position,
            configuration=self._configuration,
            relativistic=False,
            charge=self._charge,
            vec3=self._vec3,
        )

    # =========================================================================
    # LABELING
    # =========================================================================

    @property
    def label(self) -> str:
        """Human-readable label."""
        if self._charge == 0:
            return f"{self._symbol} ({self.n_electrons} electrons)"
        elif self._charge > 0:
            return f"{self._symbol}{self._charge}+ ({self.n_electrons} electrons)"
        else:
            return f"{self._symbol}{abs(self._charge)}- ({self.n_electrons} electrons)"

    @property
    def ion_label(self) -> str:
        """Ion notation like 'Fe2+' or 'Cl-'."""
        if self._charge == 0:
            return self._symbol
        elif self._charge > 0:
            return f"{self._symbol}{self._charge}+" if self._charge > 1 else f"{self._symbol}+"
        else:
            return f"{self._symbol}{abs(self._charge)}-" if self._charge < -1 else f"{self._symbol}-"

    @property
    def ket_label(self) -> str:
        """Ket notation label for use in determinants."""
        return self._symbol

    # =========================================================================
    # ENERGY CALCULATIONS
    # =========================================================================

    def energy(self, hamiltonian: "MolecularHamiltonian") -> "MatrixExpression":
        """
        Calculate ground state energy using the given Hamiltonian.

        Creates a single-atom molecule wrapper to properly handle Z values.

        Args:
            hamiltonian: MolecularHamiltonian to use

        Returns:
            MatrixExpression for ⟨Ψ|H|Ψ⟩

        Example:
            H = qm.HamiltonianBuilder.electronic().build()
            C = qm.atom('C')
            E = C.energy(H)
            E_val = E.numerical()
        """
        # Create a single-atom molecule wrapper
        # Lazy import to avoid circular dependency
        from .molecules import Molecule
        mol = Molecule([(self._symbol, *self._position)])
        return mol.energy(hamiltonian)

    # =========================================================================
    # SPECIAL METHODS
    # =========================================================================

    def __repr__(self) -> str:
        pos_str = f"({self._position[0]}, {self._position[1]}, {self._position[2]})" \
                  if isinstance(self._position, tuple) else "symbolic"
        return f"Atom({self._symbol}, Z={self._Z}, pos={pos_str}, charge={self._charge})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Atom):
            return False
        return (self._symbol == other._symbol and
                self._charge == other._charge and
                self._configuration == other._configuration)

    def __hash__(self) -> int:
        return hash((self._symbol, self._charge, hash(self._configuration)))


# =============================================================================
# ATOM FACTORY FUNCTIONS
# =============================================================================

def atom(
    element: Union[str, int],
    position: Union[Tuple[float, float, float], Coordinate3D] = (0.0, 0.0, 0.0),
    configuration: Optional[Union[ElectronConfiguration, List[Tuple], str]] = None,
    relativistic: bool = False,
    charge: int = 0,
) -> Atom:
    """
    Create an Atom.

    Args:
        element: Element symbol ('C') or atomic number (6)
        position: (x, y, z) tuple or Coordinate3D
        configuration: ElectronConfiguration, orbital list, or string notation
        relativistic: Use DiracSpinor orbitals
        charge: Ionic charge

    Returns:
        Atom object

    Example:
        C = qm.atom('C')
        C = qm.atom(6, position=(1.0, 0.0, 0.0))
        C = qm.atom('C', configuration="1s2 2s2 2p2")
        Fe2 = qm.atom('Fe', charge=2)
    """
    # Convert configuration if needed
    if isinstance(configuration, str):
        configuration = ElectronConfiguration.from_string(configuration)
    elif isinstance(configuration, list):
        configuration = ElectronConfiguration.manual(configuration)

    return Atom(
        element=element,
        position=position,
        configuration=configuration,
        relativistic=relativistic,
        charge=charge,
    )


def atoms(
    specs: List[Tuple],
    relativistic: bool = False,
) -> List[Atom]:
    """
    Create multiple atoms from tuple specifications.

    Args:
        specs: List of (element, x, y, z) or (element, x, y, z, charge) tuples
        relativistic: Use DiracSpinor orbitals for all atoms

    Returns:
        List of Atom objects

    Example:
        # Water molecule atoms
        water_atoms = qm.atoms([
            ('O', 0.0, 0.0, 0.0),
            ('H', 0.96, 0.0, 0.0),
            ('H', -0.24, 0.93, 0.0),
        ])

        # With charges
        ions = qm.atoms([
            ('Na', 0.0, 0.0, 0.0, 1),   # Na+
            ('Cl', 2.8, 0.0, 0.0, -1),  # Cl-
        ])
    """
    result = []
    for spec in specs:
        if len(spec) == 4:
            element, x, y, z = spec
            charge = 0
        elif len(spec) == 5:
            element, x, y, z, charge = spec
        else:
            raise ValueError(f"Invalid atom spec: {spec}. Expected (element, x, y, z) or (element, x, y, z, charge)")

        result.append(Atom(
            element=element,
            position=(x, y, z),
            relativistic=relativistic,
            charge=charge,
        ))
    return result


def ground_state(n_electrons: int) -> ElectronConfiguration:
    """
    Create ground state electron configuration.

    Uses aufbau principle, Hund's rules, Pauli exclusion.

    Args:
        n_electrons: Number of electrons

    Returns:
        ElectronConfiguration with ground state filling

    Example:
        config = qm.ground_state(6)  # Carbon ground state
        print(config.label)  # "1s² 2s² 2p²"
    """
    return ElectronConfiguration.aufbau(n_electrons)


def config_from_string(notation: str) -> ElectronConfiguration:
    """
    Parse electron configuration from string.

    Args:
        notation: String like "1s2 2s2 2p2" or "[He] 2s2 2p2"

    Returns:
        ElectronConfiguration object

    Example:
        config = qm.config_from_string("1s2 2s2 2p6 3s2 3p6 4s2 3d6")
        config = qm.config_from_string("[Ar] 4s2 3d6")  # Same as above
    """
    return ElectronConfiguration.from_string(notation)


# =============================================================================
