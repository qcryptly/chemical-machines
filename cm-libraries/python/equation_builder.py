"""
Symbolic Equation Builder v2

Extended system with:
- More combination types (TRIPLES, QUADRUPLES, etc.)
- Constraint handling (nearest-neighbor, cutoff radius, etc.)
- Proper quantum operator algebra
- Graph-based topology for lattices/molecules
- Automatic symmetry detection
- Code generation for numerical backends
"""

from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any, Optional, Set, Tuple, Union
from itertools import combinations, product, permutations
from enum import Enum, auto
from abc import ABC, abstractmethod
from functools import reduce
import sympy as sp
from sympy import (Symbol, Function, Sum, Indexed, IndexedBase, symbols, latex, 
                   sqrt, Abs, I, pi, exp, sin, cos, KroneckerDelta, Piecewise,
                   factorial, binomial, Rational, Matrix, eye, zeros)
from sympy.physics.quantum import Operator, Commutator, AntiCommutator, Dagger
from sympy.physics.quantum.spin import JzKet, JzBra, Jplus, Jminus, Jz, J2
import numpy as np


# ============================================================
# COMBINATION TYPES
# ============================================================

class CombinationType(Enum):
    """How to iterate over particle sets"""
    SINGLE = auto()          # Each particle: ∑_i
    PAIRS = auto()           # Unique pairs: ∑_{i<j}
    ORDERED_PAIRS = auto()   # All ordered pairs: ∑_{i≠j}
    TRIPLES = auto()         # Unique triples: ∑_{i<j<k}
    ORDERED_TRIPLES = auto() # All ordered triples: ∑_{i≠j≠k}
    QUADRUPLES = auto()      # Unique quadruples: ∑_{i<j<k<l}
    ALL_PAIRS = auto()       # Including self: ∑_{i,j}
    CROSS = auto()           # Between two sets: ∑_{i∈A, j∈B}
    CROSS_TRIPLES = auto()   # Three different sets: ∑_{i∈A, j∈B, k∈C}
    NEIGHBORS = auto()       # Connected pairs in topology
    NEXT_NEIGHBORS = auto()  # Next-nearest neighbors
    CUSTOM = auto()          # User-defined iterator


# ============================================================
# CONSTRAINTS
# ============================================================

@dataclass
class Constraint(ABC):
    """Base class for constraints on interactions"""
    name: str
    
    @abstractmethod
    def applies(self, indices: Tuple, positions: Dict = None) -> bool:
        """Check if this constraint allows the given index combination"""
        pass
    
    @abstractmethod
    def to_symbolic(self, indices: Tuple) -> sp.Expr:
        """Return symbolic factor (0 or 1) or smooth cutoff"""
        pass


@dataclass
class NearestNeighborConstraint(Constraint):
    """Only allow interactions between nearest neighbors"""
    topology: 'Topology'
    
    def applies(self, indices: Tuple, positions: Dict = None) -> bool:
        i, j = indices[:2]
        return self.topology.are_neighbors(i, j)
    
    def to_symbolic(self, indices: Tuple) -> sp.Expr:
        # Returns delta function based on adjacency
        i, j = indices[:2]
        return self.topology.adjacency_symbol(i, j)


@dataclass  
class DistanceCutoffConstraint(Constraint):
    """Only allow interactions within cutoff radius"""
    cutoff: Union[float, Symbol]
    smooth: bool = False  # Use smooth cutoff function?
    
    def applies(self, indices: Tuple, positions: Dict = None) -> bool:
        # For concrete evaluation only
        return True
    
    def to_symbolic(self, indices: Tuple) -> sp.Expr:
        r_ij = Symbol(f'r_{indices[0]}{indices[1]}', positive=True)
        if self.smooth:
            # Smooth cutoff: f(r) = 1/2 * (1 + cos(π*r/r_c)) for r < r_c
            return Piecewise(
                (Rational(1,2) * (1 + cos(pi * r_ij / self.cutoff)), r_ij < self.cutoff),
                (0, True)
            )
        else:
            return Piecewise((1, r_ij < self.cutoff), (0, True))


@dataclass
class ExcludeIndicesConstraint(Constraint):
    """Exclude specific index combinations"""
    excluded: Set[Tuple]
    
    def applies(self, indices: Tuple, positions: Dict = None) -> bool:
        return indices not in self.excluded
    
    def to_symbolic(self, indices: Tuple) -> sp.Expr:
        if indices in self.excluded:
            return sp.Integer(0)
        return sp.Integer(1)


@dataclass
class SymmetryConstraint(Constraint):
    """Enforce symmetry (e.g., only i < j to avoid double counting)"""
    
    def applies(self, indices: Tuple, positions: Dict = None) -> bool:
        return all(indices[k] < indices[k+1] for k in range(len(indices)-1))
    
    def to_symbolic(self, indices: Tuple) -> sp.Expr:
        # Product of Heaviside functions
        factors = []
        for k in range(len(indices)-1):
            factors.append(Piecewise((1, indices[k] < indices[k+1]), (0, True)))
        return reduce(lambda a, b: a * b, factors, sp.Integer(1))


# ============================================================
# TOPOLOGY
# ============================================================

class Topology:
    """
    Defines connectivity/geometry of a system.
    Used for lattices, molecular graphs, etc.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.adjacency: Dict[int, Set[int]] = {}
        self.positions: Dict[int, Tuple] = {}
        self.next_nearest: Dict[int, Set[int]] = {}
        
    def add_site(self, index: int, position: Tuple = None):
        """Add a site to the topology"""
        if index not in self.adjacency:
            self.adjacency[index] = set()
        if position is not None:
            self.positions[index] = position
            
    def add_bond(self, i: int, j: int, bidirectional: bool = True):
        """Connect two sites"""
        self.add_site(i)
        self.add_site(j)
        self.adjacency[i].add(j)
        if bidirectional:
            self.adjacency[j].add(i)
            
    def are_neighbors(self, i: int, j: int) -> bool:
        return j in self.adjacency.get(i, set())
    
    def get_neighbors(self, i: int) -> Set[int]:
        return self.adjacency.get(i, set())
    
    def get_all_pairs(self) -> List[Tuple[int, int]]:
        """Get all bonded pairs (each pair once)"""
        pairs = set()
        for i, neighbors in self.adjacency.items():
            for j in neighbors:
                pair = (min(i,j), max(i,j))
                pairs.add(pair)
        return sorted(pairs)
    
    def adjacency_symbol(self, i, j) -> sp.Expr:
        """Symbolic adjacency matrix element"""
        A = IndexedBase('A')  # Adjacency matrix
        return A[i, j]
    
    def compute_next_nearest(self):
        """Compute next-nearest neighbors (2 bonds away)"""
        for i in self.adjacency:
            self.next_nearest[i] = set()
            for j in self.adjacency[i]:  # First neighbors
                for k in self.adjacency[j]:  # Their neighbors
                    if k != i and k not in self.adjacency[i]:
                        self.next_nearest[i].add(k)
    
    @classmethod
    def chain(cls, n: int, periodic: bool = False) -> 'Topology':
        """Create a 1D chain"""
        top = cls(f"Chain_{n}")
        for i in range(n):
            top.add_site(i, position=(i,))
            if i > 0:
                top.add_bond(i-1, i)
        if periodic and n > 2:
            top.add_bond(n-1, 0)
        top.compute_next_nearest()
        return top
    
    @classmethod
    def square_lattice(cls, nx: int, ny: int, periodic: bool = False) -> 'Topology':
        """Create a 2D square lattice"""
        top = cls(f"Square_{nx}x{ny}")
        
        def idx(x, y):
            return y * nx + x
        
        for y in range(ny):
            for x in range(nx):
                i = idx(x, y)
                top.add_site(i, position=(x, y))
                
                # Right neighbor
                if x < nx - 1:
                    top.add_bond(i, idx(x+1, y))
                elif periodic:
                    top.add_bond(i, idx(0, y))
                    
                # Up neighbor
                if y < ny - 1:
                    top.add_bond(i, idx(x, y+1))
                elif periodic:
                    top.add_bond(i, idx(x, 0))
        
        top.compute_next_nearest()
        return top
    
    @classmethod
    def triangular_lattice(cls, nx: int, ny: int) -> 'Topology':
        """Create a 2D triangular lattice"""
        top = cls(f"Triangular_{nx}x{ny}")
        
        def idx(x, y):
            return y * nx + x
        
        for y in range(ny):
            for x in range(nx):
                i = idx(x, y)
                # Offset for triangular geometry
                x_pos = x + 0.5 * (y % 2)
                y_pos = y * sqrt(3) / 2
                top.add_site(i, position=(float(x_pos), float(y_pos)))
                
                # Right
                if x < nx - 1:
                    top.add_bond(i, idx(x+1, y))
                # Up-left, Up-right
                if y < ny - 1:
                    if y % 2 == 0:
                        if x > 0:
                            top.add_bond(i, idx(x-1, y+1))
                        top.add_bond(i, idx(x, y+1))
                    else:
                        top.add_bond(i, idx(x, y+1))
                        if x < nx - 1:
                            top.add_bond(i, idx(x+1, y+1))
        
        top.compute_next_nearest()
        return top


# ============================================================
# QUANTUM OPERATORS
# ============================================================

class QuantumOperators:
    """
    Factory for creating quantum mechanical operators.
    Uses SymPy's quantum module where possible.
    """
    
    @staticmethod
    def creation(site: int, spin: str = None) -> Operator:
        """Fermionic/Bosonic creation operator"""
        if spin:
            return Operator(f'c†_{site},{spin}')
        return Operator(f'c†_{site}')
    
    @staticmethod
    def annihilation(site: int, spin: str = None) -> Operator:
        """Fermionic/Bosonic annihilation operator"""
        if spin:
            return Operator(f'c_{site},{spin}')
        return Operator(f'c_{site}')
    
    @staticmethod
    def number(site: int, spin: str = None) -> Operator:
        """Number operator n = c†c"""
        if spin:
            return Operator(f'n_{site},{spin}')
        return Operator(f'n_{site}')
    
    @staticmethod
    def spin_z(site: int) -> Operator:
        """S^z spin operator"""
        return Operator(f'S^z_{site}')
    
    @staticmethod
    def spin_plus(site: int) -> Operator:
        """S^+ raising operator"""
        return Operator(f'S^+_{site}')
    
    @staticmethod
    def spin_minus(site: int) -> Operator:
        """S^- lowering operator"""
        return Operator(f'S^-_{site}')
    
    @staticmethod
    def spin_x(site: int) -> Operator:
        """S^x = (S^+ + S^-)/2"""
        return Operator(f'S^x_{site}')
    
    @staticmethod
    def spin_y(site: int) -> Operator:
        """S^y = (S^+ - S^-)/(2i)"""
        return Operator(f'S^y_{site}')
    
    @staticmethod
    def spin_dot(i: int, j: int) -> sp.Expr:
        """S_i · S_j = S^x_i S^x_j + S^y_i S^y_j + S^z_i S^z_j
                     = (1/2)(S^+_i S^-_j + S^-_i S^+_j) + S^z_i S^z_j
        """
        Sp_i, Sm_i = QuantumOperators.spin_plus(i), QuantumOperators.spin_minus(i)
        Sp_j, Sm_j = QuantumOperators.spin_plus(j), QuantumOperators.spin_minus(j)
        Sz_i, Sz_j = QuantumOperators.spin_z(i), QuantumOperators.spin_z(j)
        
        return Rational(1,2) * (Sp_i * Sm_j + Sm_i * Sp_j) + Sz_i * Sz_j
    
    @staticmethod
    def hopping(i: int, j: int, spin: str = None) -> sp.Expr:
        """Hopping term: c†_i c_j + c†_j c_i"""
        ci_dag = QuantumOperators.creation(i, spin)
        cj = QuantumOperators.annihilation(j, spin)
        cj_dag = QuantumOperators.creation(j, spin)
        ci = QuantumOperators.annihilation(i, spin)
        return ci_dag * cj + cj_dag * ci
    
    @staticmethod
    def hubbard_U(site: int) -> sp.Expr:
        """On-site Hubbard interaction: n_↑ n_↓"""
        n_up = QuantumOperators.number(site, '↑')
        n_down = QuantumOperators.number(site, '↓')
        return n_up * n_down
    
    @staticmethod
    def position(particle: int, component: str = None) -> Operator:
        """Position operator"""
        if component:
            return Operator(f'r_{particle}^{component}')
        return Operator(f'r_{particle}')
    
    @staticmethod
    def momentum(particle: int, component: str = None) -> Operator:
        """Momentum operator"""
        if component:
            return Operator(f'p_{particle}^{component}')
        return Operator(f'p_{particle}')
    
    @staticmethod
    def kinetic_energy(particle: int, mass: Symbol) -> sp.Expr:
        """T = p²/2m (using symbolic placeholder for Laplacian)"""
        hbar = Symbol('hbar', positive=True)
        nabla_sq = Function('∇²')
        r = QuantumOperators.position(particle)
        return -hbar**2 / (2 * mass) * nabla_sq(r)
    
    @staticmethod
    def coulomb(i: int, j: int, q_i: sp.Expr, q_j: sp.Expr) -> sp.Expr:
        """Coulomb interaction between charges"""
        r_ij = Symbol(f'|r_{i}-r_{j}|', positive=True)
        e = Symbol('e', positive=True)
        k = Symbol('k_e', positive=True)  # Coulomb constant
        return k * q_i * q_j / r_ij


# ============================================================
# PARTICLE SETS
# ============================================================

@dataclass
class ParticleSet:
    """A collection of identical particles"""
    name: str
    symbol: str
    count: Union[int, Symbol]
    statistics: str = 'distinguishable'  # 'fermion', 'boson', 'distinguishable'
    spin: Optional[Rational] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.count, int):
            self._count_val = self.count
            self.count = Symbol(f'N_{self.symbol}', integer=True, positive=True)
        else:
            self._count_val = None
            
        # Create indexed bases for this particle type
        self.r = IndexedBase(f'r_{self.symbol}')  # position
        self.p = IndexedBase(f'p_{self.symbol}')  # momentum
        self.s = IndexedBase(f'S_{self.symbol}')  # spin
        self.index = Symbol(f'i_{self.symbol}', integer=True, positive=True)
        
    def indices(self, n: int = 1) -> List[Symbol]:
        """Generate n index symbols for this particle set"""
        return [Symbol(f'{self.symbol}_{k}', integer=True, positive=True) for k in range(1, n+1)]
        
    def __repr__(self):
        return f"ParticleSet({self.name}, n={self.count}, {self.statistics})"


# ============================================================
# INTERACTION TERMS
# ============================================================

@dataclass 
class InteractionTerm:
    """A small equation that acts on particles"""
    name: str
    combination_type: CombinationType
    particle_sets: List[str]
    expression: Callable
    prefactor: Any = 1
    constraints: List[Constraint] = field(default_factory=list)
    description: str = ""
    topology: Optional[Topology] = None
    custom_iterator: Optional[Callable] = None  # For CUSTOM combination type
    
    def __repr__(self):
        return f"InteractionTerm({self.name}, {self.combination_type.name})"


# ============================================================
# MAIN BUILDER
# ============================================================

class EquationBuilder:
    """
    Main class for building equations from declarative definitions.
    """
    
    def __init__(self, name: str = "H"):
        self.name = name
        self.particle_sets: Dict[str, ParticleSet] = {}
        self.interactions: List[InteractionTerm] = []
        self.constants: Dict[str, Symbol] = {}
        self.topology: Optional[Topology] = None
        self.built_expression = None
        self._ops = QuantumOperators()
        
    def set_topology(self, topology: Topology):
        """Set the system topology (lattice structure, molecular graph, etc.)"""
        self.topology = topology
        return self
        
    def add_constant(self, name: str, symbol: Optional[Symbol] = None, **assumptions) -> Symbol:
        """Define a constant"""
        if symbol is None:
            symbol = Symbol(name, **assumptions)
        self.constants[name] = symbol
        return symbol
        
    def add_particles(self, name: str, symbol: str, count: Union[int, Symbol], 
                     statistics: str = 'distinguishable',
                     spin: Rational = None,
                     **properties) -> ParticleSet:
        """Add a set of particles"""
        ps = ParticleSet(name, symbol, count, statistics, spin, properties)
        self.particle_sets[name] = ps
        return ps
    
    def add_interaction(self, 
                       name: str,
                       combination_type: CombinationType,
                       particle_sets: List[str],
                       expression: Callable,
                       prefactor = 1,
                       constraints: List[Constraint] = None,
                       description: str = "",
                       custom_iterator: Callable = None) -> InteractionTerm:
        """Add an interaction term"""
        term = InteractionTerm(
            name=name,
            combination_type=combination_type,
            particle_sets=particle_sets,
            expression=expression,
            prefactor=prefactor,
            constraints=constraints or [],
            description=description,
            topology=self.topology,
            custom_iterator=custom_iterator
        )
        self.interactions.append(term)
        return term
    
    # ----- Builder methods for common terms -----
    
    def add_kinetic(self, particle_set: str, mass: Symbol = None, name: str = None):
        """Add kinetic energy term for a particle set"""
        ps = self.particle_sets[particle_set]
        if mass is None:
            mass = ps.properties.get('mass', Symbol(f'm_{ps.symbol}', positive=True))
        if name is None:
            name = f'T_{ps.symbol}'
            
        self.add_interaction(
            name, CombinationType.SINGLE, [particle_set],
            lambda p, i, m=mass: self._ops.kinetic_energy(i, m),
            description=f"Kinetic energy of {particle_set}"
        )
        return self
    
    def add_coulomb_repulsion(self, particle_set: str, charge: sp.Expr = None, name: str = None):
        """Add Coulomb repulsion between same-type particles"""
        ps = self.particle_sets[particle_set]
        if charge is None:
            charge = ps.properties.get('charge', Symbol(f'q_{ps.symbol}'))
        if name is None:
            name = f'V_{ps.symbol}{ps.symbol}'
            
        self.add_interaction(
            name, CombinationType.PAIRS, [particle_set],
            lambda p, i, p2, j, q=charge: self._ops.coulomb(i, j, q, q),
            description=f"{particle_set}-{particle_set} Coulomb repulsion"
        )
        return self
    
    def add_coulomb_attraction(self, set1: str, set2: str, 
                               charge1: sp.Expr = None, charge2: sp.Expr = None,
                               name: str = None):
        """Add Coulomb attraction between different particle types"""
        ps1 = self.particle_sets[set1]
        ps2 = self.particle_sets[set2]
        
        if charge1 is None:
            charge1 = ps1.properties.get('charge', Symbol(f'q_{ps1.symbol}'))
        if charge2 is None:
            charge2 = ps2.properties.get('charge', Symbol(f'q_{ps2.symbol}'))
        if name is None:
            name = f'V_{ps1.symbol}{ps2.symbol}'
            
        self.add_interaction(
            name, CombinationType.CROSS, [set1, set2],
            lambda p1, i, p2, j, q1=charge1, q2=charge2: self._ops.coulomb(i, j, q1, q2),
            description=f"{set1}-{set2} Coulomb interaction"
        )
        return self
    
    def add_exchange(self, particle_set: str, J: Symbol = None, name: str = "exchange"):
        """Add Heisenberg exchange interaction (requires topology)"""
        if self.topology is None:
            raise ValueError("Exchange interaction requires a topology. Call set_topology() first.")
        
        if J is None:
            J = Symbol('J', real=True)
            self.constants['J'] = J
            
        self.add_interaction(
            name, CombinationType.NEIGHBORS, [particle_set],
            lambda p, i, p2, j, J=J: -J * self._ops.spin_dot(i, j),
            description="Heisenberg exchange"
        )
        return self
    
    def add_hubbard(self, particle_set: str, U: Symbol = None, t: Symbol = None):
        """Add Hubbard model terms (requires topology for hopping)"""
        if self.topology is None:
            raise ValueError("Hubbard model requires a topology.")
        
        if U is None:
            U = Symbol('U', real=True)
            self.constants['U'] = U
        if t is None:
            t = Symbol('t', real=True, positive=True)
            self.constants['t'] = t
            
        # Hopping
        self.add_interaction(
            "hopping", CombinationType.NEIGHBORS, [particle_set],
            lambda p, i, p2, j, t=t: -t * (self._ops.hopping(i, j, '↑') + self._ops.hopping(i, j, '↓')),
            description="Hopping term"
        )
        
        # On-site interaction
        self.add_interaction(
            "onsite_U", CombinationType.SINGLE, [particle_set],
            lambda p, i, U=U: U * self._ops.hubbard_U(i),
            description="On-site Hubbard U"
        )
        return self
    
    def add_three_body(self, particle_set: str, expression: Callable, 
                      name: str = "three_body", prefactor = 1):
        """Add a three-body interaction"""
        self.add_interaction(
            name, CombinationType.TRIPLES, [particle_set],
            expression, prefactor,
            description="Three-body interaction"
        )
        return self

    # ----- Building logic -----
    
    def _generate_indices(self, term: InteractionTerm) -> List[Tuple]:
        """Generate all index combinations based on combination type"""
        combo_type = term.combination_type
        sets = [self.particle_sets[name] for name in term.particle_sets]
        
        if combo_type == CombinationType.SINGLE:
            ps = sets[0]
            n = ps._count_val
            if n is not None:
                return [(i,) for i in range(1, n + 1)]
            else:
                i = Symbol('i', integer=True, positive=True)
                return [(i,)]  # Symbolic
                
        elif combo_type == CombinationType.PAIRS:
            ps = sets[0]
            n = ps._count_val
            if n is not None:
                return list(combinations(range(1, n + 1), 2))
            else:
                i, j = symbols('i j', integer=True, positive=True)
                return [(i, j)]  # Symbolic with i < j constraint
                
        elif combo_type == CombinationType.TRIPLES:
            ps = sets[0]
            n = ps._count_val
            if n is not None:
                return list(combinations(range(1, n + 1), 3))
            else:
                i, j, k = symbols('i j k', integer=True, positive=True)
                return [(i, j, k)]
                
        elif combo_type == CombinationType.QUADRUPLES:
            ps = sets[0]
            n = ps._count_val
            if n is not None:
                return list(combinations(range(1, n + 1), 4))
            else:
                i, j, k, l = symbols('i j k l', integer=True, positive=True)
                return [(i, j, k, l)]
                
        elif combo_type == CombinationType.CROSS:
            ps1, ps2 = sets[0], sets[1]
            n1, n2 = ps1._count_val, ps2._count_val
            if n1 is not None and n2 is not None:
                return list(product(range(1, n1 + 1), range(1, n2 + 1)))
            else:
                i, j = symbols('i j', integer=True, positive=True)
                return [(i, j)]
                
        elif combo_type == CombinationType.NEIGHBORS:
            if self.topology is None:
                raise ValueError("NEIGHBORS requires topology")
            return self.topology.get_all_pairs()
            
        elif combo_type == CombinationType.NEXT_NEIGHBORS:
            if self.topology is None:
                raise ValueError("NEXT_NEIGHBORS requires topology")
            pairs = set()
            for i, nn in self.topology.next_nearest.items():
                for j in nn:
                    pairs.add((min(i,j), max(i,j)))
            return sorted(pairs)
            
        elif combo_type == CombinationType.CUSTOM:
            if term.custom_iterator is None:
                raise ValueError("CUSTOM requires custom_iterator")
            return list(term.custom_iterator(sets, self.topology))
            
        else:
            raise NotImplementedError(f"Combination type {combo_type} not implemented")
    
    def _build_term(self, term: InteractionTerm) -> Tuple[str, sp.Expr, List]:
        """Build a single interaction term"""
        indices_list = self._generate_indices(term)
        sets = [self.particle_sets[name] for name in term.particle_sets]
        
        total_expr = 0
        expanded_terms = []
        
        for indices in indices_list:
            # Check constraints
            skip = False
            constraint_factor = 1
            for constraint in term.constraints:
                if not constraint.applies(indices):
                    skip = True
                    break
                constraint_factor *= constraint.to_symbolic(indices)
            
            if skip:
                continue
            
            # Build expression for this index combination
            if len(sets) == 1:
                if len(indices) == 1:
                    expr = term.expression(sets[0], indices[0])
                elif len(indices) == 2:
                    expr = term.expression(sets[0], indices[0], sets[0], indices[1])
                elif len(indices) == 3:
                    expr = term.expression(sets[0], indices[0], sets[0], indices[1], 
                                          sets[0], indices[2])
                elif len(indices) == 4:
                    expr = term.expression(sets[0], indices[0], sets[0], indices[1],
                                          sets[0], indices[2], sets[0], indices[3])
            elif len(sets) == 2:
                expr = term.expression(sets[0], indices[0], sets[1], indices[1])
            else:
                # General case
                args = []
                for s, idx in zip(sets, indices):
                    args.extend([s, idx])
                expr = term.expression(*args)
            
            expr = term.prefactor * constraint_factor * expr
            total_expr += expr
            expanded_terms.append((indices, expr))
        
        return term.name, total_expr, expanded_terms
    
    def build(self, expand: bool = True) -> Dict[str, Any]:
        """Build the complete equation"""
        result = {'terms': {}, 'expanded': {}, 'total': 0}
        
        for term in self.interactions:
            name, expr, expanded = self._build_term(term)
            result['terms'][name] = expr
            result['expanded'][name] = expanded
            result['total'] += expr
        
        self.built_expression = result
        return result
    
    def to_latex(self, term_name: str = 'total', wrap: bool = True) -> str:
        """Get LaTeX representation"""
        if self.built_expression is None:
            self.build()
        
        expr = self.built_expression['terms'].get(term_name, 
               self.built_expression.get('total'))
        
        tex = latex(expr)
        if wrap:
            tex = f"${tex}$"
        return tex
    
    def summary(self) -> str:
        """Human-readable summary"""
        lines = [f"╔══ {self.name} ══╗\n"]
        
        lines.append("┌─ Particle Sets:")
        for name, ps in self.particle_sets.items():
            props = ", ".join(f"{k}={v}" for k, v in ps.properties.items())
            extra = f", {ps.statistics}" if ps.statistics != 'distinguishable' else ""
            lines.append(f"│  • {name} ({ps.symbol}): N={ps.count}{extra}")
            if props:
                lines.append(f"│    └─ {props}")
        
        if self.topology:
            lines.append(f"│\n├─ Topology: {self.topology.name}")
            lines.append(f"│  └─ {len(self.topology.adjacency)} sites, "
                        f"{len(self.topology.get_all_pairs())} bonds")
        
        lines.append("│\n└─ Interaction Terms:")
        for term in self.interactions:
            sets = " × ".join(term.particle_sets)
            lines.append(f"   • {term.name}: {term.combination_type.name} over {sets}")
            if term.description:
                lines.append(f"     └─ {term.description}")
            if term.constraints:
                for c in term.constraints:
                    lines.append(f"     └─ constraint: {c.name}")
        
        return "\n".join(lines)
    
    def count_terms(self) -> Dict[str, int]:
        """Count the number of concrete terms in each part"""
        if self.built_expression is None:
            self.build()
        return {name: len(expanded) 
                for name, expanded in self.built_expression['expanded'].items()}


# ============================================================
# CODE GENERATION
# ============================================================

class CodeGenerator:
    """Generate numerical code from built equations"""
    
    @staticmethod
    def to_numpy(builder: EquationBuilder, output_type: str = 'matrix') -> str:
        """Generate NumPy code for evaluating the Hamiltonian"""
        # This is a simplified example - full implementation would be more complex
        lines = [
            "import numpy as np",
            "from scipy import sparse",
            "",
            f"def build_{builder.name.lower().replace(' ', '_')}(params):",
            "    '''Auto-generated Hamiltonian builder'''",
        ]
        
        if builder.topology:
            n_sites = len(builder.topology.adjacency)
            lines.append(f"    n_sites = {n_sites}")
            lines.append(f"    # Adjacency: {builder.topology.get_all_pairs()}")
        
        lines.append("    H = 0")
        lines.append("    # TODO: Implement matrix construction")
        lines.append("    return H")
        
        return "\n".join(lines)


# ============================================================
# PRE-BUILT TEMPLATES
# ============================================================

def molecular_hamiltonian(n_electrons: int = None, n_nuclei: int = None, 
                          nuclear_charges: List[int] = None):
    """Full molecular Hamiltonian with all terms"""
    builder = EquationBuilder("Molecular Hamiltonian")
    
    # Constants
    hbar = builder.add_constant('hbar', positive=True)
    e = builder.add_constant('e', positive=True)
    m_e = builder.add_constant('m_e', positive=True)
    k_e = builder.add_constant('k_e', positive=True)
    
    # Particles
    N_e = n_electrons if n_electrons else Symbol('N_e', integer=True, positive=True)
    N_N = n_nuclei if n_nuclei else Symbol('N_N', integer=True, positive=True)
    
    electrons = builder.add_particles("electrons", "e", N_e, 
                                       statistics='fermion', spin=Rational(1,2),
                                       mass=m_e, charge=-e)
    
    M = IndexedBase('M')
    Z = IndexedBase('Z')
    nuclei = builder.add_particles("nuclei", "N", N_N,
                                    mass=M, charge=Z)
    
    # Add all terms using builder methods
    builder.add_kinetic("electrons")
    builder.add_kinetic("nuclei")
    builder.add_coulomb_repulsion("electrons")
    builder.add_coulomb_repulsion("nuclei")
    builder.add_coulomb_attraction("electrons", "nuclei")
    
    return builder


def hubbard_model(nx: int, ny: int = 1, periodic: bool = False):
    """Hubbard model on a lattice"""
    builder = EquationBuilder("Hubbard Model")
    
    # Set up lattice
    if ny == 1:
        topology = Topology.chain(nx, periodic)
    else:
        topology = Topology.square_lattice(nx, ny, periodic)
    builder.set_topology(topology)
    
    # Particles (sites with electrons)
    n_sites = len(topology.adjacency)
    sites = builder.add_particles("sites", "i", n_sites, statistics='fermion')
    
    # Add Hubbard terms
    builder.add_hubbard("sites")
    
    return builder


def heisenberg_model(nx: int, ny: int = 1, periodic: bool = False,
                     include_nnn: bool = False, anisotropy: float = 1.0):
    """
    Heisenberg spin model.
    
    H = -J Σ_{<i,j>} S_i · S_j - J' Σ_{<<i,j>>} S_i · S_j - h Σ_i S^z_i
    
    Args:
        include_nnn: Include next-nearest neighbor interactions
        anisotropy: XXZ anisotropy parameter (Δ)
    """
    builder = EquationBuilder("Heisenberg Model")
    
    # Topology
    if ny == 1:
        topology = Topology.chain(nx, periodic)
    else:
        topology = Topology.square_lattice(nx, ny, periodic)
    builder.set_topology(topology)
    
    # Constants
    J = builder.add_constant('J', real=True)
    h = builder.add_constant('h', real=True)
    
    # Spins
    n_sites = len(topology.adjacency)
    spins = builder.add_particles("spins", "s", n_sites, spin=Rational(1,2))
    
    # Exchange (nearest neighbor)
    builder.add_exchange("spins", J)
    
    # Next-nearest neighbor
    if include_nnn:
        J_prime = builder.add_constant("J'", real=True)
        builder.add_interaction(
            "nnn_exchange", CombinationType.NEXT_NEIGHBORS, ["spins"],
            lambda p, i, p2, j: -J_prime * QuantumOperators.spin_dot(i, j),
            description="Next-nearest neighbor exchange"
        )
    
    # Zeeman term
    builder.add_interaction(
        "zeeman", CombinationType.SINGLE, ["spins"],
        lambda p, i: -h * QuantumOperators.spin_z(i),
        description="External field"
    )
    
    return builder


def three_body_potential(n_particles: int):
    """System with explicit three-body interactions (e.g., Axilrod-Teller)"""
    builder = EquationBuilder("Three-Body System")
    
    # Constants
    C = builder.add_constant('C', real=True)  # 3-body coefficient
    
    particles = builder.add_particles("particles", "p", n_particles)
    
    # Two-body (Lennard-Jones style, simplified)
    epsilon = builder.add_constant('ε', positive=True)
    sigma = builder.add_constant('σ', positive=True)
    
    builder.add_interaction(
        "LJ", CombinationType.PAIRS, ["particles"],
        lambda p, i, p2, j: epsilon * (sigma / Symbol(f'r_{i}{j}', positive=True))**6,
        description="Lennard-Jones attraction"
    )
    
    # Three-body (Axilrod-Teller type)
    def axilrod_teller(p, i, p2, j, p3, k):
        """Three-body dispersion: proportional to 1 + 3*cos(θ_i)*cos(θ_j)*cos(θ_k)"""
        # Simplified - actual implementation would use angle functions
        r_ij = Symbol(f'r_{i}{j}', positive=True)
        r_jk = Symbol(f'r_{j}{k}', positive=True)
        r_ik = Symbol(f'r_{i}{k}', positive=True)
        cos_i = Symbol(f'cos_θ_{i}')
        cos_j = Symbol(f'cos_θ_{j}')
        cos_k = Symbol(f'cos_θ_{k}')
        
        return C * (1 + 3 * cos_i * cos_j * cos_k) / (r_ij * r_jk * r_ik)**3
    
    builder.add_three_body("particles", axilrod_teller, "Axilrod_Teller")
    
    return builder


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DEMO 1: Heisenberg Model on 4-site chain")
    print("=" * 70)
    
    H = heisenberg_model(nx=4, periodic=False)
    print(H.summary())
    result = H.build()
    
    print("\nExpanded terms:")
    for name, terms in result['expanded'].items():
        print(f"\n{name}:")
        for indices, expr in terms:
            print(f"  {indices}: {expr}")
    
    print(f"\nTotal term count: {H.count_terms()}")
    
    print("\n" + "=" * 70)
    print("DEMO 2: Hubbard Model on 2x2 square lattice")
    print("=" * 70)
    
    H_hub = hubbard_model(nx=2, ny=2, periodic=False)
    print(H_hub.summary())
    result = H_hub.build()
    
    print("\nExpanded terms:")
    for name, terms in result['expanded'].items():
        print(f"\n{name}:")
        for indices, expr in terms[:5]:  # First 5 only
            print(f"  {indices}: {expr}")
        if len(terms) > 5:
            print(f"  ... ({len(terms)} total)")
    
    print("\n" + "=" * 70)
    print("DEMO 3: Three-body potential (4 particles)")  
    print("=" * 70)
    
    H_3body = three_body_potential(n_particles=4)
    print(H_3body.summary())
    result = H_3body.build()
    
    print(f"\nTerm counts: {H_3body.count_terms()}")
    print("\nThree-body terms (i<j<k):")
    for indices, expr in result['expanded']['Axilrod_Teller']:
        print(f"  {indices}")
    
    print("\n" + "=" * 70)
    print("DEMO 4: Molecular Hamiltonian (H2)")
    print("=" * 70)
    
    H_mol = molecular_hamiltonian(n_electrons=2, n_nuclei=2)
    print(H_mol.summary())
    result = H_mol.build()
    
    print(f"\nTerm counts: {H_mol.count_terms()}")
    
    print("\n" + "=" * 70)
    print("DEMO 5: Custom constraint - cutoff radius")
    print("=" * 70)
    
    builder = EquationBuilder("Cutoff LJ")
    r_cut = Symbol('r_c', positive=True)
    
    particles = builder.add_particles("atoms", "a", 5)
    
    cutoff_constraint = DistanceCutoffConstraint("cutoff", r_cut, smooth=True)
    
    builder.add_interaction(
        "LJ_cutoff", CombinationType.PAIRS, ["atoms"],
        lambda p, i, p2, j: 1 / Symbol(f'r_{i}{j}', positive=True)**6,
        constraints=[cutoff_constraint],
        description="LJ with smooth cutoff"
    )
    
    print(builder.summary())
    result = builder.build()
    print(f"\nPair interactions: {len(result['expanded']['LJ_cutoff'])}")
    print("Each includes smooth cutoff function")