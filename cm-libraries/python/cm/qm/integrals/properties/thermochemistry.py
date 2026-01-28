"""
Thermochemistry Calculations

Computes thermochemical properties from vibrational frequencies
using standard statistical mechanics formulas.

Includes:
- Zero-point energy (ZPE)
- Thermal corrections to energy, enthalpy, and Gibbs free energy
- Entropy contributions (translational, rotational, vibrational, electronic)
- Heat capacity

Based on ideal gas, rigid rotor, harmonic oscillator approximations.

Reference: McQuarrie, "Statistical Mechanics" (2000)
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .frequencies import FrequencyResult, harmonic_frequencies


# Physical constants
KB = 3.166811563e-6      # Boltzmann constant in Hartree/K
H_PLANCK = 1.0           # Planck constant (atomic units, ℏ = 1)
C_LIGHT = 137.036        # Speed of light in atomic units
R_GAS = 3.166811563e-6   # Gas constant in Hartree/(mol·K)
ATM_TO_HARTREE = 3.3989e-14  # 1 atm in Hartree/Bohr³
BOHR_TO_M = 5.29177e-11  # Bohr to meters
AMU_TO_KG = 1.66054e-27  # amu to kg

# Conversion factors
HARTREE_TO_KCAL = 627.5095
HARTREE_TO_KJ = 2625.5
HARTREE_TO_WAVENUMBER = 219474.63


@dataclass
class ThermochemistryResult:
    """
    Result of thermochemistry calculation.

    All energies in Hartree unless otherwise noted.

    Attributes:
        temperature: Temperature in Kelvin
        pressure: Pressure in atm

        # Energies
        E_elec: Electronic energy
        ZPE: Zero-point vibrational energy
        E_trans: Translational thermal energy
        E_rot: Rotational thermal energy
        E_vib: Vibrational thermal energy (above ZPE)
        E_thermal: Total thermal correction (E_trans + E_rot + E_vib)

        # Thermodynamic functions
        H_corr: Enthalpy correction (E_thermal + RT)
        E_total: Total energy (E_elec + ZPE + E_thermal)
        H_total: Total enthalpy (E_elec + H_corr)

        # Entropy (in Hartree/K)
        S_trans: Translational entropy
        S_rot: Rotational entropy
        S_vib: Vibrational entropy
        S_elec: Electronic entropy
        S_total: Total entropy

        # Gibbs free energy
        G_corr: Gibbs free energy correction
        G_total: Total Gibbs free energy

        # Heat capacity (constant pressure, Hartree/K)
        Cv: Heat capacity at constant volume
        Cp: Heat capacity at constant pressure
    """
    temperature: float
    pressure: float

    # Energies
    E_elec: float
    ZPE: float
    E_trans: float
    E_rot: float
    E_vib: float
    E_thermal: float

    # Thermodynamic functions
    H_corr: float
    E_total: float
    H_total: float

    # Entropy
    S_trans: float
    S_rot: float
    S_vib: float
    S_elec: float
    S_total: float

    # Gibbs free energy
    G_corr: float
    G_total: float

    # Heat capacity
    Cv: float
    Cp: float

    def print_summary(self):
        """Print formatted thermochemistry summary."""
        print("\n" + "=" * 60)
        print("THERMOCHEMISTRY ANALYSIS")
        print("=" * 60)
        print(f"Temperature: {self.temperature:.2f} K")
        print(f"Pressure: {self.pressure:.4f} atm")
        print("-" * 60)

        print("\nEnergy Components (Hartree):")
        print(f"  Electronic energy:       {self.E_elec:15.6f}")
        print(f"  Zero-point correction:   {self.ZPE:15.6f}")
        print(f"  Thermal correction E:    {self.E_thermal:15.6f}")
        print(f"  Thermal correction H:    {self.H_corr:15.6f}")
        print(f"  Thermal correction G:    {self.G_corr:15.6f}")

        print("\nTotal Energies (Hartree):")
        print(f"  E (0 K):        {self.E_elec + self.ZPE:15.6f}")
        print(f"  E ({self.temperature:.0f} K):     {self.E_total:15.6f}")
        print(f"  H ({self.temperature:.0f} K):     {self.H_total:15.6f}")
        print(f"  G ({self.temperature:.0f} K):     {self.G_total:15.6f}")

        print("\nEntropy Components (cal/mol·K):")
        cal_per_hartree_K = HARTREE_TO_KCAL * 1000
        print(f"  Translational:  {self.S_trans * cal_per_hartree_K:10.3f}")
        print(f"  Rotational:     {self.S_rot * cal_per_hartree_K:10.3f}")
        print(f"  Vibrational:    {self.S_vib * cal_per_hartree_K:10.3f}")
        print(f"  Electronic:     {self.S_elec * cal_per_hartree_K:10.3f}")
        print(f"  Total:          {self.S_total * cal_per_hartree_K:10.3f}")

        print("\nHeat Capacity (cal/mol·K):")
        print(f"  Cv:  {self.Cv * cal_per_hartree_K:10.3f}")
        print(f"  Cp:  {self.Cp * cal_per_hartree_K:10.3f}")

        print("=" * 60)


def thermochemistry(freq_result: FrequencyResult = None,
                    atoms: List[Tuple[str, Tuple[float, float, float]]] = None,
                    E_elec: float = 0.0,
                    temperature: float = 298.15,
                    pressure: float = 1.0,
                    multiplicity: int = 1,
                    method: str = 'HF',
                    basis: str = 'STO-3G',
                    verbose: bool = True) -> ThermochemistryResult:
    """
    Compute thermochemical properties.

    Args:
        freq_result: Pre-computed frequency result (optional)
        atoms: Molecule geometry (required if no freq_result)
        E_elec: Electronic energy in Hartree
        temperature: Temperature in Kelvin
        pressure: Pressure in atmospheres
        multiplicity: Spin multiplicity (2S+1)
        method: Electronic structure method
        basis: Basis set name
        verbose: Print summary

    Returns:
        ThermochemistryResult with all thermochemical properties

    Example:
        thermo = thermochemistry(
            atoms=[('H', (0,0,0)), ('H', (0.74,0,0))],
            E_elec=-1.1167,
            temperature=298.15,
            method='HF'
        )
        thermo.print_summary()
    """
    # Compute frequencies if needed
    if freq_result is None:
        if atoms is None:
            raise ValueError("Either freq_result or atoms required")
        freq_result = harmonic_frequencies(atoms, method=method, basis=basis)

    atoms = freq_result.hessian_result.atoms
    masses = freq_result.hessian_result.masses
    n_atoms = len(atoms)

    # Get frequencies (in cm⁻¹)
    frequencies = freq_result.frequencies
    real_freqs = frequencies[frequencies > 0]

    # Zero-point energy
    ZPE = freq_result.zpe

    # Thermal energy contributions
    beta = 1.0 / (KB * temperature)

    # Translational contribution (3/2 kT)
    E_trans = 1.5 * KB * temperature

    # Rotational contribution
    # Linear: kT, Nonlinear: 3/2 kT
    is_linear = n_atoms <= 2 or _check_linear(atoms)
    E_rot = KB * temperature if is_linear else 1.5 * KB * temperature

    # Vibrational contribution
    # E_vib = Σ hν/(exp(hν/kT) - 1)
    E_vib = 0.0
    for freq in real_freqs:
        # Convert cm⁻¹ to Hartree
        nu = freq / HARTREE_TO_WAVENUMBER
        x = nu / (KB * temperature)
        if x < 100:  # Avoid overflow
            E_vib += nu / (np.exp(x) - 1)

    E_thermal = E_trans + E_rot + E_vib

    # Enthalpy correction (includes PV = RT term)
    H_corr = ZPE + E_thermal + KB * temperature

    # Total energies
    E_total = E_elec + ZPE + E_thermal
    H_total = E_elec + H_corr

    # Entropy contributions

    # Translational entropy (Sackur-Tetrode)
    total_mass = np.sum(masses)
    mass_kg = total_mass * AMU_TO_KG
    V = KB * temperature / (pressure * ATM_TO_HARTREE)

    # S_trans = R[ln(V/Λ³) + 5/2] where Λ is thermal wavelength
    # Λ = h/sqrt(2πmkT)
    h_SI = 6.626e-34
    kB_SI = 1.381e-23
    lambda_thermal = h_SI / np.sqrt(2 * np.pi * mass_kg * kB_SI * temperature)
    V_SI = V * BOHR_TO_M ** 3

    S_trans = R_GAS * (np.log(V_SI / lambda_thermal ** 3) + 2.5)

    # Rotational entropy
    # For nonlinear: S_rot = R[ln(sqrt(π I_A I_B I_C)/σ) + 3/2]
    # Simplified using moments of inertia
    I = _compute_moments_of_inertia(atoms, masses)
    sigma = 1  # Symmetry number (simplified)

    if is_linear:
        I_linear = max(I)
        if I_linear > 1e-10:
            S_rot = R_GAS * (np.log(8 * np.pi ** 2 * I_linear * KB * temperature / sigma) + 1)
        else:
            S_rot = 0.0
    else:
        I_prod = np.prod(I[I > 1e-10])
        if I_prod > 1e-30:
            S_rot = R_GAS * (np.log(np.sqrt(np.pi * I_prod) * (8 * np.pi ** 2 * KB * temperature) ** 1.5 / sigma) + 1.5)
        else:
            S_rot = 0.0

    # Vibrational entropy
    # S_vib = R Σ [x/(exp(x)-1) - ln(1-exp(-x))]
    S_vib = 0.0
    for freq in real_freqs:
        nu = freq / HARTREE_TO_WAVENUMBER
        x = nu / (KB * temperature)
        if x < 100:
            S_vib += R_GAS * (x / (np.exp(x) - 1) - np.log(1 - np.exp(-x)))

    # Electronic entropy
    # S_elec = R ln(multiplicity)
    S_elec = R_GAS * np.log(multiplicity)

    S_total = S_trans + S_rot + S_vib + S_elec

    # Gibbs free energy
    G_corr = H_corr - temperature * S_total
    G_total = E_elec + G_corr

    # Heat capacity
    # Cv = dE/dT
    Cv_trans = 1.5 * R_GAS
    Cv_rot = R_GAS if is_linear else 1.5 * R_GAS

    Cv_vib = 0.0
    for freq in real_freqs:
        nu = freq / HARTREE_TO_WAVENUMBER
        x = nu / (KB * temperature)
        if x < 100:
            ex = np.exp(x)
            Cv_vib += R_GAS * x ** 2 * ex / (ex - 1) ** 2

    Cv = Cv_trans + Cv_rot + Cv_vib
    Cp = Cv + R_GAS  # Cp = Cv + R for ideal gas

    result = ThermochemistryResult(
        temperature=temperature,
        pressure=pressure,
        E_elec=E_elec,
        ZPE=ZPE,
        E_trans=E_trans,
        E_rot=E_rot,
        E_vib=E_vib,
        E_thermal=E_thermal,
        H_corr=H_corr,
        E_total=E_total,
        H_total=H_total,
        S_trans=S_trans,
        S_rot=S_rot,
        S_vib=S_vib,
        S_elec=S_elec,
        S_total=S_total,
        G_corr=G_corr,
        G_total=G_total,
        Cv=Cv,
        Cp=Cp
    )

    if verbose:
        result.print_summary()

    return result


def _check_linear(atoms: List[Tuple[str, Tuple[float, float, float]]],
                  threshold: float = 1e-6) -> bool:
    """Check if molecule is linear."""
    if len(atoms) <= 2:
        return True

    coords = np.array([a[1] for a in atoms])
    v1 = coords[1] - coords[0]
    v1 = v1 / np.linalg.norm(v1)

    for i in range(2, len(atoms)):
        v2 = coords[i] - coords[0]
        if np.linalg.norm(v2) < threshold:
            continue
        v2 = v2 / np.linalg.norm(v2)
        cross = np.cross(v1, v2)
        if np.linalg.norm(cross) > threshold:
            return False

    return True


def _compute_moments_of_inertia(atoms: List[Tuple[str, Tuple[float, float, float]]],
                                masses: np.ndarray) -> np.ndarray:
    """Compute principal moments of inertia."""
    coords = np.array([a[1] for a in atoms])

    # Center of mass
    total_mass = np.sum(masses)
    com = np.sum(masses[:, np.newaxis] * coords, axis=0) / total_mass
    coords_com = coords - com

    # Inertia tensor
    I = np.zeros((3, 3))
    for i, (m, r) in enumerate(zip(masses, coords_com)):
        I[0, 0] += m * (r[1] ** 2 + r[2] ** 2)
        I[1, 1] += m * (r[0] ** 2 + r[2] ** 2)
        I[2, 2] += m * (r[0] ** 2 + r[1] ** 2)
        I[0, 1] -= m * r[0] * r[1]
        I[0, 2] -= m * r[0] * r[2]
        I[1, 2] -= m * r[1] * r[2]

    I[1, 0] = I[0, 1]
    I[2, 0] = I[0, 2]
    I[2, 1] = I[1, 2]

    # Principal moments
    eigenvalues, _ = np.linalg.eigh(I)

    return eigenvalues
