#!/usr/bin/env python3
"""Run the laptop COSEnu linear-stability analysis for one simulation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from itertools import permutations
from math import erf, pi, sqrt
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


PHYSICAL_OMEGA_KM_INV = 0.63
C_METERS_PER_MICROSECOND = 300.0
NUMERICAL_ZERO = 1.0e-12


def read_values(path: Path, separator: str) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or separator not in line:
            continue
        key, value = line.split(separator, 1)
        values[key.strip()] = value.split("#", 1)[0].strip().strip("\"'")
    return values


def as_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered not in {"true", "false"}:
        raise ValueError(f"Expected true or false, got {value!r}")
    return lowered == "true"


@dataclass(frozen=True)
class LSAConfig:
    nvz: int
    num: int
    vmin: float
    vmax: float
    kmin: float
    kmax: float
    alpha: float
    angular_profile: str
    neutrino_width: float
    antineutrino_width: float
    perturbation_size: float
    omega_over_mu: float
    effective_omega: float
    mu: float
    matter_potential: float
    matter_slope: float
    damping_rate: float
    antineutrino_collision_ratio: float
    dt: float
    simulation_nvz: int
    dz: float
    multienergy: bool
    energy_grid: str
    energies: np.ndarray
    signed_frequencies: np.ndarray
    energy_mode_weights: np.ndarray
    reference_energy: float


@dataclass(frozen=True)
class ComparisonResult:
    cosenu_gamma: float
    error_percent: float | None
    fit_interval: tuple[float, float]
    lsa_normalization: float


@dataclass(frozen=True)
class AngularBranchResult:
    lambda_plus: float
    lambda_minus: float
    lepton_asymmetry: float
    lepton_current: float
    no_peak_gamma: float
    no_peak_kprime: float
    io_peak_gamma: float
    io_peak_kprime: float


def load_config(
    run_dir: Path,
    nvz: int,
    num: int,
    kmin: float | None = None,
    kmax: float | None = None,
) -> LSAConfig:
    laptop = read_values(run_dir / "laptop.conf", "=")
    job = read_values(run_dir / "job.config", ":")
    if not job:
        raise FileNotFoundError(f"Missing {run_dir / 'job.config'}")

    def job_float(name: str) -> float:
        if name not in job:
            raise ValueError(f"job.config does not contain {name}")
        return float(job[name])

    vmin = job_float("v0")
    vmax = job_float("v1")
    zmin = job_float("z0")
    zmax = job_float("z1")
    nz = int(job["nz"])
    omega = job_float("omega")
    theta_radians = job_float("theta") * pi / 180.0

    vacuum_on = as_bool(laptop.get("VACUUM_OSCILLATIONS", "true"))
    collective_on = as_bool(laptop.get("COLLECTIVE_OSCILLATIONS", "true"))
    matter_on = as_bool(laptop.get("MATTER_OSCILLATIONS", "false"))
    ordering = job_float("pmo")
    effective_omega = ordering * omega * np.cos(2.0 * theta_radians) if vacuum_on else 0.0
    mu = job_float("mu") if collective_on else 0.0

    matter_offset = float(laptop.get("MATTER_OFFSET", "0"))
    matter_slope = float(laptop.get("MATTER_SLOPE", "0"))
    midpoint = 0.5 * (zmin + zmax)
    matter = matter_offset + matter_slope * (midpoint - zmin) if matter_on else 0.0
    dz = (zmax - zmin) / nz
    dt = abs(job_float("CFL") * dz / vmax)

    power_kmax = abs(job_float("k_max"))
    multienergy = job.get("multienergy", "0") == "1"
    if multienergy:
        energy_table_path = run_dir / "energy_grid.dat"
        if not energy_table_path.exists():
            raise FileNotFoundError(
                f"Multi-energy LSA requires {energy_table_path}"
            )
        energy_table = np.atleast_2d(np.loadtxt(energy_table_path, comments="#"))
        if energy_table.shape[1] < 4:
            raise ValueError("energy_grid.dat must contain at least four columns")
        energies = energy_table[:, 1]
        signed_frequencies = energy_table[:, 2]
        energy_mode_weights = energy_table[:, 3]
    else:
        energies = np.empty(0)
        signed_frequencies = np.empty(0)
        energy_mode_weights = np.empty(0)
    config = LSAConfig(
        nvz=nvz,
        num=num,
        vmin=vmin,
        vmax=vmax,
        kmin=float(laptop.get("LSA_K_MIN", "-1.5")) if kmin is None else kmin,
        kmax=float(laptop.get("LSA_K_MAX", "1.5")) if kmax is None else kmax,
        alpha=float(laptop.get("ALPHA", "0.85")),
        angular_profile=laptop.get("ANGULAR_PROFILE", "gaussian").lower(),
        neutrino_width=float(laptop.get("NEUTRINO_WIDTH", "0.6")),
        antineutrino_width=float(laptop.get("ANTINEUTRINO_WIDTH", "0.6")),
        perturbation_size=job_float("perturbation_size"),
        omega_over_mu=omega,
        effective_omega=float(effective_omega),
        mu=mu,
        matter_potential=matter,
        matter_slope=matter_slope if matter_on else 0.0,
        damping_rate=float(laptop.get("DAMPING_RATE", "0")),
        antineutrino_collision_ratio=float(
            laptop.get("ANTINEUTRINO_COLLISION_RATIO", "0")
        ),
        dt=dt,
        simulation_nvz=int(job["nvz"]),
        dz=dz,
        multienergy=multienergy,
        energy_grid=job.get("energy_grid", "mono"),
        energies=energies,
        signed_frequencies=signed_frequencies,
        energy_mode_weights=energy_mode_weights,
        reference_energy=float(job.get("reference_energy", "10.0")),
    )
    if config.nvz < 2 or config.num < 2:
        raise ValueError("LSA_NVZ and LSA_NUM must both be at least 2")
    if config.vmax <= config.vmin or config.kmax <= config.kmin:
        raise ValueError("The velocity interval and LSA k interval must increase")
    if config.neutrino_width <= 0 or config.antineutrino_width <= 0:
        raise ValueError("Angular-distribution widths must be positive")
    if config.angular_profile not in {
        "gaussian", "two_beam", "symmetric_two_beam", "soumya_two_beam"
    }:
        raise ValueError(
            "ANGULAR_PROFILE must be gaussian, two_beam, symmetric_two_beam, or soumya_two_beam"
        )
    if config.angular_profile in {
        "two_beam", "symmetric_two_beam", "soumya_two_beam"
    } and config.nvz != 2:
        raise ValueError("Two-beam profiles require LSA_NVZ=2")
    if config.damping_rate < 0 or config.antineutrino_collision_ratio < 0:
        raise ValueError("Collision rate and antineutrino collision ratio must be non-negative")
    if config.kmin < -power_kmax or config.kmax > power_kmax:
        raise ValueError(
            "The LSA k interval must lie inside COSEnu's -K_MAX to +K_MAX interval"
        )
    return config


def normalized_gaussian(v: np.ndarray, width: float, mean: float = 1.0) -> np.ndarray:
    normalization = width * sqrt(pi / 2.0) * (
        erf((1.0 + mean) / (width * sqrt(2.0)))
        + erf((1.0 - mean) / (width * sqrt(2.0)))
    )
    return np.exp(-((v - mean) ** 2) / (2.0 * width**2)) / normalization


def angular_background(config: LSAConfig) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    dv = (config.vmax - config.vmin) / config.nvz
    v = config.vmin + (np.arange(config.nvz) + 0.5) * dv
    if config.multienergy:
        modes_per_sector = config.signed_frequencies.size // 2
        neutrino_number = float(
            np.sum(config.energy_mode_weights[:modes_per_sector])
        )
        antineutrino_number = float(
            -np.sum(config.energy_mode_weights[modes_per_sector:])
        )
        rho = neutrino_number * normalized_gaussian(
            v, config.neutrino_width
        )
        antirho = antineutrino_number * normalized_gaussian(
            v, config.antineutrino_width
        )
        return v, dv, rho, antirho
    # This is the electron-flavor fraction at the center of COSEnu's seed.
    electron_fraction = 0.5 * (1.0 + sqrt(max(0.0, 1.0 - config.perturbation_size**2)))
    if config.angular_profile == "two_beam":
        # Exact quadrature for delta(v_z-1) neutrinos and
        # alpha*delta(v_z+1) antineutrinos, matching the paper benchmark.
        v = np.array([-1.0, 1.0])
        dv = 1.0
        rho = np.array([0.0, electron_fraction])
        antirho = np.array([config.alpha * electron_fraction, 0.0])
        return v, dv, rho, antirho
    if config.angular_profile == "symmetric_two_beam":
        # Equal v_z=+/-1 populations with the same integrated normalization
        # as the continuous profiles.  The ELN is positive in both beams for
        # alpha<1, so this profile has no fast angular crossing.
        v = np.array([-1.0, 1.0])
        dv = 1.0
        rho = np.full(2, 0.5 * electron_fraction)
        antirho = np.full(2, 0.5 * config.alpha * electron_fraction)
        return v, dv, rho, antirho
    if config.angular_profile == "soumya_two_beam":
        # The quoted occupation numbers apply separately to each beam.
        v = np.array([-1.0, 1.0])
        dv = 1.0
        rho = np.full(2, electron_fraction)
        antirho = np.full(2, config.alpha * electron_fraction)
        return v, dv, rho, antirho
    rho = electron_fraction * normalized_gaussian(v, config.neutrino_width)
    antirho = (
        electron_fraction
        * config.alpha
        * normalized_gaussian(v, config.antineutrino_width)
    )
    return v, dv, rho, antirho


def stability_matrix(
    config: LSAConfig,
    v: np.ndarray,
    dv: float,
    rho: np.ndarray,
    antirho: np.ndarray,
    k: float,
) -> np.ndarray:
    n = config.nvz
    coupling = config.mu * dv * (1.0 - v[:, None] * v[None, :])
    hee = coupling @ (rho - antirho)
    matrix = np.zeros((2 * n, 2 * n), dtype=np.complex128)
    matrix[:n, :n] = -rho[:, None] * coupling
    matrix[:n, n:] = rho[:, None] * coupling
    matrix[n:, :n] = -antirho[:, None] * coupling
    matrix[n:, n:] = antirho[:, None] * coupling

    # For modes proportional to exp(-i Omega t), dS/dt = -C S contributes
    # -i C to Omega. Unequal neutrino and antineutrino rates can therefore
    # produce a collisional flavor instability even without an ELN crossing.
    neutrino_rate = config.damping_rate
    antineutrino_rate = config.antineutrino_collision_ratio * neutrino_rate
    indices = np.arange(n)
    common = hee + config.matter_potential + v * k
    matrix[indices, indices] += (
        common - config.effective_omega - 1j * neutrino_rate
    )
    matrix[n + indices, n + indices] += (
        common
        + config.effective_omega
        - 1j * antineutrino_rate
    )
    return matrix


def scan_modes(
    config: LSAConfig, v: np.ndarray, dv: float, rho: np.ndarray, antirho: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, complex, np.ndarray, np.ndarray]:
    k_values = np.linspace(config.kmin, config.kmax, config.num)
    roots = np.empty(config.num, dtype=np.complex128)
    champion_index = 0
    champion_root = 0.0j
    champion_vector = np.empty(2 * config.nvz, dtype=np.complex128)
    champion_spectrum = np.empty(2 * config.nvz, dtype=np.complex128)

    print(f"Scanning {config.num} wavenumbers with {config.nvz} LSA angular bins...")
    for index, k in enumerate(k_values):
        matrix = stability_matrix(config, v, dv, rho, antirho, float(k))
        eigenvalues = np.linalg.eigvals(matrix)
        root = eigenvalues[int(np.argmax(eigenvalues.imag))]
        roots[index] = root
        if index == 0 or root.imag > champion_root.imag:
            champion_index = index
            champion_root = root
            champion_spectrum = eigenvalues
        displayed_growth = 0.0 if abs(root.imag) < NUMERICAL_ZERO else root.imag
        print(
            f"  {index + 1:>{len(str(config.num))}}/{config.num} "
            f"({100.0 * (index + 1) / config.num:5.1f}%)  k={k: .6g}  "
            f"Im(Omega)={displayed_growth:.6g}",
            flush=True,
        )

    # Recompute only the champion with eigenvectors. This keeps most of the scan faster.
    matrix = stability_matrix(
        config, v, dv, rho, antirho, float(k_values[champion_index])
    )
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    root_index = int(np.argmax(eigenvalues.imag))
    champion_root = eigenvalues[root_index]
    champion_vector = eigenvectors[:, root_index]
    champion_spectrum = eigenvalues
    roots[champion_index] = champion_root
    return (
        k_values,
        roots,
        np.array([champion_index]),
        champion_root,
        champion_vector,
        champion_spectrum,
    )


def multi_energy_operator_data(
    config: LSAConfig, v: np.ndarray, dv: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened mode coordinates and quadrature weights for Eq. (20)."""
    frequencies = config.signed_frequencies
    mode_weights = config.energy_mode_weights
    if frequencies.size == 0 or frequencies.size % 2:
        raise ValueError("The multi-energy grid must contain equal neutrino/antineutrino sectors")
    modes_per_sector = frequencies.size // 2
    profiles = np.empty((frequencies.size, v.size))
    profiles[:modes_per_sector] = normalized_gaussian(
        v, config.neutrino_width
    )[None, :]
    profiles[modes_per_sector:] = normalized_gaussian(
        v, config.antineutrino_width
    )[None, :]
    quadrature_weights = mode_weights[:, None] * profiles * dv
    flat_v = np.tile(v, frequencies.size)
    flat_omega = np.repeat(frequencies, v.size)
    flat_weights = quadrature_weights.reshape(-1)
    damping = np.where(
        flat_omega > 0.0,
        config.damping_rate,
        config.damping_rate * config.antineutrino_collision_ratio,
    )
    return flat_v, flat_omega, flat_weights, damping


def multi_energy_matrix_or_operator(
    config: LSAConfig,
    flat_v: np.ndarray,
    flat_omega: np.ndarray,
    flat_weights: np.ndarray,
    damping: np.ndarray,
    k: float,
):
    """Build the diagonal-plus-rank-2 discretization of article Eq. (20)."""
    diagonal = multi_energy_diagonal(
        config, flat_v, flat_omega, flat_weights, damping, k
    )
    size = diagonal.size
    if size <= 600:
        kernel = 1.0 - flat_v[:, None] * flat_v[None, :]
        return np.diag(diagonal) - config.mu * kernel * flat_weights[None, :]
    raise ValueError("Use the rank-2 dispersion determinant for matrices larger than 600")


def multi_energy_diagonal(
    config: LSAConfig,
    flat_v: np.ndarray,
    flat_omega: np.ndarray,
    flat_weights: np.ndarray,
    damping: np.ndarray,
    k: float,
) -> np.ndarray:
    eta = float(np.sum(flat_weights))
    current = float(np.sum(flat_weights * flat_v))
    vacuum_factor = (
        config.effective_omega / config.omega_over_mu
        if config.omega_over_mu != 0.0
        else 0.0
    )
    diagonal = (
        -vacuum_factor * flat_omega
        + k * flat_v
        + config.matter_potential
        + config.mu * (eta - flat_v * current)
        - 1j * damping
    )
    return diagonal


def multi_energy_determinant(
    omega_root: complex,
    diagonal: np.ndarray,
    flat_v: np.ndarray,
    flat_weights: np.ndarray,
    mu: float,
) -> tuple[complex, complex]:
    """Rank-2 dispersion determinant and its complex derivative."""
    denominator = omega_root - diagonal
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        response = flat_weights / denominator
        derivative_response = -flat_weights / denominator**2
        moments = np.asarray(
            [np.sum(response), np.sum(flat_v * response), np.sum(flat_v**2 * response)]
        )
        derivatives = np.asarray(
            [
                np.sum(derivative_response),
                np.sum(flat_v * derivative_response),
                np.sum(flat_v**2 * derivative_response),
            ]
        )
    i0, i1, i2 = moments
    d0, d1, d2 = derivatives
    determinant = (1.0 + mu * i0) * (1.0 - mu * i2) + mu**2 * i1**2
    derivative = (
        mu * d0 * (1.0 - mu * i2)
        - mu * d2 * (1.0 + mu * i0)
        + 2.0 * mu**2 * i1 * d1
    )
    return complex(determinant), complex(derivative)


def multi_energy_dispersion_roots(
    diagonal: np.ndarray,
    flat_v: np.ndarray,
    flat_weights: np.ndarray,
    mu: float,
    previous_roots: list[complex] | None = None,
) -> list[complex]:
    """Locate collective roots without diagonalizing the continuum matrix."""
    real_diagonal = diagonal.real
    real_seeds = np.linspace(float(real_diagonal.min()), float(real_diagonal.max()), 16)
    imag_seeds = (0.005, 0.02, 0.06, 0.12, 0.24, 0.4)
    seeds = list(previous_roots or [])
    seeds.extend(complex(real, imag) for real in real_seeds for imag in imag_seeds)
    roots: list[complex] = []
    damping_present = bool(np.any(np.abs(diagonal.imag) > NUMERICAL_ZERO))
    for seed in seeds:
        candidate = seed
        converged = False
        for _ in range(80):
            determinant, derivative = multi_energy_determinant(
                candidate, diagonal, flat_v, flat_weights, mu
            )
            if (
                not np.isfinite(determinant.real)
                or not np.isfinite(determinant.imag)
                or abs(derivative) < 1.0e-15
            ):
                break
            step = determinant / derivative
            if abs(step) > 0.2:
                step *= 0.2 / abs(step)
            candidate -= step
            if abs(step) < 1.0e-11 * (1.0 + abs(candidate)):
                residual, _ = multi_energy_determinant(
                    candidate, diagonal, flat_v, flat_weights, mu
                )
                converged = abs(residual) < 1.0e-8
                break
        if not converged:
            continue
        if not damping_present and candidate.imag < 0.0:
            candidate = candidate.conjugate()
        if candidate.imag < -NUMERICAL_ZERO:
            continue
        if all(abs(candidate - known) > 1.0e-5 for known in roots):
            roots.append(candidate)
    return roots


def multi_energy_root_vector(
    root: complex,
    diagonal: np.ndarray,
    flat_v: np.ndarray,
    flat_weights: np.ndarray,
    mu: float,
) -> np.ndarray:
    denominator = root - diagonal
    response = flat_weights / denominator
    i0 = np.sum(response)
    i1 = np.sum(flat_v * response)
    i2 = np.sum(flat_v**2 * response)
    reduced = np.array(
        [[1.0 + mu * i0, -mu * i1], [mu * i1, 1.0 - mu * i2]],
        dtype=np.complex128,
    )
    _, _, right = np.linalg.svd(reduced)
    moment0, moment1 = right[-1].conjugate()
    return mu * (-moment0 + flat_v * moment1) / denominator


def leading_multi_energy_eigenpairs(matrix_or_operator, vectors: bool):
    if vectors:
        values, modes = np.linalg.eig(matrix_or_operator)
        return values, modes
    return np.linalg.eigvals(matrix_or_operator), None


def scan_multi_energy_modes(
    config: LSAConfig, v: np.ndarray, dv: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, complex, np.ndarray, np.ndarray]:
    k_values = np.linspace(config.kmin, config.kmax, config.num)
    roots = np.empty(config.num, dtype=np.complex128)
    flat_v, flat_omega, flat_weights, damping = multi_energy_operator_data(
        config, v, dv
    )
    dimension = flat_v.size
    print(
        f"Scanning {config.num} wavenumbers with {len(config.signed_frequencies)} "
        f"energy modes x {config.nvz} angular bins "
        f"(matrix dimension {dimension})..."
    )
    champion_index = 0
    champion_root = -1j * np.inf
    champion_spectrum = np.empty(0, dtype=np.complex128)
    previous_roots: list[complex] = []
    for index, k in enumerate(k_values):
        if dimension <= 600:
            operator = multi_energy_matrix_or_operator(
                config, flat_v, flat_omega, flat_weights, damping, float(k)
            )
            eigenvalues, _ = leading_multi_energy_eigenpairs(operator, vectors=False)
        else:
            diagonal = multi_energy_diagonal(
                config, flat_v, flat_omega, flat_weights, damping, float(k)
            )
            found = multi_energy_dispersion_roots(
                diagonal, flat_v, flat_weights, config.mu, previous_roots
            )
            if not found:
                raise RuntimeError(
                    f"Could not locate a collective multi-energy LSA root at k={k:.8g}"
                )
            previous_roots = found
            eigenvalues = np.asarray(found, dtype=np.complex128)
        root = complex(eigenvalues[int(np.argmax(eigenvalues.imag))])
        roots[index] = root
        if index == 0 or root.imag > champion_root.imag:
            champion_index = index
            champion_root = root
            champion_spectrum = eigenvalues
        displayed_growth = 0.0 if abs(root.imag) < NUMERICAL_ZERO else root.imag
        print(
            f"  {index + 1:>{len(str(config.num))}}/{config.num} "
            f"({100.0 * (index + 1) / config.num:5.1f}%)  k={k: .6g}  "
            f"Im(Omega)={displayed_growth:.6g}",
            flush=True,
        )

    if dimension <= 600:
        champion_operator = multi_energy_matrix_or_operator(
            config,
            flat_v,
            flat_omega,
            flat_weights,
            damping,
            float(k_values[champion_index]),
        )
        eigenvalues, eigenvectors = leading_multi_energy_eigenpairs(
            champion_operator, vectors=True
        )
        root_index = int(np.argmax(eigenvalues.imag))
        champion_root = complex(eigenvalues[root_index])
        champion_vector = eigenvectors[:, root_index]
        champion_spectrum = eigenvalues
    else:
        diagonal = multi_energy_diagonal(
            config,
            flat_v,
            flat_omega,
            flat_weights,
            damping,
            float(k_values[champion_index]),
        )
        champion_roots = multi_energy_dispersion_roots(
            diagonal, flat_v, flat_weights, config.mu, [champion_root]
        )
        champion_spectrum = np.asarray(champion_roots, dtype=np.complex128)
        champion_root = complex(
            champion_spectrum[int(np.argmax(champion_spectrum.imag))]
        )
        champion_vector = multi_energy_root_vector(
            champion_root, diagonal, flat_v, flat_weights, config.mu
        )
    roots[champion_index] = champion_root
    return (
        k_values,
        roots,
        np.array([champion_index]),
        champion_root,
        champion_vector,
        champion_spectrum,
    )


def save_figure(fig: plt.Figure, destination: Path) -> None:
    # Apply minor ticks consistently, including physical-unit secondary axes.
    pending_axes = list(fig.axes)
    visited_axes: set[int] = set()
    while pending_axes:
        ax = pending_axes.pop()
        if id(ax) in visited_axes:
            continue
        visited_axes.add(id(ax))
        ax.minorticks_on()
        pending_axes.extend(ax.child_axes)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  created {destination.name}")


def plot_background(
    output_dir: Path, v: np.ndarray, rho: np.ndarray, antirho: np.ndarray
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].plot(v, rho, label=r"$\rho_{ee}$")
    axes[0].plot(v, antirho, label=r"$\bar\rho_{ee}$")
    axes[0].set(xlabel=r"$v_z$", ylabel="Angular density", xlim=(-1.0, 1.0))
    axes[0].legend()
    axes[1].plot(v, rho - antirho, color="tab:purple")
    axes[1].axhline(0.0, color="black", lw=0.8, ls="--")
    axes[1].set(xlabel=r"$v_z$", ylabel=r"$\rho_{ee}-\bar\rho_{ee}$", xlim=(-1.0, 1.0))
    save_figure(fig, output_dir / "background.pdf")


def plot_dispersion(
    output_dir: Path, k_values: np.ndarray, roots: np.ndarray, champion_index: int
) -> None:
    growth = np.where(np.abs(roots.imag) < NUMERICAL_ZERO, 0.0, roots.imag)
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 6.0), sharex=True)
    axes[0].plot(k_values, roots.real, color="tab:blue")
    axes[0].set_ylabel(r"Re$(\Omega)$ [$\mu$]")
    axes[1].plot(k_values, growth, color="tab:red")
    axes[1].set(xlabel=r"$k$ [$\mu$]", ylabel=r"Im$(\Omega)$ [$\mu$]")
    for ax in axes:
        ax.axvline(0.0, color="0.35", ls=":", lw=1.0)
        ax.axvline(k_values[champion_index], color="black", ls="-", lw=1.0)
    save_figure(fig, output_dir / "dispersion.pdf")


def angular_branch_roots_at_zero(
    config: LSAConfig,
    number_neutrinos: float,
    number_antineutrinos: float,
    lepton_asymmetry: float,
    angular_eigenvalue: float,
    signed_omega: float,
) -> np.ndarray:
    """Return both energy roots for one angular branch at k'=0."""
    common = config.matter_potential + config.mu * lepton_asymmetry
    energy_matrix = np.array(
        [
            [
                common
                - signed_omega
                - config.mu * angular_eigenvalue * number_neutrinos,
                config.mu * angular_eigenvalue * number_antineutrinos,
            ],
            [
                -config.mu * angular_eigenvalue * number_neutrinos,
                common
                + signed_omega
                + config.mu * angular_eigenvalue * number_antineutrinos,
            ],
        ],
        dtype=np.complex128,
    )
    return np.linalg.eigvals(energy_matrix)


def angular_branch_root_at_zero(
    config: LSAConfig,
    number_neutrinos: float,
    number_antineutrinos: float,
    lepton_asymmetry: float,
    angular_eigenvalue: float,
    signed_omega: float,
) -> complex:
    """Return the fastest-growing single-energy root at k'=0."""
    roots = angular_branch_roots_at_zero(
        config,
        number_neutrinos,
        number_antineutrinos,
        lepton_asymmetry,
        angular_eigenvalue,
        signed_omega,
    )
    return complex(roots[int(np.argmax(roots.imag))])


def exact_two_beam_branch_curves(
    config: LSAConfig,
    v: np.ndarray,
    dv: float,
    rho: np.ndarray,
    antirho: np.ndarray,
    kprime_values: np.ndarray,
    lambda_plus: float,
    lambda_minus: float,
    number_neutrinos: float,
    number_antineutrinos: float,
    lepton_asymmetry: float,
    lepton_current: float,
) -> dict[tuple[str, str], np.ndarray]:
    """Track all four two-beam roots through finite-k mode mergers."""
    zero_index = int(np.argmin(np.abs(kprime_values)))
    branch_labels = ("lambda_plus", "lambda_minus")
    angular_eigenvalues = (lambda_plus, lambda_minus)
    curves: dict[tuple[str, str], np.ndarray] = {}

    for ordering, ordering_sign in {"NO": 1.0, "IO": -1.0}.items():
        signed_omega = ordering_sign * abs(config.effective_omega)
        ordered_config = replace(config, effective_omega=signed_omega)
        target_roots = np.concatenate(
            [
                angular_branch_roots_at_zero(
                    config,
                    number_neutrinos,
                    number_antineutrinos,
                    lepton_asymmetry,
                    angular_eigenvalue,
                    signed_omega,
                )
                for angular_eigenvalue in angular_eigenvalues
            ]
        )
        spectrum_at_zero = np.linalg.eigvals(
            stability_matrix(
                ordered_config, v, dv, rho, antirho, lepton_current
            )
        )
        initial_order = min(
            permutations(range(len(spectrum_at_zero))),
            key=lambda order: float(
                np.sum(np.abs(spectrum_at_zero[list(order)] - target_roots))
            ),
        )
        tracks = np.full(
            (len(spectrum_at_zero), len(kprime_values)),
            np.nan + 1j * np.nan,
            dtype=np.complex128,
        )
        tracks[:, zero_index] = spectrum_at_zero[list(initial_order)]

        for indices in (
            range(zero_index + 1, len(kprime_values)),
            range(zero_index - 1, -1, -1),
        ):
            previous = tracks[:, zero_index]
            for index in indices:
                actual_k = float(kprime_values[index] + lepton_current)
                spectrum = np.linalg.eigvals(
                    stability_matrix(
                        ordered_config, v, dv, rho, antirho, actual_k
                    )
                )
                best_order = min(
                    permutations(range(len(spectrum))),
                    key=lambda order: float(
                        np.sum(np.abs(spectrum[list(order)] - previous))
                    ),
                )
                previous = spectrum[list(best_order)]
                tracks[:, index] = previous

        curves[(ordering, branch_labels[0])] = tracks[0:2]
        curves[(ordering, branch_labels[1])] = tracks[2:4]
    return curves


def angular_branch_determinant(
    root: complex,
    kprime: float,
    signed_omega: float,
    config: LSAConfig,
    v: np.ndarray,
    dv: float,
    angular_distribution: np.ndarray,
    number_neutrinos: float,
    number_antineutrinos: float,
    lepton_asymmetry: float,
) -> tuple[complex, complex]:
    """Rank-2 single-energy dispersion determinant and d/dOmega."""
    common = config.matter_potential + config.mu * lepton_asymmetry
    neutrino_denominator = root - (common - signed_omega) - kprime * v
    antineutrino_denominator = root - (common + signed_omega) - kprime * v
    response = (
        number_neutrinos / neutrino_denominator
        - number_antineutrinos / antineutrino_denominator
    )
    derivative_response = (
        -number_neutrinos / neutrino_denominator**2
        + number_antineutrinos / antineutrino_denominator**2
    )
    moments = np.array(
        [
            dv * np.sum(angular_distribution * v**power * response)
            for power in range(3)
        ]
    )
    derivatives = np.array(
        [
            dv
            * np.sum(angular_distribution * v**power * derivative_response)
            for power in range(3)
        ]
    )
    integral0, integral1, integral2 = moments
    derivative0, derivative1, derivative2 = derivatives
    mu = config.mu
    determinant = (
        (1.0 + mu * integral0) * (1.0 - mu * integral2)
        + mu**2 * integral1**2
    )
    derivative = (
        mu * derivative0 * (1.0 - mu * integral2)
        - mu * derivative2 * (1.0 + mu * integral0)
        + 2.0 * mu**2 * integral1 * derivative1
    )
    return complex(determinant), complex(derivative)


def continue_angular_branch(
    seed: complex,
    kprime_values: np.ndarray,
    signed_omega: float,
    config: LSAConfig,
    v: np.ndarray,
    dv: float,
    angular_distribution: np.ndarray,
    number_neutrinos: float,
    number_antineutrinos: float,
    lepton_asymmetry: float,
) -> np.ndarray:
    """Continue one unstable root away from k'=0 in both directions."""
    roots = np.full(len(kprime_values), np.nan + 1j * np.nan, dtype=np.complex128)
    zero_index = int(np.argmin(np.abs(kprime_values)))
    roots[zero_index] = seed
    growth_tolerance = 1.0e-8
    residual_tolerance = 1.0e-9

    def solve(initial: complex, kprime: float) -> complex | None:
        root = initial
        for _ in range(60):
            determinant, derivative = angular_branch_determinant(
                root,
                kprime,
                signed_omega,
                config,
                v,
                dv,
                angular_distribution,
                number_neutrinos,
                number_antineutrinos,
                lepton_asymmetry,
            )
            if not np.isfinite(determinant) or abs(derivative) < 1.0e-14:
                return None
            step = determinant / derivative
            if abs(step) > 0.1:
                step *= 0.1 / abs(step)
            root -= step
            if abs(step) < 1.0e-12 * (1.0 + abs(root)):
                residual, _ = angular_branch_determinant(
                    root,
                    kprime,
                    signed_omega,
                    config,
                    v,
                    dv,
                    angular_distribution,
                    number_neutrinos,
                    number_antineutrinos,
                    lepton_asymmetry,
                )
                if abs(residual) <= residual_tolerance:
                    return root.conjugate() if root.imag < 0.0 else root
                return None
        return None

    for indices in (
        range(zero_index + 1, len(kprime_values)),
        range(zero_index - 1, -1, -1),
    ):
        previous = seed
        previous_kprime = 0.0
        for index in indices:
            target_kprime = float(kprime_values[index])
            substeps = max(
                1, int(np.ceil(abs(target_kprime - previous_kprime) / 0.005))
            )
            root: complex | None = previous
            for intermediate_kprime in np.linspace(
                previous_kprime, target_kprime, substeps + 1
            )[1:]:
                root = solve(root, float(intermediate_kprime))
                if root is None or root.imag <= growth_tolerance:
                    break
            if root is None or root.imag <= growth_tolerance:
                break
            roots[index] = root
            previous = root
            previous_kprime = target_kprime
    return roots


def add_branch_arrow(
    ax: plt.Axes,
    kprime: np.ndarray,
    values: np.ndarray,
    color: str,
) -> None:
    """Show the continuation direction away from k'=0 on each branch side."""
    for side in (kprime < 0.0, kprime > 0.0):
        indices = np.flatnonzero(side & np.isfinite(values))
        if len(indices) < 5:
            continue
        indices = indices[::-1] if np.all(kprime[indices] < 0.0) else indices
        position = max(1, min(len(indices) - 2, len(indices) // 2))
        start = int(indices[position - 1])
        end = int(indices[position])
        ax.annotate(
            "",
            xy=(kprime[end], values[end]),
            xytext=(kprime[start], values[start]),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2},
        )


def render_angular_branches(
    output_dir: Path,
    config: LSAConfig,
    kprime_values: np.ndarray,
    lambda_plus: float,
    lambda_minus: float,
    lepton_asymmetry: float,
    lepton_current: float,
    curves: dict[tuple[str, str], np.ndarray],
    title: str,
    terminal_label: str,
) -> AngularBranchResult:
    """Render Appendix-C branches shared by mono- and multi-energy LSA."""
    colors = {"lambda_plus": "#7B5AA6", "lambda_minus": "#009E73"}
    orderings = {"NO": "-", "IO": "--"}
    fig, axes = plt.subplots(2, 1, figsize=(5.2, 6.2), sharex=True)
    for ordering, linestyle in orderings.items():
        for branch_name in ("lambda_plus", "lambda_minus"):
            roots = np.asarray(curves[(ordering, branch_name)])
            root_curves = roots[np.newaxis, :] if roots.ndim == 1 else roots
            color = colors[branch_name]
            for root_curve in root_curves:
                valid = np.isfinite(root_curve)
                if not np.any(valid):
                    continue
                axes[0].plot(
                    kprime_values[valid], root_curve.real[valid], color=color,
                    linestyle=linestyle, linewidth=2.0,
                )
                axes[1].plot(
                    kprime_values[valid], root_curve.imag[valid], color=color,
                    linestyle=linestyle, linewidth=2.0,
                )
                add_branch_arrow(axes[0], kprime_values, root_curve.real, color)
                add_branch_arrow(axes[1], kprime_values, root_curve.imag, color)

    axes[0].set_ylabel(r"Re$(\Omega)$ [$\mu$]")
    axes[1].set(
        xlabel=r"Shifted wavenumber $k'=k-J_z$ [$\mu$]",
        ylabel=r"Im$(\Omega)$ [$\mu$]",
        xlim=(config.kmin, config.kmax),
    )
    axes[1].set_ylim(bottom=0.0)
    legend_handles = [
        Line2D(
            [0], [0], color=colors["lambda_plus"], lw=2.0,
            label=rf"$\lambda_+={lambda_plus:.3f}$",
        ),
        Line2D(
            [0], [0], color=colors["lambda_minus"], lw=2.0,
            label=rf"$\lambda_-={lambda_minus:.3f}$",
        ),
        Line2D([0], [0], color="black", lw=1.8, ls="-", label="NO"),
        Line2D([0], [0], color="black", lw=1.8, ls="--", label="IO"),
    ]
    axes[0].legend(handles=legend_handles, fontsize=8.5, ncol=2, loc="best")
    axes[0].set_title(title)
    axes[0].text(
        0.02,
        0.04,
        rf"$J_z/\mu={lepton_current:.4g}$",
        transform=axes[0].transAxes,
        fontsize=8.2,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )
    save_figure(fig, output_dir / "branches.pdf")

    data_columns: list[np.ndarray] = [kprime_values]
    data_names = ["kprime_over_mu"]
    for ordering in ("NO", "IO"):
        for branch_name in ("lambda_plus", "lambda_minus"):
            roots = np.asarray(curves[(ordering, branch_name)])
            root_curves = roots[np.newaxis, :] if roots.ndim == 1 else roots
            for mode_index, root_curve in enumerate(root_curves, start=1):
                suffix = "" if len(root_curves) == 1 else f"_mode{mode_index}"
                data_columns.extend((root_curve.real, root_curve.imag))
                data_names.extend(
                    (
                        f"Re_Omega_{branch_name}{suffix}_{ordering}_over_mu",
                        f"Im_Omega_{branch_name}{suffix}_{ordering}_over_mu",
                    )
                )
    np.savetxt(
        output_dir / "branches.dat",
        np.column_stack(data_columns),
        header="  ".join(data_names),
    )

    def peak(ordering: str) -> tuple[float, float]:
        candidates: list[tuple[float, float]] = []
        for branch_name in ("lambda_plus", "lambda_minus"):
            roots = np.asarray(curves[(ordering, branch_name)])
            root_curves = roots[np.newaxis, :] if roots.ndim == 1 else roots
            for root_curve in root_curves:
                valid = np.flatnonzero(np.isfinite(root_curve))
                if len(valid):
                    index = int(valid[np.argmax(root_curve.imag[valid])])
                    candidates.append(
                        (float(root_curve.imag[index]), float(kprime_values[index]))
                    )
        return max(candidates, default=(0.0, 0.0))

    no_peak_gamma, no_peak_kprime = peak("NO")
    io_peak_gamma, io_peak_kprime = peak("IO")
    print(
        f"  {terminal_label}: lambda+={lambda_plus:.8g}, "
        f"lambda-={lambda_minus:.8g}, Jz={lepton_current:.8g} mu"
    )
    return AngularBranchResult(
        lambda_plus=lambda_plus,
        lambda_minus=lambda_minus,
        lepton_asymmetry=lepton_asymmetry,
        lepton_current=lepton_current,
        no_peak_gamma=no_peak_gamma,
        no_peak_kprime=no_peak_kprime,
        io_peak_gamma=io_peak_gamma,
        io_peak_kprime=io_peak_kprime,
    )


def multi_energy_branch_seed(
    config: LSAConfig,
    angular_eigenvalue: float,
    ordering_sign: float,
    eta: float,
) -> complex | None:
    """Find the growing k'=0 root of Appendix-C Eq. (C6)."""
    vacuum_factor = (
        abs(config.effective_omega / config.omega_over_mu)
        if config.omega_over_mu != 0.0
        else 0.0
    )
    diagonal = (
        -ordering_sign * vacuum_factor * config.signed_frequencies
        + config.matter_potential
        + config.mu * eta
    )
    energy_operator = np.diag(diagonal) - config.mu * angular_eigenvalue * np.tile(
        config.energy_mode_weights, (diagonal.size, 1)
    )
    roots = np.linalg.eigvals(energy_operator)
    growing = roots[roots.imag > 1.0e-8]
    if growing.size == 0:
        return None
    return complex(growing[int(np.argmax(growing.imag))])


def continue_multi_energy_branch(
    seed: complex,
    kprime_values: np.ndarray,
    ordering_sign: float,
    config: LSAConfig,
    flat_v: np.ndarray,
    flat_omega: np.ndarray,
    flat_weights: np.ndarray,
    damping: np.ndarray,
) -> np.ndarray:
    """Continue one Eq. (C6) root through the full multi-energy determinant."""
    roots = np.full(len(kprime_values), np.nan + 1j * np.nan, dtype=np.complex128)
    zero_index = int(np.argmin(np.abs(kprime_values)))
    roots[zero_index] = seed
    eta = float(np.sum(flat_weights))
    vacuum_factor = (
        abs(config.effective_omega / config.omega_over_mu)
        if config.omega_over_mu != 0.0
        else 0.0
    )
    base_diagonal = (
        -ordering_sign * vacuum_factor * flat_omega
        + config.matter_potential
        + config.mu * eta
        - 1j * damping
    )

    def solve(initial: complex, kprime: float) -> complex | None:
        diagonal = base_diagonal + kprime * flat_v
        root = initial
        for _ in range(80):
            determinant, derivative = multi_energy_determinant(
                root, diagonal, flat_v, flat_weights, config.mu
            )
            if not np.isfinite(determinant) or abs(derivative) < 1.0e-15:
                return None
            step = determinant / derivative
            if abs(step) > 0.1:
                step *= 0.1 / abs(step)
            root -= step
            if abs(step) < 1.0e-11 * (1.0 + abs(root)):
                residual, _ = multi_energy_determinant(
                    root, diagonal, flat_v, flat_weights, config.mu
                )
                if abs(residual) <= 1.0e-8:
                    return root.conjugate() if root.imag < 0.0 else root
                return None
        return None

    for indices in (
        range(zero_index + 1, len(kprime_values)),
        range(zero_index - 1, -1, -1),
    ):
        previous = seed
        previous_kprime = 0.0
        for index in indices:
            target_kprime = float(kprime_values[index])
            substeps = max(
                1, int(np.ceil(abs(target_kprime - previous_kprime) / 0.005))
            )
            root: complex | None = previous
            for intermediate_kprime in np.linspace(
                previous_kprime, target_kprime, substeps + 1
            )[1:]:
                root = solve(root, float(intermediate_kprime))
                if root is None or root.imag <= 1.0e-8:
                    break
            if root is None or root.imag <= 1.0e-8:
                break
            roots[index] = root
            previous = root
            previous_kprime = target_kprime
    return roots


def plot_multi_energy_angular_branches(
    output_dir: Path,
    config: LSAConfig,
    v: np.ndarray,
    dv: float,
) -> AngularBranchResult:
    """Make the multi-energy Appendix-C Figure 12 branch decomposition."""
    profile = normalized_gaussian(v, config.neutrino_width)
    angular_moments = np.array(
        [dv * np.sum(profile * v**power) for power in range(3)]
    )
    moment_matrix = np.array(
        [
            [angular_moments[0], angular_moments[1]],
            [-angular_moments[1], -angular_moments[2]],
        ]
    )
    angular_eigenvalues = np.sort(np.linalg.eigvals(moment_matrix).real)[::-1]
    lambda_plus = float(angular_eigenvalues[0])
    lambda_minus = float(angular_eigenvalues[1])
    kprime_values = np.unique(
        np.concatenate(
            (np.linspace(config.kmin, config.kmax, config.num), np.array([0.0]))
        )
    )
    kprime_values.sort()
    flat_v, flat_omega, flat_weights, damping = multi_energy_operator_data(
        config, v, dv
    )
    eta = float(np.sum(flat_weights))
    lepton_current = config.mu * float(np.sum(flat_weights * flat_v))
    labels = {"lambda_plus": lambda_plus, "lambda_minus": lambda_minus}
    orderings = {"NO": 1.0, "IO": -1.0}
    curves: dict[tuple[str, str], np.ndarray] = {}
    for ordering, ordering_sign in orderings.items():
        for branch_name, angular_eigenvalue in labels.items():
            seed = multi_energy_branch_seed(
                config, angular_eigenvalue, ordering_sign, eta
            )
            if seed is None:
                curves[(ordering, branch_name)] = np.full(
                    len(kprime_values),
                    np.nan + 1j * np.nan,
                    dtype=np.complex128,
                )
                continue
            curves[(ordering, branch_name)] = continue_multi_energy_branch(
                seed,
                kprime_values,
                ordering_sign,
                config,
                flat_v,
                flat_omega,
                flat_weights,
                damping,
            )
    return render_angular_branches(
        output_dir,
        config,
        kprime_values,
        lambda_plus,
        lambda_minus,
        eta,
        lepton_current,
        curves,
        "Multi-energy angular-mode branches",
        "Fig. 12 multi-energy branches",
    )


def plot_single_energy_angular_branches(
    output_dir: Path,
    config: LSAConfig,
    v: np.ndarray,
    dv: float,
    rho: np.ndarray,
    antirho: np.ndarray,
) -> AngularBranchResult | None:
    """Make the mono- or multi-energy equivalent of Appendix-C Figure 12."""
    for previous_output in (
        output_dir / "branches.pdf",
        output_dir / "branches.dat",
        output_dir / "fig12_single_energy.png",
        output_dir / "fig12_single_energy.dat",
    ):
        previous_output.unlink(missing_ok=True)
    if config.mu == 0.0:
        print("  Fig. 12 branch plot skipped: self-interactions are disabled")
        return None
    if config.damping_rate != 0.0:
        print("  Fig. 12 branch plot skipped: Appendix C assumes collisionless evolution")
        return None
    if not (config.kmin < 0.0 < config.kmax):
        print("  Fig. 12 branch plot skipped: the LSA k interval must contain k'=0")
        return None

    number_neutrinos = float(dv * np.sum(rho))
    number_antineutrinos = float(dv * np.sum(antirho))
    if number_neutrinos <= 0.0 or number_antineutrinos <= 0.0:
        print("  Fig. 12 branch plot skipped: both populations must be nonzero")
        return None
    angular_distribution = rho / number_neutrinos
    antineutrino_distribution = antirho / number_antineutrinos
    if not np.allclose(
        angular_distribution,
        antineutrino_distribution,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        print(
            "  Fig. 12 branch plot skipped: neutrinos and antineutrinos do not "
            "share one factorized angular distribution"
        )
        return None
    if config.multienergy:
        return plot_multi_energy_angular_branches(output_dir, config, v, dv)

    angular_moments = np.array(
        [
            dv * np.sum(angular_distribution * v**power)
            for power in range(3)
        ]
    )
    moment_matrix = np.array(
        [
            [angular_moments[0], angular_moments[1]],
            [-angular_moments[1], -angular_moments[2]],
        ]
    )
    angular_eigenvalues = np.sort(np.linalg.eigvals(moment_matrix).real)[::-1]
    lambda_plus = float(angular_eigenvalues[0])
    lambda_minus = float(angular_eigenvalues[1])
    lepton_asymmetry = number_neutrinos - number_antineutrinos
    lepton_current = config.mu * lepton_asymmetry * angular_moments[1]
    kprime_values = np.unique(
        np.concatenate(
            (np.linspace(config.kmin, config.kmax, config.num), np.array([0.0]))
        )
    )
    kprime_values.sort()

    labels = {"lambda_plus": lambda_plus, "lambda_minus": lambda_minus}
    orderings = {"NO": (1.0, "-"), "IO": (-1.0, "--")}
    curves: dict[tuple[str, str], np.ndarray] = {}
    omega_magnitude = abs(config.effective_omega)
    if len(v) == 2 and np.allclose(v, np.array([-1.0, 1.0])):
        curves = exact_two_beam_branch_curves(
            config,
            v,
            dv,
            rho,
            antirho,
            kprime_values,
            lambda_plus,
            lambda_minus,
            number_neutrinos,
            number_antineutrinos,
            lepton_asymmetry,
            lepton_current,
        )
    else:
        for ordering, (ordering_sign, _) in orderings.items():
            signed_omega = ordering_sign * omega_magnitude
            for branch_name, angular_eigenvalue in labels.items():
                seed = angular_branch_root_at_zero(
                    config,
                    number_neutrinos,
                    number_antineutrinos,
                    lepton_asymmetry,
                    angular_eigenvalue,
                    signed_omega,
                )
                if seed.imag <= 1.0e-8:
                    curves[(ordering, branch_name)] = np.full(
                        len(kprime_values),
                        np.nan + 1j * np.nan,
                        dtype=np.complex128,
                    )
                    continue
                curves[(ordering, branch_name)] = continue_angular_branch(
                    seed,
                    kprime_values,
                    signed_omega,
                    config,
                    v,
                    dv,
                    angular_distribution,
                    number_neutrinos,
                    number_antineutrinos,
                    lepton_asymmetry,
                )

    return render_angular_branches(
        output_dir,
        config,
        kprime_values,
        lambda_plus,
        lambda_minus,
        lepton_asymmetry,
        lepton_current,
        curves,
        "Single-energy angular-mode branches",
        "Fig. 12 single-energy branches",
    )


def plot_dominant_mode(
    output_dir: Path,
    config: LSAConfig,
    v: np.ndarray,
    vector: np.ndarray,
    spectrum: np.ndarray,
    root: complex,
    k_star: float,
) -> None:
    if config.multienergy:
        modes = config.signed_frequencies.size
        per_sector = modes // 2
        normalized = vector / max(
            float(np.max(np.abs(vector))), np.finfo(float).tiny
        )
        shaped = np.abs(normalized.reshape(modes, len(v)))
        fig, axes = plt.subplots(2, 2, figsize=(10, 7.4))
        sectors = (slice(0, per_sector), slice(per_sector, modes))
        colors = ("tab:blue", "tab:orange")
        slice_target_mev = 15.0
        for column, (sector, title, color) in enumerate(
            zip(sectors, (r"$\nu$", r"$\bar\nu$"), colors)
        ):
            ax = axes[0, column]
            energies = config.energies[sector]
            order = np.argsort(energies)
            image = ax.pcolormesh(
                energies[order],
                v,
                shaped[sector][order].T,
                shading="auto",
                cmap="magma",
                vmin=0.0,
                vmax=1.0,
            )
            ax.set(xlabel=r"Energy $E$ [MeV]", ylabel=r"$v_z$", title=title)
            ax.set_ylim(-1.0, 1.0)
            colorbar = fig.colorbar(image, ax=ax)
            colorbar.set_label(r"$|Q(E,v_z)|/|Q|_{\rm max}$")

            slice_index = int(np.argmin(np.abs(energies - slice_target_mev)))
            slice_energy = float(energies[slice_index])
            slice_ax = axes[1, column]
            slice_ax.plot(v, shaped[sector][slice_index], color=color, lw=1.8)
            slice_ax.set(
                xlabel=r"$v_z$",
                ylabel=r"$|Q(E,v_z)|/|Q|_{\rm max}$",
                xlim=(-1.0, 1.0),
                ylim=(0.0, 1.05),
                title=rf"{title} slice at $E={slice_energy:g}$ MeV",
            )
        fig.suptitle(rf"Dominant multi-energy mode at $k_*={k_star:.4g}\,\mu$")
        save_figure(fig, output_dir / "dominant_mode.pdf")
        return
    n = len(v)
    scale = max(float(np.max(np.abs(vector))), np.finfo(float).tiny)
    normalized = vector / scale
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    axes[0].plot(v, np.abs(normalized[:n]), color="tab:blue", label=r"$|Q_\nu|$")
    axes[0].plot(v, np.abs(normalized[n:]), color="tab:orange", label=r"$|Q_{\bar\nu}|$")
    axes[0].set(xlabel=r"$v_z$", ylabel="Normalized mode amplitude", xlim=(-1.0, 1.0))
    axes[0].legend()
    axes[1].scatter(spectrum.real, spectrum.imag, s=8, alpha=0.45, color="tab:gray")
    axes[1].scatter([root.real], [root.imag], s=45, color="tab:red", label="dominant root")
    axes[1].axhline(0.0, color="black", lw=0.7)
    axes[1].set(xlabel=r"Re$(\Omega)$ [$\mu$]", ylabel=r"Im$(\Omega)$ [$\mu$]")
    axes[1].legend()
    fig.suptitle(rf"Dominant mode at $k_*={k_star:.4g}\,\mu$")
    save_figure(fig, output_dir / "dominant_mode.pdf")


def physical_time_scale(
    omega_over_mu: float, reference_energy: float = 10.0
) -> float | None:
    if omega_over_mu == 0.0:
        return None
    physical_omega = PHYSICAL_OMEGA_KM_INV * 10.0 / reference_energy
    meters_per_mu_inverse = 1000.0 * abs(omega_over_mu) / physical_omega
    return meters_per_mu_inverse / C_METERS_PER_MICROSECOND


def add_physical_time_axis(ax: plt.Axes, scale: float) -> None:
    upper = ax.secondary_xaxis(
        "top", functions=(lambda t: t * scale, lambda t: t / scale)
    )
    upper.set_xlabel("Physical time [μs]")


def snapshot_number(path: Path) -> int:
    match = re.fullmatch(r"rho_(\d+)\.(?:dat|npy)", path.name)
    if not match:
        raise ValueError(f"Unexpected snapshot name: {path.name}")
    return int(match.group(1))


def find_snapshots(run_dir: Path) -> list[Path]:
    """Return snapshots by iteration, preferring compact NPY over legacy text."""
    by_iteration = {
        snapshot_number(path): path for path in run_dir.glob("rho_*.dat")
    }
    by_iteration.update(
        {snapshot_number(path): path for path in run_dir.glob("rho_*.npy")}
    )
    return [by_iteration[number] for number in sorted(by_iteration)]


def integrated_neutrino_coherence(
    run_dir: Path, config: LSAConfig
) -> tuple[np.ndarray, np.ndarray] | None:
    if config.multienergy:
        compact_path = run_dir / "multienergy_coherence.dat"
        if compact_path.exists():
            compact = np.atleast_2d(np.loadtxt(compact_path, comments="#"))
            if compact.shape[1] >= 2:
                return compact[:, 0] * config.dt, compact[:, 1]
    snapshots = find_snapshots(run_dir)
    if not snapshots:
        return None
    times: list[float] = []
    neutrino: list[float] = []
    # Average over the angular grid after integrating |rho_ex| over z.
    z_weight = config.dz / config.simulation_nvz
    for snapshot in snapshots:
        if snapshot.suffix == ".npy":
            data = np.atleast_2d(np.load(snapshot, mmap_mode="r"))[:, (4, 5)]
        else:
            data = np.atleast_2d(np.loadtxt(snapshot, usecols=(4, 5)))
        neutrino.append(float(np.sum(np.hypot(data[:, 0], data[:, 1])) * z_weight))
        times.append(snapshot_number(snapshot) * config.dt)
    return np.asarray(times), np.asarray(neutrino)


def integrated_growth_fit(
    time: np.ndarray, amplitude: np.ndarray
) -> tuple[float, float, tuple[float, float], np.ndarray]:
    """Fit the exponential region between 1% and 20% of the pre-saturation peak."""
    valid = np.flatnonzero(
        np.isfinite(time) & np.isfinite(amplitude) & (amplitude > 0.0)
    )
    if len(valid) < 2:
        raise ValueError("Integrated coherence has too few positive snapshots to fit")
    values = amplitude[valid]
    peak_position = int(np.argmax(values))
    prepeak = valid[: peak_position + 1]
    peak = float(np.max(amplitude[prepeak]))
    fit_indices = prepeak[
        (amplitude[prepeak] >= 0.01 * peak)
        & (amplitude[prepeak] <= 0.20 * peak)
    ]
    if len(fit_indices) < 3:
        half_peak_crossings = np.flatnonzero(amplitude[prepeak] >= 0.5 * peak)
        end_position = (
            int(half_peak_crossings[0])
            if len(half_peak_crossings)
            else len(prepeak) - 1
        )
        end_position = min(len(prepeak) - 1, max(2, end_position))
        fit_indices = prepeak[: end_position + 1]
    if len(fit_indices) < 2 or np.unique(time[fit_indices]).size < 2:
        raise ValueError("the saved snapshots do not resolve an exponential-growth interval")
    fitted_gamma, fitted_log_normalization = np.polyfit(
        time[fit_indices], np.log(amplitude[fit_indices]), 1
    )
    interval = (float(time[fit_indices[0]]), float(time[fit_indices[-1]]))
    return float(fitted_gamma), float(fitted_log_normalization), interval, fit_indices


def plot_comparison(
    output_dir: Path,
    run_dir: Path,
    config: LSAConfig,
    gamma: float,
) -> ComparisonResult | None:
    if abs(config.matter_slope) > NUMERICAL_ZERO:
        print(
            "  homogeneous exponential comparison skipped: a matter gradient "
            "sweeps k through the unstable band"
        )
        return None
    integrated = integrated_neutrino_coherence(run_dir, config)
    if integrated is None:
        print("  comparison skipped: no density snapshots are present")
        return None
    snapshot_times, neutrino = integrated
    try:
        fitted_gamma, _, fit_interval, fit_indices = integrated_growth_fit(
            snapshot_times, neutrino
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        print(f"  comparison skipped: {error}")
        return None
    percent_error = (
        100.0 * abs(fitted_gamma - gamma) / abs(gamma) if gamma != 0.0 else None
    )
    lsa_log_normalization = float(
        np.median(
            np.log(neutrino[fit_indices]) - gamma * snapshot_times[fit_indices]
        )
    )
    lsa_normalization = float(np.exp(lsa_log_normalization))
    fit_span = fit_interval[1] - fit_interval[0]
    plot_start = max(float(snapshot_times[0]), fit_interval[0] - fit_span)
    half_peak = 0.5 * float(np.max(neutrino))
    saturation = np.flatnonzero(neutrino >= half_peak)
    plot_end = (
        float(snapshot_times[saturation[0]])
        if len(saturation)
        else fit_interval[1] + fit_span
    )
    plot_end = max(plot_end, fit_interval[1])
    theory_time = np.linspace(plot_start, plot_end, 300)
    lsa_curve = np.exp(
        np.clip(lsa_log_normalization + gamma * theory_time, -745.0, 700.0)
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    data_label = (
        r"COSEnu $\int dE\,dv_z\,dz\,|\rho_{ex}|$"
        if config.multienergy
        else r"COSEnu $\langle\int dz\,|\rho_{ex}|\rangle_{v_z}$"
    )
    ax.semilogy(
        snapshot_times,
        neutrino,
        color="tab:green",
        label=data_label,
    )
    ax.semilogy(
        theory_time,
        lsa_curve,
        color="tab:red",
        ls="--",
        label=rf"LSA $\propto e^{{\mathrm{{Im}}(\Omega)t}}$",
    )
    ax.set(xlabel=r"Time [$\mu^{-1}$]", ylabel="Off-diagonal coherence")
    ax.set_xlim(left=0.0)
    error_label = f"{percent_error:.2f}%" if percent_error is not None else "undefined"
    ax.text(
        0.02,
        0.97,
        "Growth-rate comparison\n"
        + rf"$\gamma_{{\rm COSEnu}}={fitted_gamma:.5g}\,\mu$"
        + "\n"
        + rf"$\gamma_{{\rm LSA}}={gamma:.5g}\,\mu$"
        + "\n"
        + f"relative error = {error_label}",
        transform=ax.transAxes,
        va="top",
        fontsize=8.2,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )
    ax.legend(fontsize=8, loc="lower right")
    physical_scale = physical_time_scale(
        config.omega_over_mu, config.reference_energy
    )
    if physical_scale is not None:
        add_physical_time_axis(ax, physical_scale)
    save_figure(fig, output_dir / "coherence_comparison.pdf")
    return ComparisonResult(
        cosenu_gamma=fitted_gamma,
        error_percent=percent_error,
        fit_interval=fit_interval,
        lsa_normalization=lsa_normalization,
    )


def write_outputs(
    output_dir: Path,
    run_dir: Path,
    config: LSAConfig,
    v: np.ndarray,
    rho: np.ndarray,
    antirho: np.ndarray,
    k_values: np.ndarray,
    roots: np.ndarray,
    champion_index: int,
    champion_root: complex,
    champion_vector: np.ndarray,
    comparison: ComparisonResult | None,
    angular_branches: AngularBranchResult | None,
) -> None:
    growth = np.where(np.abs(roots.imag) < NUMERICAL_ZERO, 0.0, roots.imag)
    np.savetxt(
        output_dir / "growth_rates.dat",
        np.column_stack((k_values, roots.real, growth)),
        header="k_over_mu  Re_Omega_over_mu  Im_Omega_over_mu",
    )
    scale = max(float(np.max(np.abs(champion_vector))), np.finfo(float).tiny)
    mode = champion_vector / scale
    n = config.nvz
    if config.multienergy:
        reshaped = mode.reshape(config.signed_frequencies.size, n)
        mode_rows = []
        for energy_index in range(config.signed_frequencies.size):
            sector = 1 if config.signed_frequencies[energy_index] > 0.0 else -1
            for velocity_index, velocity in enumerate(v):
                value = reshaped[energy_index, velocity_index]
                mode_rows.append(
                    (
                        sector,
                        config.energies[energy_index],
                        config.signed_frequencies[energy_index],
                        velocity,
                        value.real,
                        value.imag,
                        abs(value),
                    )
                )
        np.savetxt(
            output_dir / "dominant_mode.dat",
            np.asarray(mode_rows),
            header="sector E_MeV omega_over_mu v_z Re_Q Im_Q abs_Q",
        )
    else:
        np.savetxt(
            output_dir / "dominant_mode.dat",
            np.column_stack(
                (v, mode[:n].real, mode[:n].imag, np.abs(mode[:n]),
                 mode[n:].real, mode[n:].imag, np.abs(mode[n:]))
            ),
            header="v_z  Re_Q_nu  Im_Q_nu  abs_Q_nu  Re_Q_antinu  Im_Q_antinu  abs_Q_antinu",
        )

    gamma = 0.0 if abs(champion_root.imag) < NUMERICAL_ZERO else float(champion_root.imag)
    lines = [
        "COSEnu linear stability analysis",
        "",
        f"source run: {run_dir}",
        f"LSA angular bins: {config.nvz}",
        f"LSA k samples: {config.num}",
        f"energy treatment: {'multi-energy' if config.multienergy else 'monoenergetic'}",
        f"k interval (mu): {config.kmin:.10g} to {config.kmax:.10g}",
        f"dominant k (mu): {k_values[champion_index]:.16g}",
        f"dominant Re(Omega) (mu): {champion_root.real:.16g}",
        f"dominant Im(Omega) (mu): {gamma:.16g}",
        f"e-folding time (mu^-1): {1.0 / gamma:.16g}" if gamma > 0.0 else "e-folding time: no growing mode",
        "",
        f"alpha: {config.alpha:.16g}",
        f"angular profile: {config.angular_profile}",
        f"neutrino width: {config.neutrino_width:.16g}",
        f"antineutrino width: {config.antineutrino_width:.16g}",
        "ELN angular crossing on LSA grid: "
        f"{'yes' if np.min(rho - antirho) < 0.0 < np.max(rho - antirho) else 'no'}",
        f"omega/MU from COSEnu: {config.omega_over_mu:.16g}",
        f"effective omega cos(2 theta) (mu): {config.effective_omega:.16g}",
        f"self-interaction strength (mu): {config.mu:.16g}",
        f"midpoint matter potential (mu): {config.matter_potential:.16g}",
        f"matter slope (mu^2): {config.matter_slope:.16g}",
        f"damping rate (mu): {config.damping_rate:.16g}",
        f"antineutrino collision ratio: {config.antineutrino_collision_ratio:.16g}",
        "antineutrino damping rate (mu): "
        f"{config.antineutrino_collision_ratio * config.damping_rate:.16g}",
    ]
    if abs(config.matter_slope) > NUMERICAL_ZERO:
        lines.extend(
            [
                "",
                "Matter-gradient interpretation",
                "A constant local matter potential shifts Re(Omega) but not Im(Omega).",
                "For lambda(z)=lambda0+m(z-zmin), the effective wavenumber follows k_eff=k-m*t.",
                "A single constant-rate COSEnu/LSA fit is therefore not applicable; see matter_gradient.pdf.",
            ]
        )
    if config.multienergy:
        lines.extend(
            [
                f"energy modes: {config.signed_frequencies.size}",
                f"energy coordinate: {config.energy_grid}",
                "Multi-energy matrix evaluated as a diagonal plus rank-2 self-interaction update.",
            ]
        )
    if comparison is not None:
        error_text = (
            f"{comparison.error_percent:.16g}"
            if comparison.error_percent is not None
            else "undefined"
        )
        lines.extend(
            [
                "",
                "COSEnu/LSA spatially integrated neutrino coherence comparison",
                f"fitted COSEnu growth rate (mu): {comparison.cosenu_gamma:.16g}",
                f"LSA predicted growth rate (mu): {gamma:.16g}",
                f"relative growth-rate error (percent): {error_text}",
                f"LSA exponential normalization: {comparison.lsa_normalization:.16g}",
                "COSEnu fit interval (mu^-1): "
                f"{comparison.fit_interval[0]:.16g} to "
                f"{comparison.fit_interval[1]:.16g}",
                "The COSEnu rate is a log-linear fit to the green curve over this interval.",
                "The LSA exponential slope is fixed to Im(Omega).",
            ]
        )
    if angular_branches is not None:
        branch_lines = [
                "",
                "Appendix-C "
                + ("multi-energy" if config.multienergy else "single-energy")
                + " angular branches",
                f"lambda_plus: {angular_branches.lambda_plus:.16g}",
                f"lambda_minus: {angular_branches.lambda_minus:.16g}",
                f"lepton-number asymmetry eta: {angular_branches.lepton_asymmetry:.16g}",
                f"lepton-number current Jz (mu): {angular_branches.lepton_current:.16g}",
                "shifted-wavenumber definition: kprime = k - Jz",
                f"NO branch peak Im(Omega) (mu): {angular_branches.no_peak_gamma:.16g}",
                f"NO branch peak kprime (mu): {angular_branches.no_peak_kprime:.16g}",
                f"IO branch peak Im(Omega) (mu): {angular_branches.io_peak_gamma:.16g}",
                f"IO branch peak kprime (mu): {angular_branches.io_peak_kprime:.16g}",
                "The LSA diagonal splitting uses omega cos(2 theta), matching COSEnu.",
            ]
        if config.multienergy:
            branch_lines.append(
                "Eq. (C6) uses the configured energy quadrature at kprime=0; "
                "each root is continued with the full energy-angle determinant."
            )
        else:
            branch_lines.append(
                "The single-energy spectrum contains delta-function modes at +omega and -omega."
            )
        lines.extend(branch_lines)
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(
    run_dir: Path,
    nvz: int,
    num: int,
    kmin: float | None = None,
    kmax: float | None = None,
) -> Path:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    config = load_config(run_dir, nvz, num, kmin, kmax)
    output_dir = run_dir / "lsa"
    output_dir.mkdir(exist_ok=True)
    for old_png in output_dir.glob("*.png"):
        old_png.unlink()

    print(f"Linear stability analysis: {run_dir}")
    v, dv, rho, antirho = angular_background(config)
    antineutrino_rate = (
        config.antineutrino_collision_ratio * config.damping_rate
    )
    has_angular_crossing = bool(np.min(rho - antirho) < 0.0 < np.max(rho - antirho))
    print(
        f"Collision rates: C={config.damping_rate:.8g} mu, "
        f"Cbar={antineutrino_rate:.8g} mu "
        f"(Cbar/C={config.antineutrino_collision_ratio:.8g})"
    )
    print(f"ELN angular crossing on LSA grid: {'yes' if has_angular_crossing else 'no'}")
    if config.multienergy:
        k_values, roots, champion_array, root, vector, spectrum = (
            scan_multi_energy_modes(config, v, dv)
        )
    else:
        k_values, roots, champion_array, root, vector, spectrum = scan_modes(
            config, v, dv, rho, antirho
        )
    champion_index = int(champion_array[0])
    k_star = float(k_values[champion_index])
    gamma = 0.0 if abs(root.imag) < NUMERICAL_ZERO else float(root.imag)
    print(f"Dominant mode: k={k_star:.8g} mu, Im(Omega)={gamma:.8g} mu")

    plot_background(output_dir, v, rho, antirho)
    plot_dispersion(output_dir, k_values, roots, champion_index)
    angular_branches = plot_single_energy_angular_branches(
        output_dir, config, v, dv, rho, antirho
    )
    plot_dominant_mode(output_dir, config, v, vector, spectrum, root, k_star)
    comparison = plot_comparison(output_dir, run_dir, config, gamma)
    write_outputs(
        output_dir, run_dir, config, v, rho, antirho, k_values, roots,
        champion_index, root, vector, comparison, angular_branches
    )
    if comparison is not None:
        error_text = (
            f"{comparison.error_percent:.2f}%"
            if comparison.error_percent is not None
            else "undefined"
        )
        print(
            f"Growth-rate comparison: COSEnu={comparison.cosenu_gamma:.8g} mu, "
            f"LSA={gamma:.8g} mu, relative error={error_text}"
        )
    print(f"LSA complete: {output_dir}")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_directory",
        type=Path,
        help="completed run, for example runs/laptop-20260903-191021",
    )
    parser.add_argument("--nvz", type=int, default=200, help="independent LSA angular bins")
    parser.add_argument("--num", type=int, default=20, help="number of LSA k samples")
    parser.add_argument("--kmin", type=float, help="lower LSA k bound in MU")
    parser.add_argument("--kmax", type=float, help="upper LSA k bound in MU")
    args = parser.parse_args()
    try:
        run_analysis(args.run_directory, args.nvz, args.num, args.kmin, args.kmax)
    except Exception as exc:
        print(f"Linear stability analysis failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
