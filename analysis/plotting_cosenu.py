#!/usr/bin/env python3
"""Generate the standard student-facing COSEnu analysis plots for one run."""

from __future__ import annotations

import argparse
from copy import copy
from pathlib import Path
import re
import sys
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PHASE_MAGNITUDE_FLOOR = 1.0e-4
PHASE_CMAP = copy(matplotlib.colormaps["twilight_shifted"])
PHASE_CMAP.set_bad(color="0.85")


C_METERS_PER_MICROSECOND = 299.792458  # exact: c = 299792.458 km/s
ASSUMED_PHYSICAL_OMEGA_KM_INV = 0.63  # 10 MeV reference neutrino


def read_key_values(path: Path, separator: str) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or separator not in line:
            continue
        key, value = line.split(separator, 1)
        values[key.strip()] = value.split("#", 1)[0].strip().strip('"\'')
    return values


def load_table(path: Path, usecols=None) -> np.ndarray:
    if path.suffix == ".npy":
        data = np.atleast_2d(np.load(path, mmap_mode="r"))
        requested = None if usecols is None else np.atleast_1d(usecols).astype(int)
        needs_polarization = requested is None or bool(np.any(requested >= 10))
        if data.shape[1] == 10 and needs_polarization:
            # Compact snapshots retain only z, v_z, and the eight independent
            # density-matrix columns. Reconstruct the legacy polarization
            # columns so all analysis routines and old .dat files share one API.
            expanded = np.empty((data.shape[0], 16), dtype=np.float64)
            expanded[:, :10] = data
            neutrino_trace = data[:, 2] + data[:, 3]
            antineutrino_trace = data[:, 6] + data[:, 7]
            with np.errstate(divide="ignore", invalid="ignore"):
                expanded[:, 10] = np.divide(
                    2.0 * data[:, 4], neutrino_trace,
                    out=np.zeros_like(neutrino_trace), where=neutrino_trace != 0.0,
                )
                expanded[:, 11] = np.divide(
                    -2.0 * data[:, 5], neutrino_trace,
                    out=np.zeros_like(neutrino_trace), where=neutrino_trace != 0.0,
                )
                expanded[:, 12] = np.divide(
                    data[:, 2] - data[:, 3], neutrino_trace,
                    out=np.zeros_like(neutrino_trace), where=neutrino_trace != 0.0,
                )
                expanded[:, 13] = np.divide(
                    2.0 * data[:, 8], antineutrino_trace,
                    out=np.zeros_like(antineutrino_trace), where=antineutrino_trace != 0.0,
                )
                expanded[:, 14] = np.divide(
                    2.0 * data[:, 9], antineutrino_trace,
                    out=np.zeros_like(antineutrino_trace), where=antineutrino_trace != 0.0,
                )
                expanded[:, 15] = np.divide(
                    data[:, 6] - data[:, 7], antineutrino_trace,
                    out=np.zeros_like(antineutrino_trace), where=antineutrino_trace != 0.0,
                )
            data = expanded
        if usecols is not None:
            data = data[:, usecols]
    else:
        data = np.loadtxt(path, comments="#", usecols=usecols)
    return np.atleast_2d(data)


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


def numbered_diagnostic(path: Path, prefix: str) -> int:
    match = re.fullmatch(rf"{re.escape(prefix)}_(\d+)\.dat", path.name)
    if not match:
        raise ValueError(f"Unexpected {prefix} diagnostic name: {path.name}")
    return int(match.group(1))


def save_figure(
    fig: plt.Figure, destination: Path, *, use_tight_layout: bool = True
) -> None:
    # Apply minor ticks consistently, including physical-unit secondary axes
    # and colorbar axes wherever Matplotlib supports them.
    pending_axes = list(fig.axes)
    visited_axes: set[int] = set()
    while pending_axes:
        ax = pending_axes.pop()
        if id(ax) in visited_axes:
            continue
        visited_axes.add(id(ax))
        ax.minorticks_on()
        pending_axes.extend(ax.child_axes)
    if use_tight_layout:
        fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  created {destination.name}")


def plain_number(value: float, decimal_places: int = 6) -> str:
    """Format plot annotations without scientific notation."""
    rounded = round(float(value))
    if np.isclose(value, rounded, rtol=0.0, atol=10.0 ** (-decimal_places)):
        return str(int(rounded))
    return f"{float(value):.{decimal_places}f}".rstrip("0").rstrip(".")


def physical_scales(config: dict[str, str]):
    """Return scales using the configured omega, which is already omega/mu."""
    omega_over_mu = abs(float(config["omega"]))
    if omega_over_mu == 0.0:
        return None
    reference_energy = float(config.get("reference_energy", "10.0"))
    physical_omega_km_inv = (
        ASSUMED_PHYSICAL_OMEGA_KM_INV * 10.0 / reference_energy
    )
    mu_physical_km_inv = physical_omega_km_inv / omega_over_mu
    meters_per_mu_inverse = 1000.0 / mu_physical_km_inv
    microseconds_per_mu_inverse = meters_per_mu_inverse / C_METERS_PER_MICROSECOND
    return (
        omega_over_mu,
        mu_physical_km_inv,
        meters_per_mu_inverse,
        microseconds_per_mu_inverse,
    )


def add_physical_time_axis(ax: plt.Axes, microseconds_per_mu_inverse: float) -> None:
    upper = ax.secondary_xaxis(
        "top",
        functions=(
            lambda time_mu: time_mu * microseconds_per_mu_inverse,
            lambda time_microseconds: time_microseconds / microseconds_per_mu_inverse,
        ),
    )
    upper.set_xlabel("Physical time [μs]")


def add_physical_distance_axis(ax: plt.Axes, meters_per_mu_inverse: float) -> None:
    upper = ax.secondary_xaxis(
        "top",
        functions=(
            lambda z_mu: z_mu * meters_per_mu_inverse,
            lambda z_meters: z_meters / meters_per_mu_inverse,
        ),
    )
    upper.set_xlabel("Physical z [m]")


def add_physical_distance_y_axis(
    ax: plt.Axes, meters_per_mu_inverse: float
) -> None:
    right = ax.secondary_yaxis(
        "right",
        functions=(
            lambda z_mu: z_mu * meters_per_mu_inverse,
            lambda z_meters: z_meters / meters_per_mu_inverse,
        ),
    )
    right.set_ylabel("Physical z [m]")


def is_pure_vacuum_case(config: dict[str, str], laptop: dict[str, str]) -> bool:
    """Return true for the textbook pure-electron-flavor vacuum setup."""
    return (
        laptop.get("VACUUM_OSCILLATIONS", "true").lower() == "true"
        and laptop.get("MATTER_OSCILLATIONS", "false").lower() == "false"
        and laptop.get("COLLECTIVE_OSCILLATIONS", "true").lower() == "false"
        and float(laptop.get("DAMPING_RATE", "0")) == 0.0
        and float(config.get("perturbation_size", "0")) == 0.0
    )


def two_flavor_vacuum_probabilities(
    time: np.ndarray, config: dict[str, str]
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic probabilities for the Hamiltonian convention used by COSEnu."""
    theta = np.deg2rad(float(config["theta"]))
    omega = float(config["pmo"]) * float(config["omega"])
    conversion = np.sin(2.0 * theta) ** 2 * np.sin(0.5 * omega * time) ** 2
    return 1.0 - conversion, conversion


def plot_survival(
    run_dir: Path,
    plots_dir: Path,
    dt: float,
    microseconds_per_mu_inverse: Optional[float],
    config: dict[str, str],
    laptop: dict[str, str],
) -> tuple[float, float, Optional[float]]:
    data = load_table(run_dir / "survival_probability.dat")
    if data.shape[1] < 3:
        raise ValueError("survival_probability.dat must contain at least three columns")
    time = data[:, 0] * dt
    pee = data[:, 1]
    pbee = data[:, 2]

    # Two stacked panels, each with an approximately 5:4 frame proportion.
    fig, axes = plt.subplots(2, 1, figsize=(5, 8), sharex=True)
    axes[0].plot(time, pee, color="tab:blue", label=r"$\langle P_{ee}\rangle$")
    axes[0].plot(time, pbee, "--", color="tab:orange", label=r"$\langle \bar{P}_{ee}\rangle$")
    analytic_pee: Optional[np.ndarray] = None
    analytic_conversion: Optional[np.ndarray] = None
    vacuum_error: Optional[float] = None
    if is_pure_vacuum_case(config, laptop):
        analytic_pee, analytic_conversion = two_flavor_vacuum_probabilities(
            time, config
        )
        axes[0].plot(
            time,
            analytic_pee,
            "-.",
            color="tab:purple",
            lw=1.4,
            label="Two-flavor vacuum theory",
        )
        axes[0].text(
            0.5,
            0.025,
            r"$P_{ee}(t)=\bar P_{ee}(t)="
            r"1-\sin^2(2\theta)\sin^2(\omega t/2)$"
            "\n"
            r"$P_{ex}(t)=\bar P_{ex}(t)="
            r"\sin^2(2\theta)\sin^2(\omega t/2)$",
            transform=axes[0].transAxes,
            ha="center",
            va="bottom",
            fontsize=7.5,
            bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
        )
        vacuum_error = max(
            float(np.max(np.abs(pee - analytic_pee))),
            float(np.max(np.abs(pbee - analytic_pee))),
        )
        print(f"  vacuum-theory maximum probability error: {vacuum_error:.3e}")
    axes[0].axhline(0.5, color="black", ls=":", lw=1.0, label="Equipartition")
    axes[0].set_ylabel("Survival probability")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend()
    if microseconds_per_mu_inverse is not None:
        add_physical_time_axis(axes[0], microseconds_per_mu_inverse)

    axes[1].plot(time, 1.0 - pee, color="tab:green", label=r"$1-\langle P_{ee}\rangle$")
    axes[1].plot(
        time,
        1.0 - pbee,
        "--",
        color="tab:red",
        label=r"$1-\langle \bar{P}_{ee}\rangle$",
    )
    if analytic_conversion is not None:
        axes[1].plot(
            time,
            analytic_conversion,
            "-.",
            color="tab:purple",
            lw=1.4,
            label="Two-flavor vacuum theory",
        )
    axes[1].axhline(0.5, color="black", ls=":", lw=1.0, label="Equipartition")
    axes[1].set_xlabel(r"Time [$\mu^{-1}$]")
    axes[1].set_ylabel("Conversion probability")
    axes[1].set_xlim(left=0.0)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    save_figure(fig, plots_dir / "survival_probabilities.pdf")
    return float(pee[-1]), float(pbee[-1]), vacuum_error


def mass_projection_from_snapshots(
    run_dir: Path, config: dict[str, str], dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the multi-energy mass-basis invariant from rho snapshots."""
    snapshots = find_snapshots(run_dir)
    if not snapshots:
        raise FileNotFoundError("No rho snapshots are available for conservation")
    nz = int(config["nz"])
    nvz = int(config["nvz"])
    dz = (float(config["z1"]) - float(config["z0"])) / nz
    dv = (float(config["v1"]) - float(config["v0"])) / nvz
    theta = np.deg2rad(float(config["theta"]))
    projections: list[float] = []
    times: list[float] = []
    for snapshot in snapshots:
        data = load_table(snapshot, usecols=tuple(range(2, 10)))
        # The saved antineutrino columns use the physical-antineutrino
        # convention. Convert them back to the flavor-isospin sum.
        px = 2.0 * (data[:, 2] - data[:, 6])
        pz = data[:, 0] - data[:, 1] - data[:, 4] + data[:, 5]
        projection = np.sum(np.sin(2.0 * theta) * px - np.cos(2.0 * theta) * pz)
        projections.append(float(projection * dz * dv))
        times.append(snapshot_number(snapshot) * dt)
    return np.asarray(times), np.asarray(projections)


def plot_conservation(
    run_dir: Path,
    plots_dir: Path,
    dt: float,
    microseconds_per_mu_inverse: Optional[float],
    config: dict[str, str],
) -> None:
    path = run_dir / "conserved_quantities.dat"
    if not path.exists():
        return
    data = load_table(path)
    if data.shape[1] < 5:
        return
    time = data[:, 0] * dt
    multienergy = config.get("multienergy", "0") == "1"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    floor = np.finfo(float).tiny
    if multienergy:
        columns = (2, 3)
        labels = (r"$\Delta P$", r"$\Delta\bar P$")
    else:
        columns = (1, 2, 3)
        labels = (
            r"$\delta P_{max}$",
            r"$\langle\delta P\rangle$",
            r"$\langle\delta\bar P\rangle$",
        )
    for column, label in zip(columns, labels):
        axes[0].semilogy(time, np.maximum(np.abs(data[:, column]), floor), label=label)
    axes[0].set_xlabel(r"Time [$\mu^{-1}$]")
    axes[0].set_ylabel("Polarization-vector deviation")
    axes[0].set_xlim(left=0.0)
    axes[0].set_ylim(1.0e-6, 1.0e0)
    if multienergy:
        axes[0].set_title("Spectrum-weighted polarization-vector norm error")
    axes[0].legend()

    invariant_time = time
    invariant = data[:, 4]
    header = path.open("r", encoding="utf-8").readline()
    if multienergy and "Lmass" not in header:
        invariant_time, invariant = mass_projection_from_snapshots(
            run_dir, config, dt
        )
        print("  reconstructed mass-basis invariant from legacy rho snapshots")
    initial_conserved = float(invariant[0])
    normalizable = (
        np.isfinite(initial_conserved)
        and abs(initial_conserved) > np.finfo(float).tiny
    )
    if normalizable:
        percent_deviation = 100.0 * np.abs(invariant / initial_conserved - 1.0)
    else:
        # In the omega=0 fast limit H_vac vanishes identically, so Q=M0.Hvac
        # is zero by definition and a relative percentage is meaningless.
        # Retain the panel as a direct numerical check of absolute drift.
        percent_deviation = np.abs(invariant - initial_conserved)
    # This quantity is typically conserved close to machine precision. A
    # logarithmic percentage scale keeps those very small errors visible while
    # retaining 1% as the upper monitoring threshold.
    percent_floor = 1.0e-14
    axes[1].semilogy(
        invariant_time, np.maximum(percent_deviation, percent_floor)
    )
    axes[1].set_xlabel(r"Time [$\mu^{-1}$]")
    if multienergy and normalizable:
        axes[1].set_ylabel(r"$|L_m/L_{m,0}-1|$ [%]")
        axes[1].set_title("Mass-basis lepton-number conservation")
        axes[1].text(
            0.03,
            0.96,
            r"$L_m=\int dz\,dv_z\sum_\omega \mathbf{B}\cdot\mathbf{P}_{\omega,v_z}$"
            "\n"
            r"$=\int dz\,dv_z\sum_\omega[\sin(2\theta)P_x-\cos(2\theta)P_z]$",
            transform=axes[1].transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )
    elif normalizable:
        axes[1].set_ylabel(r"$|Q/Q_0-1|$ [%], $Q=M_0\cdot H_{vac}$")
        axes[1].text(
            0.03,
            0.96,
            r"$Q=\mathbf{M}_0\cdot\mathbf{H}_{\rm vac}$"
            r"$=-\sin(2\theta)M_{0x}+\cos(2\theta)M_{0z}$"
            "\n"
            r"$\mathbf{M}_0=\int dz\,dv_z\,[G\mathbf{P}-\bar G\,\bar{\mathbf{P}}]$",
            transform=axes[1].transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )
    else:
        axes[1].set_ylabel(r"$|Q-Q_0|$, $Q=M_0\cdot H_{vac}$")
        axes[1].set_title(r"$Q=0$ in the $\omega=0$ fast limit")
        axes[1].text(
            0.03,
            0.96,
            r"$Q=\mathbf{M}_0\cdot\mathbf{H}_{\rm vac}=0$",
            transform=axes[1].transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )
    axes[1].set_xlim(left=0.0)
    axes[1].set_ylim(percent_floor, 1.0)
    if microseconds_per_mu_inverse is not None:
        add_physical_time_axis(axes[0], microseconds_per_mu_inverse)
        add_physical_time_axis(axes[1], microseconds_per_mu_inverse)
    save_figure(fig, plots_dir / "conservation.pdf")

    if data.shape[1] > 5:
        fig, ax = plt.subplots(figsize=(8, 5))
        for column in range(5, data.shape[1]):
            ax.semilogy(time, np.maximum(np.abs(data[:, column]), floor), label=f"M{column - 5}")
        ax.set_xlabel(r"Time [$\mu^{-1}$]")
        ax.set_ylabel("Angular moment magnitude")
        ax.set_xlim(left=0.0)
        ax.legend(ncol=2, fontsize=8)
        if microseconds_per_mu_inverse is not None:
            add_physical_time_axis(ax, microseconds_per_mu_inverse)
        save_figure(fig, plots_dir / "angular_moments.pdf")


def plot_snapshot(
    snapshot: Path,
    plots_dir: Path,
    nz: int,
    nvz: int,
    dt: float,
    label: str,
    meters_per_mu_inverse: Optional[float],
) -> None:
    data = load_table(snapshot)
    expected_rows = nz * nvz
    if data.shape[0] != expected_rows or data.shape[1] < 16:
        raise ValueError(
            f"{snapshot.name} has shape {data.shape}; expected ({expected_rows}, at least 16)"
        )

    shaped = data.reshape(nz, nvz, data.shape[1])
    z = shaped[:, 0, 0]
    velocity = shaped[0, :, 1]
    eln = shaped[:, :, 2] - shaped[:, :, 6] - shaped[:, :, 3] + shaped[:, :, 7]
    p3 = shaped[:, :, 12]
    extent = (float(z.min()), float(z.max()), float(velocity.min()), float(velocity.max()))
    iteration = snapshot_number(snapshot)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    eln_limit = float(np.nanmax(np.abs(eln)))
    if not np.isfinite(eln_limit) or eln_limit == 0.0:
        eln_limit = 1.0
    image = axes[0].imshow(
        eln.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="coolwarm",
        vmin=-eln_limit,
        vmax=eln_limit,
    )
    fig.colorbar(image, ax=axes[0], label=r"$\rho_{ee}-\bar\rho_{ee}-\rho_{xx}+\bar\rho_{xx}$")
    axes[0].set_xlabel(r"z [$\mu^{-1}$]")
    axes[0].set_ylabel(r"$v_z$")
    axes[0].set_title("Electron lepton number")

    image = axes[1].imshow(
        p3.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="viridis",
        vmin=-1.0,
        vmax=1.0,
    )
    fig.colorbar(image, ax=axes[1], label=r"$P_3$")
    axes[1].set_xlabel(r"z [$\mu^{-1}$]")
    axes[1].set_ylabel(r"$v_z$")
    axes[1].set_title("Flavor polarization")
    if meters_per_mu_inverse is not None:
        add_physical_distance_axis(axes[0], meters_per_mu_inverse)
        add_physical_distance_axis(axes[1], meters_per_mu_inverse)
    fig.suptitle(
        f"{label.capitalize()} snapshot: t = {plain_number(iteration * dt)}"
    )
    save_figure(fig, plots_dir / f"snapshot_{label}.pdf")


def plot_rhoex_position(
    snapshot: Path,
    plots_dir: Path,
    nz: int,
    nvz: int,
    dt: float,
    label: str,
    meters_per_mu_inverse: Optional[float],
    multienergy: bool,
) -> None:
    """Plot Re(rho_ex) and Im(rho_ex) along z in two velocity-resolved panels.

    Multi-energy rho snapshots already contain the quadrature-weighted energy
    integral from the solver. Use those values directly, without integrating
    over velocity or applying a second energy weight.
    """
    data = load_table(snapshot, usecols=(0, 1, 4, 5))
    expected_rows = nz * nvz
    if data.shape != (expected_rows, 4):
        raise ValueError(
            f"{snapshot.name} has shape {data.shape}; expected ({expected_rows}, 4)"
        )
    shaped = data.reshape(nz, nvz, 4)
    z = shaped[:, 0, 0]
    velocity = shaped[0, :, 1]
    # Include both endpoints; small grids (including exact two-beam runs)
    # use all available bins without repeating a curve.
    selected_bins = np.rint(np.linspace(0, nvz - 1, min(6, nvz))).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)
    for ax, column, part, title in (
        (axes[0], 2, "Re", "Real part"),
        (axes[1], 3, "Im", "Imaginary part"),
    ):
        for index in selected_bins:
            ax.plot(
                z,
                shaped[:, index, column],
                lw=2.0,
                label=rf"$v_z={plain_number(velocity[index], 3)}$",
            )
        ax.set_xlabel(r"$z$ [$\mu^{-1}$]")
        ax.set_ylabel(
            rf"$\mathrm{{{part}}}\!\left[\int dE\,\rho_{{ex}}(z,v_z,E)\right]$"
            if multienergy else rf"$\mathrm{{{part}}}(\rho_{{ex}})(z,v_z)$"
        )
        ax.set_title(title)
        ax.margins(x=0)
        ax.legend(loc="upper right", fontsize=11, framealpha=0.95)
        if meters_per_mu_inverse is not None:
            add_physical_distance_axis(ax, meters_per_mu_inverse)
    fig.suptitle(
        rf"{label.capitalize()}: $t={plain_number(snapshot_number(snapshot) * dt)}"
        rf"\,\mu^{{-1}}$"
    )
    save_figure(fig, plots_dir / f"rhoex_z_{label}.pdf")


def plot_phase_snapshots(
    initial_snapshot: Path,
    final_snapshot: Path,
    plots_dir: Path,
    nz: int,
    nvz: int,
    dt: float,
    meters_per_mu_inverse: Optional[float],
) -> None:
    """Compare the initial and final phase of rho_ex on the z-v_z plane."""
    expected_rows = nz * nvz

    def phase_field(snapshot: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = load_table(snapshot, usecols=(0, 1, 4, 5))
        if data.shape != (expected_rows, 4):
            raise ValueError(
                f"{snapshot.name} has shape {data.shape}; expected ({expected_rows}, 4)"
            )
        shaped = data.reshape(nz, nvz, 4)
        z = shaped[:, 0, 0]
        velocity = shaped[0, :, 1]
        coherence = shaped[:, :, 2] + 1j * shaped[:, :, 3]
        phase_over_pi = np.ma.masked_where(
            np.abs(coherence) < PHASE_MAGNITUDE_FLOOR,
            np.angle(coherence) / np.pi,
        )
        return z, velocity, phase_over_pi

    initial_z, initial_velocity, initial_phase = phase_field(initial_snapshot)
    final_z, final_velocity, final_phase = phase_field(final_snapshot)
    if not np.allclose(initial_z, final_z) or not np.allclose(
        initial_velocity, final_velocity
    ):
        raise ValueError("The initial and final phase snapshots use different grids")

    extent = (
        float(initial_z.min()),
        float(initial_z.max()),
        float(initial_velocity.min()),
        float(initial_velocity.max()),
    )
    initial_time = snapshot_number(initial_snapshot) * dt
    final_time = snapshot_number(final_snapshot) * dt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for ax, phase, title in (
        (
            axes[0],
            initial_phase,
            rf"Initial: $\rho_{{ex}}=|\rho_{{ex}}|e^{{i\phi}}$ "
            rf"($t={plain_number(initial_time)}$)",
        ),
        (
            axes[1],
            final_phase,
            rf"Final: $\rho_{{ex}}=|\rho_{{ex}}|e^{{i\phi}}$ "
            rf"($t={plain_number(final_time)}$)",
        ),
    ):
        image = ax.imshow(
            phase.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=PHASE_CMAP,
            vmin=-1.0,
            vmax=1.0,
        )
        fig.colorbar(image, ax=ax, label=r"Phase $\phi/\pi$")
        ax.set_xlabel(r"$z$ [$\mu^{-1}$]")
        ax.set_ylabel(r"$v_z$")
        ax.set_title(title)
        if meters_per_mu_inverse is not None:
            add_physical_distance_axis(ax, meters_per_mu_inverse)
    save_figure(fig, plots_dir / "phase_zvz.pdf")


def plot_hamiltonian_scales(
    initial_snapshot: Path,
    final_snapshot: Path,
    plots_dir: Path,
    nz: int,
    nvz: int,
    dv: float,
    dt: float,
    config: dict[str, str],
    laptop: dict[str, str],
    meters_per_mu_inverse: Optional[float],
) -> None:
    """Compare velocity-averaged flavor-dependent Hamiltonian splittings."""
    expected_rows = nz * nvz
    collective_on = laptop.get("COLLECTIVE_OSCILLATIONS", "true").lower() == "true"
    matter_on = laptop.get("MATTER_OSCILLATIONS", "false").lower() == "true"
    vacuum_on = laptop.get("VACUUM_OSCILLATIONS", "true").lower() == "true"
    mu = float(config.get("mu", "1"))

    def self_interaction_scale(
        snapshot: Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        data = load_table(snapshot, usecols=tuple(range(10)))
        if data.shape != (expected_rows, 10):
            raise ValueError(
                f"{snapshot.name} has shape {data.shape}; expected ({expected_rows}, 10)"
            )
        shaped = data.reshape(nz, nvz, 10)
        z = shaped[:, 0, 0]
        velocity = shaped[0, :, 1]
        if not collective_on:
            return z, np.zeros_like(z)

        # The solver uses H_nunu(v) = mu * integral dv' (1-v*v')
        # [rho(v')-bar_rho(v')]. Its antineutrino convention makes the
        # off-diagonal combination rho_ex-bar_rho_ex*, hence Re differences
        # and Im sums below. For every target-v bin, discard the identity part
        # by taking the eigenvalue splitting, then form the normalized angular
        # integral <Delta H_nunu>_v = integral(dv Delta H)/(vmax-vmin).
        density_difference = (
            shaped[:, :, 2]
            - shaped[:, :, 3]
            - shaped[:, :, 6]
            + shaped[:, :, 7]
        )
        coherence_real = shaped[:, :, 4] - shaped[:, :, 8]
        coherence_imag = shaped[:, :, 5] + shaped[:, :, 9]
        moment0_diagonal = dv * np.sum(density_difference, axis=1)
        moment1_diagonal = dv * np.sum(
            velocity[None, :] * density_difference, axis=1
        )
        moment0_real = dv * np.sum(coherence_real, axis=1)
        moment1_real = dv * np.sum(velocity[None, :] * coherence_real, axis=1)
        moment0_imag = dv * np.sum(coherence_imag, axis=1)
        moment1_imag = dv * np.sum(velocity[None, :] * coherence_imag, axis=1)
        diagonal_difference = (
            moment0_diagonal[:, None]
            - moment1_diagonal[:, None] * velocity[None, :]
        )
        off_diagonal_real = (
            moment0_real[:, None] - moment1_real[:, None] * velocity[None, :]
        )
        off_diagonal_imag = (
            moment0_imag[:, None] - moment1_imag[:, None] * velocity[None, :]
        )
        splitting_by_velocity = abs(mu) * np.sqrt(
            diagonal_difference**2
            + 4.0 * (off_diagonal_real**2 + off_diagonal_imag**2)
        )
        return z, np.mean(splitting_by_velocity, axis=1)

    z, initial_self = self_interaction_scale(initial_snapshot)
    final_z, final_self = self_interaction_scale(final_snapshot)
    if not np.allclose(z, final_z):
        raise ValueError("The initial and final Hamiltonian snapshots use different grids")

    if matter_on:
        matter_offset = float(laptop.get("MATTER_OFFSET", "0"))
        matter_slope = float(laptop.get("MATTER_SLOPE", "0"))
        matter_scale = np.abs(
            matter_offset + matter_slope * (z - float(config["z0"]))
        )
    else:
        matter_scale = np.zeros_like(z)
    vacuum_scale = np.full_like(
        z,
        abs(float(config.get("pmo", "1")) * float(config["omega"]))
        if vacuum_on
        else 0.0,
    )

    initial_time = snapshot_number(initial_snapshot) * dt
    final_time = snapshot_number(final_snapshot) * dt
    self_suffix = "" if collective_on else " (disabled)"
    matter_suffix = "" if matter_on else " (disabled)"
    vacuum_suffix = "" if vacuum_on else " (disabled)"
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.axhline(0.0, color="0.55", lw=0.8, zorder=0)
    ax.plot(
        z,
        initial_self,
        color="tab:blue",
        label=rf"$\langle\Delta H_{{\nu\nu}}\rangle_v$, initial "
        rf"($t={plain_number(initial_time)}$){self_suffix}",
    )
    ax.plot(
        z,
        final_self,
        color="tab:orange",
        label=rf"$\langle\Delta H_{{\nu\nu}}\rangle_v$, final "
        rf"($t={plain_number(final_time)}$){self_suffix}",
    )
    ax.plot(
        z,
        matter_scale,
        "--",
        color="tab:green",
        label=rf"$\Delta H_{{\rm mat}}=|\lambda(z)|${matter_suffix}",
    )
    ax.plot(
        z,
        vacuum_scale,
        ":",
        color="black",
        label=rf"$\Delta H_{{\rm vac}}=|\omega|${vacuum_suffix}",
    )
    ax.set_xlabel(r"$z$ [$\mu^{-1}$]")
    ax.set_ylabel(r"Hamiltonian eigenvalue splitting [$\mu$]")
    ax.set_title("Velocity-averaged traceless Hamiltonian scales")
    ax.set_xlim(float(config["z0"]), float(config["z1"]))
    maximum_scale = max(
        float(np.max(initial_self)),
        float(np.max(final_self)),
        float(np.max(matter_scale)),
        float(np.max(vacuum_scale)),
        1.0e-12,
    )
    ax.set_ylim(-0.04 * maximum_scale, 1.08 * maximum_scale)
    splitting_definition = (
        r"$\Delta H_X=[(H_{X,ee}-H_{X,xx})^2+4|H_{X,ex}|^2]^{1/2}$"
    )
    ax.legend(title=splitting_definition, fontsize=6.5, title_fontsize=6.5)
    if meters_per_mu_inverse is not None:
        add_physical_distance_axis(ax, meters_per_mu_inverse)
    save_figure(fig, plots_dir / "hamiltonian_scales.pdf")


def plot_coherence(
    snapshots: list[Path],
    plots_dir: Path,
    dt: float,
    microseconds_per_mu_inverse: Optional[float],
) -> None:
    times: list[float] = []
    neutrino: list[float] = []
    antineutrino: list[float] = []
    for snapshot in snapshots:
        off_diagonal = load_table(snapshot, usecols=(4, 5, 8, 9))
        neutrino.append(float(np.mean(np.hypot(off_diagonal[:, 0], off_diagonal[:, 1]))))
        antineutrino.append(float(np.mean(np.hypot(off_diagonal[:, 2], off_diagonal[:, 3]))))
        times.append(snapshot_number(snapshot) * dt)

    floor = np.finfo(float).tiny
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(times, np.maximum(neutrino, floor), label=r"$\langle|\rho_{ex}|\rangle$")
    ax.semilogy(times, np.maximum(antineutrino, floor), "--", label=r"$\langle|\bar\rho_{ex}|\rangle$")
    ax.set_xlabel(r"Time [$\mu^{-1}$]")
    ax.set_ylabel("Mean off-diagonal coherence")
    ax.set_xlim(left=0.0)
    ax.legend()
    if microseconds_per_mu_inverse is not None:
        add_physical_time_axis(ax, microseconds_per_mu_inverse)
    save_figure(fig, plots_dir / "coherence.pdf")


def plot_matter_gradient(
    run_dir: Path,
    plots_dir: Path,
    snapshots: list[Path],
    nz: int,
    nvz: int,
    dv: float,
    dt: float,
    config: dict[str, str],
    laptop: dict[str, str],
    microseconds_per_mu_inverse: Optional[float],
) -> Optional[dict[str, float]]:
    """Compare a linear matter-gradient run with the swept-k LSA estimate."""
    matter_on = laptop.get("MATTER_OSCILLATIONS", "false").lower() == "true"
    slope = float(laptop.get("MATTER_SLOPE", "0")) if matter_on else 0.0
    if abs(slope) <= 1.0e-14:
        return None

    growth_path = run_dir / "lsa" / "growth_rates.dat"
    if not growth_path.exists():
        print("  matter-gradient plot skipped: run the LSA to provide Im(Omega(k))")
        return None
    growth_data = load_table(growth_path)
    if growth_data.shape[1] < 3:
        print("  matter-gradient plot skipped: growth_rates.dat is incomplete")
        return None
    order = np.argsort(growth_data[:, 0])
    lsa_k = growth_data[order, 0]
    growth = np.maximum(growth_data[order, 2], 0.0)

    times = np.asarray([snapshot_number(path) * dt for path in snapshots])
    actual = np.empty(len(snapshots))
    for index, snapshot in enumerate(snapshots):
        off_diagonal = load_table(snapshot, usecols=(4, 5))
        actual[index] = float(
            np.mean(np.hypot(off_diagonal[:, 0], off_diagonal[:, 1]))
        )

    initial = load_table(snapshots[0], usecols=(0, 4, 5))
    shaped = initial.reshape(nz, nvz, 3)
    z = shaped[:, 0, 0]
    dz = abs(float(config["z1"]) - float(config["z0"])) / nz
    integrated = dv * np.sum(
        shaped[:, :, 1] + 1j * shaped[:, :, 2], axis=1
    )
    box_k = 2.0 * np.pi * np.fft.fftfreq(nz, d=dz)
    initial_power = np.abs(np.fft.fft(integrated) / nz) ** 2
    positive = initial_power > np.max(initial_power) * 1.0e-24
    log_power = np.full(nz, -np.inf)
    log_power[positive] = np.log(initial_power[positive])

    cumulative = np.zeros((len(times), nz))
    for index in range(1, len(times)):
        midpoint_time = 0.5 * (times[index] + times[index - 1])
        effective_k = box_k - slope * midpoint_time
        local_growth = np.interp(
            effective_k, lsa_k, growth, left=0.0, right=0.0
        )
        cumulative[index] = (
            cumulative[index - 1]
            + local_growth * (times[index] - times[index - 1])
        )

    homogeneous_growth = np.interp(
        box_k, lsa_k, growth, left=0.0, right=0.0
    )
    homogeneous_cumulative = times[:, None] * homogeneous_growth[None, :]

    def rms_log_amplitude(efolds: np.ndarray) -> np.ndarray:
        terms = log_power[None, :] + 2.0 * efolds
        maximum = np.max(terms, axis=1)
        return 0.5 * (
            maximum
            + np.log(np.sum(np.exp(terms - maximum[:, None]), axis=1))
        )

    swept_log = rms_log_amplitude(cumulative)
    homogeneous_log = rms_log_amplitude(homogeneous_cumulative)
    floor = np.finfo(float).tiny
    normalization = np.log(max(actual[0], floor)) - swept_log[0]
    swept_amplitude = np.exp(np.clip(swept_log + normalization, -745.0, 700.0))
    homogeneous_amplitude = np.exp(
        np.clip(homogeneous_log + normalization, -745.0, 700.0)
    )

    final_contribution = log_power + 2.0 * cumulative[-1]
    dominant_index = int(np.argmax(final_contribution))
    dominant_initial_k = float(box_k[dominant_index])
    dominant_final_k = dominant_initial_k - slope * float(times[-1])
    available_efolds = float(cumulative[-1, dominant_index])
    homogeneous_efolds = float(
        homogeneous_growth[dominant_index] * times[-1]
    )

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 7.4))
    axes[0].plot(lsa_k, growth, color="tab:red", lw=1.6)
    axes[0].axvline(0.0, color="0.35", ls=":", lw=1.0)
    axes[0].scatter(
        [dominant_initial_k],
        [np.interp(dominant_initial_k, lsa_k, growth, left=0.0, right=0.0)],
        color="#ff9800",
        s=35,
        zorder=4,
        label=rf"seeded mode $k={dominant_initial_k:.3g}\,\mu$",
    )
    k_span = float(lsa_k.max() - lsa_k.min())
    visible_final_k = float(
        np.clip(
            dominant_final_k,
            float(lsa_k.min()) + 0.02 * k_span,
            float(lsa_k.max()) - 0.02 * k_span,
        )
    )
    axes[0].annotate(
        "",
        xy=(visible_final_k, 0.035 * max(float(np.max(growth)), 1.0e-12)),
        xytext=(dominant_initial_k, 0.035 * max(float(np.max(growth)), 1.0e-12)),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
    )
    axes[0].set(
        xlabel=r"Effective wavenumber $k_{\rm eff}$ [$\mu$]",
        ylabel=r"Growth rate Im$(\Omega)$ [$\mu$]",
        xlim=(float(lsa_k.min()), float(lsa_k.max())),
        title=rf"Matter gradient: $k_{{\rm eff}}=k-(d\lambda/dz)t$",
    )
    axes[0].legend(fontsize=8)

    axes[1].semilogy(
        times,
        np.maximum(actual, floor),
        color="tab:green",
        lw=1.6,
        label="COSEnu",
    )
    axes[1].semilogy(
        times,
        np.maximum(swept_amplitude, floor),
        "--",
        color="tab:red",
        lw=1.5,
        label="swept-k linear estimate",
    )
    axes[1].semilogy(
        times,
        np.maximum(homogeneous_amplitude, floor),
        ":",
        color="0.35",
        lw=1.5,
        label="homogeneous-matter linear estimate",
    )
    axes[1].set(
        xlabel=r"Time [$\mu^{-1}$]",
        ylabel=r"Mean neutrino coherence $\langle|\rho_{ex}|\rangle$",
        xlim=(0.0, float(times.max())),
    )
    upper_limit = max(float(np.max(actual)) * 3.0, actual[0] * 10.0)
    axes[1].set_ylim(max(actual[0] * 0.3, 1.0e-14), upper_limit)
    axes[1].legend(fontsize=8)
    axes[1].text(
        0.03,
        0.96,
        rf"$d\lambda/dz={slope:.4g}\,\mu^2$" + "\n"
        + rf"available linear growth $\simeq {available_efolds:.2f}$ e-folds",
        transform=axes[1].transAxes,
        va="top",
        fontsize=8.2,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )
    if microseconds_per_mu_inverse is not None:
        add_physical_time_axis(axes[1], microseconds_per_mu_inverse)
    save_figure(fig, plots_dir / "matter_gradient.pdf")

    return {
        "slope": slope,
        "dominant_initial_k": dominant_initial_k,
        "dominant_final_k": dominant_final_k,
        "available_efolds": available_efolds,
        "homogeneous_efolds": homogeneous_efolds,
        "predicted_final_ratio": float(
            swept_amplitude[-1] / max(homogeneous_amplitude[-1], floor)
        ),
    }


def plot_time_position_map(
    snapshots: list[Path],
    plots_dir: Path,
    nz: int,
    nvz: int,
    dv: float,
    dt: float,
    meters_per_mu_inverse: Optional[float],
    microseconds_per_mu_inverse: Optional[float],
) -> None:
    """Plot v_z-integrated survival and coherence versus time and position."""
    expected_rows = nz * nvz
    times = np.asarray([snapshot_number(path) * dt for path in snapshots])
    survival_rows: list[np.ndarray] = []
    coherence_rows: list[np.ndarray] = []
    phase_rows: list[np.ndarray] = []
    z_reference: Optional[np.ndarray] = None
    initial_density: Optional[np.ndarray] = None

    for snapshot in snapshots:
        # Columns are z, rho_ee, Re(rho_ex), and Im(rho_ex). Loading only the
        # required fields keeps this additional analysis modest in memory.
        data = load_table(snapshot, usecols=(0, 2, 4, 5))
        if data.shape != (expected_rows, 4):
            raise ValueError(
                f"{snapshot.name} has shape {data.shape}; expected ({expected_rows}, 4)"
            )
        shaped = data.reshape(nz, nvz, 4)
        z = shaped[:, 0, 0]
        if z_reference is None:
            z_reference = z.copy()
            initial_density = np.sum(shaped[:, :, 1], axis=1)
            if np.any(np.abs(initial_density) <= np.finfo(float).tiny):
                raise ValueError(
                    "Cannot normalize the local survival probability because "
                    "the initial angle-integrated density is zero"
                )
        elif not np.allclose(z, z_reference):
            raise ValueError(f"Position grid in {snapshot.name} differs from the initial snapshot")

        assert initial_density is not None
        # The common dv factor cancels from the survival-probability ratio.
        survival_rows.append(np.sum(shaped[:, :, 1], axis=1) / initial_density)
        integrated_re = dv * np.sum(shaped[:, :, 2], axis=1)
        integrated_im = dv * np.sum(shaped[:, :, 3], axis=1)
        coherence_rows.append(np.hypot(integrated_re, integrated_im))
        phase_rows.append(np.angle(integrated_re + 1j * integrated_im) / np.pi)

    assert z_reference is not None
    survival = np.stack(survival_rows)
    coherence = np.stack(coherence_rows)
    integrated_phase = np.stack(phase_rows)
    masked_integrated_phase = np.ma.masked_where(
        coherence < PHASE_MAGNITUDE_FLOOR, integrated_phase
    )
    log_coherence = np.log10(np.maximum(coherence, 1.0e-10))

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 4.8), sharex=True, sharey=True)
    survival_image = axes[0].pcolormesh(
        times,
        z_reference,
        survival.T,
        shading="nearest",
        cmap="rainbow",
        vmin=0.0,
        vmax=1.0,
        rasterized=True,
    )
    axes[0].set_xlabel(r"Time [$\mu^{-1}$]")
    axes[0].set_ylabel(r"$z$ [$\mu^{-1}$]")
    axes[0].set_xlim(left=0.0)
    axes[0].set_title("Neutrino survival probability")

    coherence_image = axes[1].pcolormesh(
        times,
        z_reference,
        log_coherence.T,
        shading="nearest",
        cmap="rainbow",
        vmin=-5.0,
        vmax=0.0,
        rasterized=True,
    )
    axes[1].set_xlabel(r"Time [$\mu^{-1}$]")
    axes[1].set_xlim(left=0.0)
    axes[1].set_title("Neutrino coherence")

    phase_image = axes[2].pcolormesh(
        times,
        z_reference,
        masked_integrated_phase.T,
        shading="nearest",
        cmap=PHASE_CMAP,
        vmin=-1.0,
        vmax=1.0,
        rasterized=True,
    )
    axes[2].set_xlabel(r"Time [$\mu^{-1}$]")
    axes[2].set_xlim(left=0.0)
    axes[2].set_title(r"Phase: $\rho_{ex}=|\rho_{ex}|e^{i\phi}$")

    # In the natural units of the lower axes c=1, so the two light-cone
    # branches obey z=+t and z=-t. Show them on both coherence diagnostics.
    positive_light_end = min(float(times.max()), float(z_reference.max()))
    negative_light_end = min(float(times.max()), -float(z_reference.min()))
    for ax in axes[1:]:
        if positive_light_end > 0.0:
            ax.plot(
                [0.0, positive_light_end],
                [0.0, positive_light_end],
                color="black",
                lw=1.5,
                ls=":",
                label="Light cone",
            )
        if negative_light_end > 0.0:
            ax.plot(
                [0.0, negative_light_end],
                [0.0, -negative_light_end],
                color="black",
                lw=1.5,
                ls=":",
            )
        if positive_light_end > 0.0 or negative_light_end > 0.0:
            ax.legend(loc="upper left", framealpha=0.9)

    if microseconds_per_mu_inverse is not None:
        for ax in axes:
            add_physical_time_axis(ax, microseconds_per_mu_inverse)
    if meters_per_mu_inverse is not None:
        for ax in axes:
            add_physical_distance_y_axis(ax, meters_per_mu_inverse)
    # Leave room between each secondary y axis and its colorbar.
    fig.colorbar(
        survival_image, ax=axes[0], pad=0.18, label=r"$P_{ee}(t,z)$"
    )
    fig.colorbar(
        coherence_image,
        ax=axes[1],
        pad=0.18,
        label=r"$\log_{10}|\int dv_z\,\rho_{ex}(t,z,v_z)|$",
    )
    fig.colorbar(
        phase_image, ax=axes[2], pad=0.18, label=r"Phase $\phi/\pi$"
    )
    save_figure(fig, plots_dir / "tzcolormap.pdf")


def plot_velocity_integrated_antineutrino_coherence(
    snapshots: list[Path],
    plots_dir: Path,
    nz: int,
    nvz: int,
    dv: float,
    dt: float,
    config: dict[str, str],
    meters_per_mu_inverse: Optional[float],
    microseconds_per_mu_inverse: Optional[float],
) -> None:
    """Plot the two-beam quantity shown in Soumya's legacy t-z maps.

    The old title identified v_z=+1 even though both z=+t and z=-t branches
    were present.  Its two visible branches correspond to integrating the
    transverse-coherence magnitude of both beams, not taking the magnitude
    after their complex phases have been summed.
    """
    if config.get("angular_profile", "").lower() not in {
        "symmetric_two_beam", "soumya_two_beam"
    }:
        return

    expected_rows = nz * nvz
    times = np.asarray([snapshot_number(path) * dt for path in snapshots])
    coherence_rows: list[np.ndarray] = []
    z_reference: Optional[np.ndarray] = None

    for snapshot in snapshots:
        # z, rhobar_ee, rhobar_xx, Re(rhobar_ex), Im(rhobar_ex)
        data = load_table(snapshot, usecols=(0, 6, 7, 8, 9))
        if data.shape != (expected_rows, 5):
            raise ValueError(
                f"{snapshot.name} has shape {data.shape}; expected ({expected_rows}, 5)"
            )
        shaped = data.reshape(nz, nvz, 5)
        z = shaped[:, 0, 0]
        if z_reference is None:
            z_reference = z.copy()
        elif not np.allclose(z, z_reference):
            raise ValueError(
                f"Position grid in {snapshot.name} differs from the initial snapshot"
            )

        trace = dv * np.sum(shaped[:, :, 1] + shaped[:, :, 2], axis=1)
        integrated_magnitude = dv * np.sum(
            np.hypot(shaped[:, :, 3], shaped[:, :, 4]), axis=1
        )
        normalized = np.divide(
            2.0 * integrated_magnitude,
            np.abs(trace),
            out=np.zeros_like(trace),
            where=np.abs(trace) > np.finfo(float).tiny,
        )
        coherence_rows.append(normalized)

    assert z_reference is not None
    log_coherence = np.log10(
        np.maximum(np.stack(coherence_rows), 1.0e-8)
    )

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    image = ax.pcolormesh(
        times,
        z_reference,
        log_coherence.T,
        shading="nearest",
        cmap="rainbow",
        vmin=-8.0,
        vmax=0.0,
        rasterized=True,
    )
    ax.set_xlabel(r"Time [$\mu^{-1}$]")
    ax.set_ylabel(r"$z$ [$\mu^{-1}$]")
    ax.set_xlim(left=0.0)
    slope = float(config.get("matter_slope", "0"))
    ax.set_title(
        "Velocity-integrated antineutrino coherence\n"
        + rf"$d\lambda/dz={slope:g}\,\mu^2$"
    )

    positive_light_end = min(float(times.max()), float(z_reference.max()))
    negative_light_end = min(float(times.max()), -float(z_reference.min()))
    if positive_light_end > 0.0:
        ax.plot(
            [0.0, positive_light_end],
            [0.0, positive_light_end],
            color="black",
            lw=1.0,
            ls=":",
            label="Light cone",
        )
    if negative_light_end > 0.0:
        ax.plot(
            [0.0, negative_light_end],
            [0.0, -negative_light_end],
            color="black",
            lw=1.0,
            ls=":",
        )
    if positive_light_end > 0.0 or negative_light_end > 0.0:
        ax.legend(loc="upper left", framealpha=0.9)

    if microseconds_per_mu_inverse is not None:
        add_physical_time_axis(ax, microseconds_per_mu_inverse)
    if meters_per_mu_inverse is not None:
        add_physical_distance_y_axis(ax, meters_per_mu_inverse)
    fig.colorbar(
        image,
        ax=ax,
        pad=0.18,
        label=(
            r"$\log_{10}\!\left["
            r"2\int dv_z\,|\bar\rho_{ex}|/"
            r"\int dv_z(\bar\rho_{ee}+\bar\rho_{xx})\right]$"
        ),
    )
    save_figure(fig, plots_dir / "antineutrino_coherence_tz.pdf")


def plot_power_spectrum(
    run_dir: Path,
    plots_dir: Path,
    snapshots: list[Path],
    nz: int,
    nvz: int,
    dv: float,
    dt: float,
    config: dict[str, str],
) -> None:
    """Plot Fig.-9-style spectra of the velocity-integrated density matrix."""
    if not snapshots:
        return

    expected_rows = nz * nvz
    k_max = float(config["k_max"])
    k_bins = int(config["k_bins"])
    k_values = np.linspace(-k_max, k_max, k_bins)

    coordinates = load_table(snapshots[0], usecols=(0,))
    if coordinates.shape[0] != expected_rows:
        raise ValueError(
            f"{snapshots[0].name} has {coordinates.shape[0]} rows; "
            f"expected {expected_rows} for the power spectrum"
        )
    z = coordinates[:, 0].reshape(nz, nvz)[:, 0]
    dz = (float(config["z1"]) - float(config["z0"])) / nz

    # Match the solver's existing spectral diagnostic: a continuous k grid
    # and a four-term Nuttall window.  The transform matrix is reused for all
    # snapshots and all three flavor components.
    grid_index = np.arange(nz, dtype=float)
    window = (
        0.355768
        - 0.487396 * np.cos(2.0 * np.pi * grid_index / (nz - 1.0))
        + 0.144232 * np.cos(4.0 * np.pi * grid_index / (nz - 1.0))
        - 0.012604 * np.cos(6.0 * np.pi * grid_index / (nz - 1.0))
    )
    transform = np.exp(-1j * np.outer(k_values, z)) * (window * dz)

    # Figure 9 of arXiv:2604.06404 uses a dense family of time-colored curves.
    # Forty-eight curves retain that appearance without obscuring individual
    # spectra or making high-snapshot runs unnecessarily slow to plot.
    selected_indices = np.unique(
        np.linspace(
            0,
            len(snapshots) - 1,
            min(48, len(snapshots)),
            dtype=int,
        )
    )
    selected_snapshots = [snapshots[index] for index in selected_indices]

    spectrum_cache: dict[Path, np.ndarray] = {}

    def density_power(snapshot: Path) -> np.ndarray:
        cached = spectrum_cache.get(snapshot)
        if cached is not None:
            return cached
        data = load_table(snapshot, usecols=(2, 3, 4, 5))
        if data.shape != (expected_rows, 4):
            raise ValueError(
                f"{snapshot.name} has shape {data.shape}; "
                f"expected ({expected_rows}, 4) for the power spectrum"
            )
        fields = data.reshape(nz, nvz, 4)
        density = np.empty((nz, 3), dtype=np.complex128)
        density[:, 0] = np.sum(fields[:, :, 0], axis=1) * dv
        density[:, 1] = np.sum(fields[:, :, 1], axis=1) * dv
        density[:, 2] = np.sum(
            fields[:, :, 2] + 1j * fields[:, :, 3], axis=1
        ) * dv
        power = np.abs(transform @ density) ** 2
        spectrum_cache[snapshot] = power
        return power

    times = np.asarray(
        [snapshot_number(snapshot) * dt for snapshot in selected_snapshots]
    )
    if np.isclose(times[0], times[-1]):
        time_norm = matplotlib.colors.Normalize(times[0], times[0] + 1.0)
    else:
        time_norm = matplotlib.colors.Normalize(times[0], times[-1])
    time_cmap = matplotlib.colormaps["winter"]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.2, 4.25),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    floor = np.finfo(float).tiny
    panel_components = (2, 0, 1)
    displayed_power_max = 0.0
    for snapshot, time in zip(selected_snapshots, times):
        power = density_power(snapshot)
        displayed_power_max = max(displayed_power_max, float(np.nanmax(power)))
        color = time_cmap(time_norm(time))
        for component, ax in zip(panel_components, axes):
            ax.semilogy(
                k_values,
                np.maximum(power[:, component], floor),
                color=color,
                lw=0.65,
                alpha=0.78,
            )

    # If the LSA recorded the fitted linear-growth interval, emphasize its
    # beginning and end exactly as the black and brown curves do in Fig. 9.
    highlighted: list[tuple[Path, str, str]] = []
    lsa_summary = run_dir / "lsa" / "summary.txt"
    if lsa_summary.exists():
        match = re.search(
            r"COSEnu fit interval \(mu\^-1\):\s*"
            r"([-+0-9.eE]+)\s+to\s+([-+0-9.eE]+)",
            lsa_summary.read_text(encoding="utf-8"),
        )
        if match:
            all_times = np.asarray(
                [snapshot_number(snapshot) * dt for snapshot in snapshots]
            )
            for requested_time, color, phase_name in (
                (float(match.group(1)), "black", "early linear"),
                (float(match.group(2)), "#8c3f16", "late linear"),
            ):
                nearest = snapshots[int(np.argmin(np.abs(all_times - requested_time)))]
                if all(nearest != item[0] for item in highlighted):
                    highlighted.append((nearest, color, phase_name))

    for snapshot, color, phase_name in highlighted:
        power = density_power(snapshot)
        displayed_power_max = max(displayed_power_max, float(np.nanmax(power)))
        time = snapshot_number(snapshot) * dt
        for component, ax in zip(panel_components, axes):
            label = (
                rf"{phase_name}: $t={plain_number(time)}\,\mu^{{-1}}$"
                if component == 2
                else None
            )
            ax.semilogy(
                k_values,
                np.maximum(power[:, component], floor),
                color=color,
                lw=1.35,
                label=label,
            )

    # Overlay Im(Omega(k)) on the coherence spectrum with an arbitrary
    # logarithmic normalization, following Fig. 9.  Anchor its peak to the
    # later highlighted COSEnu spectrum so that shape and peak location, not
    # an unrelated normalization, are what the eye compares.
    growth_path = run_dir / "lsa" / "growth_rates.dat"
    if growth_path.exists():
        growth_data = load_table(growth_path)
        if growth_data.shape[1] >= 3:
            lsa_k = growth_data[:, 0]
            growth = np.maximum(growth_data[:, 2], 0.0)
            finite = np.isfinite(lsa_k) & np.isfinite(growth)
            if np.any(finite) and float(np.max(growth[finite])) > 0.0:
                lsa_k = lsa_k[finite]
                growth = growth[finite]
                champion = int(np.argmax(growth))
                k_star = float(lsa_k[champion])
                gamma_star = float(growth[champion])
                reference_snapshot = (
                    highlighted[-1][0] if highlighted else selected_snapshots[-1]
                )
                reference_power = density_power(reference_snapshot)[:, 2]
                reference_at_peak = float(
                    reference_power[int(np.argmin(np.abs(k_values - k_star)))]
                )
                peak_power = float(np.clip(reference_at_peak, 1.0e-8, 1.0e1))
                normalized_growth = growth / gamma_star
                lsa_power = peak_power * 10.0 ** (
                    -3.0 * (1.0 - normalized_growth)
                )
                lsa_power[growth <= gamma_star * 1.0e-6] = np.nan
                axes[0].plot(
                    lsa_k,
                    lsa_power,
                    "--",
                    color="red",
                    lw=1.7,
                    label=r"LSA $\mathrm{Im}\,\Omega(k)$ (normalized)",
                )
                axes[0].scatter(
                    [k_star],
                    [peak_power],
                    s=38,
                    color="#ff9800",
                    edgecolors="none",
                    zorder=6,
                    label=rf"most unstable: $k_*={k_star:.3g}\,\mu$",
                )

    panel_titles = (r"$ex$", r"$ee$", r"$xx$")
    # Preserve the requested 1e-10 floor, but leave 25% headroom above the
    # largest displayed value so the homogeneous diagonal modes are visible.
    power_upper = max(1.0e2, 1.25 * displayed_power_max)
    for ax, title in zip(axes, panel_titles):
        ax.axvline(0.0, color="0.35", ls=":", lw=1.0)
        ax.set_xlim(-k_max, k_max)
        ax.set_ylim(1.0e-10, power_upper)
        ax.set_xlabel(r"Wavenumber $k$ [$\mu$]")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel(r"Power $|\widetilde{\rho}_{ab}(k)|^2$")
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(loc="lower left", fontsize=7.2)

    scalar_mappable = matplotlib.cm.ScalarMappable(
        norm=time_norm, cmap=time_cmap
    )
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(
        scalar_mappable,
        ax=axes,
        pad=0.02,
        shrink=0.93,
        aspect=28,
    )
    colorbar.set_label(r"Snapshot time $t$ [$\mu^{-1}$]")
    save_figure(
        fig,
        plots_dir / "power_spectrum.pdf",
        use_tight_layout=False,
    )


def angular_density(snapshot: Path, nz: int, nvz: int):
    """Return z-averaged rho_ee and anti-rho_ee as functions of v_z."""
    data = load_table(snapshot, usecols=(1, 2, 6))
    expected_rows = nz * nvz
    if data.shape != (expected_rows, 3):
        raise ValueError(
            f"{snapshot.name} has shape {data.shape}; expected ({expected_rows}, 3)"
        )
    shaped = data.reshape(nz, nvz, 3)
    velocity = shaped[0, :, 0]
    rho_ee = np.mean(shaped[:, :, 1], axis=0)
    anti_rho_ee = np.mean(shaped[:, :, 2], axis=0)
    return velocity, rho_ee, anti_rho_ee


def select_steady_snapshots(
    snapshots: list[Path], dt: float, t_steady: float
) -> list[Path]:
    times = np.asarray([snapshot_number(path) * dt for path in snapshots])
    # Snapshots are discrete, so begin with the sample nearest T_STEADY and
    # include every available sample after it.
    start = int(np.argmin(np.abs(times - t_steady)))
    return snapshots[start:]


def plot_angular_distributions(
    snapshots: list[Path],
    plots_dir: Path,
    nz: int,
    nvz: int,
    dt: float,
    t_steady: float,
) -> None:
    cache: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def distribution(snapshot: Path):
        if snapshot not in cache:
            cache[snapshot] = angular_density(snapshot, nz, nvz)
        return cache[snapshot]

    velocity, initial_rho, initial_anti_rho = distribution(snapshots[0])
    final_velocity, final_rho, final_anti_rho = distribution(snapshots[-1])
    if not np.allclose(velocity, final_velocity):
        raise ValueError("Velocity grids differ between the initial and final snapshots")

    steady_snapshots = select_steady_snapshots(snapshots, dt, t_steady)
    steady_rho: list[np.ndarray] = []
    steady_anti_rho: list[np.ndarray] = []
    for snapshot in steady_snapshots:
        snapshot_velocity, rho_ee, anti_rho_ee = distribution(snapshot)
        if not np.allclose(velocity, snapshot_velocity):
            raise ValueError(f"Velocity grid in {snapshot.name} differs from the initial snapshot")
        steady_rho.append(rho_ee)
        steady_anti_rho.append(anti_rho_ee)

    mean_rho = np.mean(np.stack(steady_rho), axis=0)
    mean_anti_rho = np.mean(np.stack(steady_anti_rho), axis=0)
    initial_time = snapshot_number(snapshots[0]) * dt
    final_time = snapshot_number(snapshots[-1]) * dt

    fig, ax = plt.subplots(figsize=(5.25, 4.25))
    ax.plot(
        velocity,
        initial_rho,
        ":",
        color="tab:blue",
        label=rf"$\langle\rho_{{ee}}\rangle_z$: initial "
        rf"($t={plain_number(initial_time)}$)",
    )
    ax.plot(
        velocity,
        initial_anti_rho,
        ":",
        color="tab:orange",
        label=rf"$\langle\bar{{\rho}}_{{ee}}\rangle_z$: initial "
        rf"($t={plain_number(initial_time)}$)",
    )
    ax.plot(
        velocity,
        final_rho,
        color="tab:blue",
        label=rf"$\langle\rho_{{ee}}\rangle_z$: final "
        rf"($t={plain_number(final_time)}$)",
    )
    ax.plot(
        velocity,
        final_anti_rho,
        color="tab:orange",
        label=rf"$\langle\bar{{\rho}}_{{ee}}\rangle_z$: final "
        rf"($t={plain_number(final_time)}$)",
    )
    ax.plot(
        velocity,
        mean_rho,
        linewidth=2.2,
        color="tab:green",
        label=rf"$\langle\rho_{{ee}}\rangle_{{z,t}}$: "
        rf"$t\geq {plain_number(t_steady)}$",
    )
    ax.plot(
        velocity,
        mean_anti_rho,
        linewidth=2.2,
        color="tab:purple",
        label=rf"$\langle\bar{{\rho}}_{{ee}}\rangle_{{z,t}}$: "
        rf"$t\geq {plain_number(t_steady)}$",
    )
    ax.set_xlabel(r"$v_z$")
    ax.set_ylabel(r"Electron-flavor angular density")
    ax.set_xlim(-1.0, 1.0)
    ax.set_title(
        f"time average: {len(steady_snapshots)} snapshot(s); "
        rf"$T_{{steady}}={plain_number(t_steady)}\,\mu^{{-1}}$"
    )
    ax.legend(ncol=1, fontsize=7.5)
    save_figure(fig, plots_dir / "angular_distributions.pdf")


def plot_energy_diagnostics(
    run_dir: Path,
    plots_dir: Path,
    dt: float,
    microseconds_per_mu_inverse: Optional[float],
) -> None:
    """Plot compact z/v-averaged diagnostics from a multi-energy run."""
    spectral_files = sorted(
        run_dir.glob("energy_[0-9]*.dat"),
        key=lambda path: numbered_diagnostic(path, "energy"),
    )
    angle_files = sorted(
        run_dir.glob("energy_angle_[0-9]*.dat"),
        key=lambda path: numbered_diagnostic(path, "energy_angle"),
    )
    if not spectral_files:
        return

    initial = load_table(spectral_files[0])
    final = load_table(spectral_files[-1])
    sector_specs = ((1, r"$\nu$"), (-1, r"$\bar\nu$"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, (sector, sector_label) in zip(axes, sector_specs):
        initial_rows = initial[initial[:, 0] == sector]
        final_rows = final[final[:, 0] == sector]
        order0 = np.argsort(initial_rows[:, 1])
        orderf = np.argsort(final_rows[:, 1])
        initial_rows = initial_rows[order0]
        final_rows = final_rows[orderf]
        ax.plot(initial_rows[:, 1], initial_rows[:, 5], "--", color="tab:blue", label=r"initial $e$")
        ax.plot(initial_rows[:, 1], initial_rows[:, 6], "--", color="tab:orange", label=r"initial $x$")
        ax.plot(final_rows[:, 1], final_rows[:, 7], color="tab:blue", label=r"final $e$")
        ax.plot(final_rows[:, 1], final_rows[:, 8], color="tab:orange", label=r"final $x$")
        ax.set_xlabel(r"Energy $E$ [MeV]")
        ax.set_ylabel(r"$\nu_e$-normalized number spectrum [MeV$^{-1}$]")
        ax.set_title(sector_label)
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.legend(fontsize=8, ncol=1)
    save_figure(fig, plots_dir / "energy_spectra.pdf")

    times = np.asarray(
        [numbered_diagnostic(path, "energy") * dt for path in spectral_files]
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True, sharey=True)
    for panel, (ax, (sector, sector_label)) in enumerate(zip(axes, sector_specs)):
        rows0 = initial[initial[:, 0] == sector]
        order = np.argsort(rows0[:, 1])
        energies = rows0[order, 1]
        survival_rows = []
        for path in spectral_files:
            rows = load_table(path)
            rows = rows[rows[:, 0] == sector]
            survival_rows.append(rows[np.argsort(rows[:, 1]), 3])
        survival = np.stack(survival_rows)
        image = ax.pcolormesh(
            times,
            energies,
            survival.T,
            shading="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            rasterized=True,
        )
        ax.set_xlabel(r"Time [$\mu^{-1}$]")
        if panel == 0:
            ax.set_ylabel(r"Energy $E$ [MeV]")
        ax.text(
            0.03,
            0.96,
            rf"{sector_label} survival probability",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
        ax.set_xlim(left=0.0)
        ax.set_ylim(float(energies.min()), float(energies.max()))
        colorbar = fig.colorbar(image, ax=ax, pad=0.04)
        colorbar.set_label(r"$P_{\rm sur}(E,t)$")
        if microseconds_per_mu_inverse is not None:
            add_physical_time_axis(ax, microseconds_per_mu_inverse)
    save_figure(fig, plots_dir / "energy_survival.pdf")

    # The lepton-number spectrum makes the energy crossings explicit and is a
    # useful normalization check for both energy- and omega-uniform grids.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, (sector, sector_label) in zip(axes, sector_specs):
        rows = final[final[:, 0] == sector]
        rows = rows[np.argsort(rows[:, 1])]
        initial_g = rows[:, 9]
        final_g = initial_g * (2.0 * rows[:, 3] - 1.0)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.plot(rows[:, 1], initial_g, "--", color="0.35", label="initial")
        ax.plot(rows[:, 1], final_g, color="tab:green", label="final")
        ax.set_xlabel(r"Energy $E$ [MeV]")
        ax.set_ylabel(r"E-XLN spectrum [MeV$^{-1}$]")
        ax.set_title(sector_label)
        ax.set_xlim(left=0.0)
        ax.legend()
    save_figure(fig, plots_dir / "lepton_spectrum.pdf")

    if angle_files:
        angle_data = load_table(angle_files[-1])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, (sector, sector_label) in zip(axes, sector_specs):
            rows = angle_data[angle_data[:, 0] == sector]
            energies = np.unique(rows[:, 1])
            velocities = np.unique(rows[:, 3])
            energies.sort()
            velocities.sort()
            p3 = np.empty((velocities.size, energies.size))
            for column, energy in enumerate(energies):
                energy_rows = rows[np.isclose(rows[:, 1], energy)]
                energy_rows = energy_rows[np.argsort(energy_rows[:, 3])]
                p3[:, column] = energy_rows[:, 4]
            image = ax.pcolormesh(
                energies,
                velocities,
                p3,
                shading="auto",
                cmap="coolwarm",
                vmin=-1.0,
                vmax=1.0,
                rasterized=True,
            )
            ax.set_xlabel(r"Energy $E$ [MeV]")
            ax.set_ylabel(r"$v_z$")
            ax.set_title(
                rf"{sector_label}: $\langle P_3\rangle_z$ at "
                rf"$t={plain_number(numbered_diagnostic(angle_files[-1], 'energy_angle') * dt)}"
                rf"\,\mu^{{-1}}$"
            )
            ax.set_xlim(float(energies.min()), float(energies.max()))
            ax.set_ylim(-1.0, 1.0)
            colorbar = fig.colorbar(image, ax=ax)
            colorbar.set_label(r"$\langle P_3\rangle_z$")
        save_figure(fig, plots_dir / "energy_angle.pdf")


def write_summary(
    run_dir: Path,
    plots_dir: Path,
    config: dict[str, str],
    laptop: dict[str, str],
    snapshots: list[Path],
    final_pee: float,
    final_pbee: float,
    vacuum_error: Optional[float],
    dt: float,
    t_steady: float,
    scales,
    matter_gradient: Optional[dict[str, float]],
) -> None:
    elapsed_path = run_dir / "time.txt"
    elapsed = elapsed_path.read_text(encoding="utf-8").strip() if elapsed_path.exists() else "unknown"
    box_length_mu_inverse = abs(float(config["z1"]) - float(config["z0"]))
    numerical_final_time_mu_inverse = max(int(config["N_ITER"]) - 1, 0) * dt
    requested_final_time_mu_inverse = float(
        laptop.get("END_TIME", numerical_final_time_mu_inverse)
    )
    lines = [
        "COSEnu run summary",
        "==================",
        f"scheme: {config.get('scheme', 'unknown')}",
        f"grid: {config.get('nz', '?')} x {config.get('nvz', '?')}",
        f"iterations: {config.get('N_ITER', '?')}",
        f"snapshots found: {len(snapshots)}",
        f"elapsed simulation time (seconds): {elapsed}",
        f"requested final time (MU^-1): {requested_final_time_mu_inverse:.8g}",
        f"numerical final time (MU^-1): {numerical_final_time_mu_inverse:.8g}",
        f"box length (MU^-1): {box_length_mu_inverse:.8g}",
        f"final <Pee>: {final_pee:.8g}",
        f"final <Pbee>: {final_pbee:.8g}",
    ]
    if config.get("multienergy", "0") == "1":
        lines.extend(
            [
                "",
                "Multi-energy calculation",
                f"energy coordinate: {config.get('energy_grid', 'unknown')}",
                f"energy bins per sector: {config.get('energy_bins', '?')}",
                f"total energy modes: {2 * int(config.get('energy_bins', '0'))}",
                f"reference energy (MeV): {config.get('reference_energy', '?')}",
                f"pinching alpha: {config.get('spectral_alpha', '?')}",
                "Full energy-dependent state is retained only in state.bin; rho snapshots are energy integrated.",
            ]
        )
        conservation_path = run_dir / "conserved_quantities.dat"
        if conservation_path.exists():
            conservation = load_table(conservation_path)
            header = conservation_path.open("r", encoding="utf-8").readline()
            if "Lmass" in header:
                invariant = conservation[:, 4]
            else:
                _, invariant = mass_projection_from_snapshots(run_dir, config, dt)
            invariant_error = 100.0 * np.abs(invariant / invariant[0] - 1.0)
            lines.extend(
                [
                    f"final spectrum-weighted neutrino norm error: {conservation[-1, 2]:.8g}",
                    f"final spectrum-weighted antineutrino norm error: {conservation[-1, 3]:.8g}",
                    "maximum mass-basis lepton-number drift (percent): "
                    f"{np.max(invariant_error):.8g}",
                ]
            )
    if vacuum_error is not None:
        vacuum_omega = abs(float(config["pmo"]) * float(config["omega"]))
        vacuum_theta_degrees = float(config["theta"])
        vacuum_amplitude = np.sin(2.0 * np.deg2rad(vacuum_theta_degrees)) ** 2
        vacuum_period = 2.0 * np.pi / vacuum_omega if vacuum_omega > 0.0 else np.inf
        lines.extend(
            [
                "",
                "Pure two-flavor vacuum comparison",
                "Pee(t) = 1 - sin^2(2 theta) sin^2(omega t / 2)",
                "Pex(t) = sin^2(2 theta) sin^2(omega t / 2)",
                f"configured |omega|/MU: {vacuum_omega:.8g}",
                f"configured mixing angle (degrees): {vacuum_theta_degrees:.8g}",
                f"predicted conversion amplitude sin^2(2 theta): {vacuum_amplitude:.8g}",
                f"predicted oscillation period (MU^-1): {vacuum_period:.8g}",
                f"maximum absolute COSEnu/theory probability error: {vacuum_error:.8g}",
            ]
        )
    if "k_max" in config and "k_bins" in config:
        lines.extend(
            [
                f"power-spectrum k range (MU): {-float(config['k_max']):.8g} to {float(config['k_max']):.8g}",
                f"power-spectrum k bins: {config['k_bins']}",
            ]
        )
    steady_snapshots = select_steady_snapshots(snapshots, dt, t_steady)
    steady_times = [snapshot_number(path) * dt for path in steady_snapshots]
    lines.extend(
        [
            f"requested angular-distribution T_STEADY (MU^-1): {t_steady:.8g}",
            f"angular time-average snapshots: {len(steady_snapshots)}",
            f"angular time-average sampled interval (MU^-1): {steady_times[0]:.8g} to {steady_times[-1]:.8g}",
        ]
    )
    if scales is not None:
        (
            omega_over_mu,
            mu_physical_km_inv,
            meters_per_mu_inverse,
            microseconds_per_mu_inverse,
        ) = scales
        lines.extend(
            [
                "",
                "Physical-unit conversion (assuming omega=0.63 km^-1 at E=10 MeV)",
                f"configured reference energy (MeV): {config.get('reference_energy', '10.0')}",
                f"configured OMEGA (= omega/MU): {omega_over_mu:.8g}",
                f"implied MU (km^-1): {mu_physical_km_inv:.8g}",
                f"one MU^-1 (meters): {meters_per_mu_inverse:.8g}",
                f"one MU^-1 (microseconds): {microseconds_per_mu_inverse:.8g}",
                f"requested final time as light-travel distance (meters): {requested_final_time_mu_inverse * meters_per_mu_inverse:.8g}",
                f"requested final time (microseconds): {requested_final_time_mu_inverse * microseconds_per_mu_inverse:.8g}",
                f"numerical final time as light-travel distance (meters): {numerical_final_time_mu_inverse * meters_per_mu_inverse:.8g}",
                f"numerical final time (microseconds): {numerical_final_time_mu_inverse * microseconds_per_mu_inverse:.8g}",
                f"box length (meters): {box_length_mu_inverse * meters_per_mu_inverse:.8g}",
            ]
        )
    for key in (
        "ALPHA",
        "NEUTRINO_WIDTH",
        "ANTINEUTRINO_WIDTH",
        "MATTER_SLOPE",
        "MATTER_OFFSET",
        "DAMPING_RATE",
        "ANTINEUTRINO_COLLISION_RATIO",
    ):
        if key in laptop:
            lines.append(f"{key.lower()}: {laptop[key]}")
    if matter_gradient is not None:
        lines.extend(
            [
                "",
                "Linear matter-gradient interpretation",
                "effective wavenumber: k_eff = k - (d lambda/dz) t",
                f"matter slope (MU^2): {matter_gradient['slope']:.8g}",
                f"dominant seeded k at t=0 (MU): {matter_gradient['dominant_initial_k']:.8g}",
                f"same mode k_eff at final time (MU): {matter_gradient['dominant_final_k']:.8g}",
                f"available swept-band growth (e-folds): {matter_gradient['available_efolds']:.8g}",
                f"homogeneous estimate for same mode (e-folds): {matter_gradient['homogeneous_efolds']:.8g}",
                "swept/homogeneous predicted final linear amplitude ratio: "
                f"{matter_gradient['predicted_final_ratio']:.8g}",
                "The linear estimates apply only before nonlinear saturation.",
            ]
        )
    (plots_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  created summary.txt")


def generate_plots(run_dir: Path) -> Path:
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    config = read_key_values(run_dir / "job.config", ":")
    laptop = read_key_values(run_dir / "laptop.conf", "=")
    required = ("nz", "nvz", "z0", "z1", "v0", "v1", "CFL", "omega")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"job.config is missing: {', '.join(missing)}")

    nz = int(config["nz"])
    nvz = int(config["nvz"])
    dz = (float(config["z1"]) - float(config["z0"])) / nz
    dv = abs((float(config["v1"]) - float(config["v0"])) / nvz)
    dt = abs(float(config["CFL"]) * dz / float(config["v1"]))
    scales = physical_scales(config)
    meters_per_mu_inverse = scales[2] if scales is not None else None
    microseconds_per_mu_inverse = scales[3] if scales is not None else None
    snapshots = find_snapshots(run_dir)
    if not snapshots:
        raise FileNotFoundError(f"No rho_*.npy or legacy rho_*.dat snapshots found in {run_dir}")
    final_snapshot_time = snapshot_number(snapshots[-1]) * dt
    requested_final_time = float(laptop.get("END_TIME", final_snapshot_time))
    t_steady = float(laptop.get("T_STEADY", min(400.0, requested_final_time)))

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    # PDF is the sole output format. Remove legacy raster figures when a run is
    # regenerated, and remove the former standalone phase map (its information
    # is now the third panel of tzcolormap.pdf).
    for legacy_png in plots_dir.glob("*.png"):
        legacy_png.unlink()
    (plots_dir / "phase_tzcolormap.pdf").unlink(missing_ok=True)
    print(f"Generating analysis in {plots_dir}")

    final_pee, final_pbee, vacuum_error = plot_survival(
        run_dir,
        plots_dir,
        dt,
        microseconds_per_mu_inverse,
        config,
        laptop,
    )
    plot_conservation(
        run_dir,
        plots_dir,
        dt,
        microseconds_per_mu_inverse,
        config,
    )
    plot_snapshot(
        snapshots[0], plots_dir, nz, nvz, dt, "initial", meters_per_mu_inverse
    )
    plot_snapshot(
        snapshots[-1], plots_dir, nz, nvz, dt, "final", meters_per_mu_inverse
    )
    for snapshot, label in ((snapshots[0], "initial"), (snapshots[-1], "final")):
        plot_rhoex_position(
            snapshot,
            plots_dir,
            nz,
            nvz,
            dt,
            label,
            meters_per_mu_inverse,
            config.get("multienergy", "0") == "1",
        )
    plot_phase_snapshots(
        snapshots[0],
        snapshots[-1],
        plots_dir,
        nz,
        nvz,
        dt,
        meters_per_mu_inverse,
    )
    plot_hamiltonian_scales(
        snapshots[0],
        snapshots[-1],
        plots_dir,
        nz,
        nvz,
        dv,
        dt,
        config,
        laptop,
        meters_per_mu_inverse,
    )
    plot_coherence(snapshots, plots_dir, dt, microseconds_per_mu_inverse)
    matter_gradient = plot_matter_gradient(
        run_dir,
        plots_dir,
        snapshots,
        nz,
        nvz,
        dv,
        dt,
        config,
        laptop,
        microseconds_per_mu_inverse,
    )
    plot_time_position_map(
        snapshots,
        plots_dir,
        nz,
        nvz,
        dv,
        dt,
        meters_per_mu_inverse,
        microseconds_per_mu_inverse,
    )
    plot_velocity_integrated_antineutrino_coherence(
        snapshots,
        plots_dir,
        nz,
        nvz,
        dv,
        dt,
        config,
        meters_per_mu_inverse,
        microseconds_per_mu_inverse,
    )
    plot_power_spectrum(
        run_dir,
        plots_dir,
        snapshots,
        nz,
        nvz,
        dv,
        dt,
        config,
    )
    plot_angular_distributions(
        snapshots, plots_dir, nz, nvz, dt, t_steady
    )
    if config.get("multienergy", "0") == "1":
        plot_energy_diagnostics(
            run_dir, plots_dir, dt, microseconds_per_mu_inverse
        )
    write_summary(
        run_dir,
        plots_dir,
        config,
        laptop,
        snapshots,
        final_pee,
        final_pbee,
        vacuum_error,
        dt,
        t_steady,
        scales,
        matter_gradient,
    )
    return plots_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_directory",
        type=Path,
        help="completed run, for example runs/laptop-20260903-191021",
    )
    args = parser.parse_args()
    try:
        plots_dir = generate_plots(args.run_directory)
    except Exception as exc:
        print(f"Plotting failed: {exc}", file=sys.stderr)
        return 1
    print(f"Analysis complete: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
