#!/usr/bin/env python3
"""Compare completed Soumya matter-gradient runs in COSEnu notation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plotting_cosenu import (
    add_physical_distance_y_axis,
    add_physical_time_axis,
    find_snapshots,
    load_table,
    physical_scales,
    plain_number,
    read_key_values,
    save_figure,
    snapshot_number,
)


def run_diagnostics(run_dir: Path) -> dict[str, object]:
    config = read_key_values(run_dir / "job.config", ":")
    laptop = read_key_values(run_dir / "laptop.conf", "=")
    nz = int(config["nz"])
    nvz = int(config["nvz"])
    if config.get("angular_profile", "").lower() not in {
        "symmetric_two_beam", "soumya_two_beam"
    }:
        raise ValueError(f"{run_dir} is not a supported symmetric two-beam run")
    if nvz != 2:
        raise ValueError(f"{run_dir} has NVZ={nvz}, expected 2")

    dz = (float(config["z1"]) - float(config["z0"])) / nz
    dv = abs((float(config["v1"]) - float(config["v0"])) / nvz)
    dt = abs(float(config["CFL"]) * dz / float(config["v1"]))
    snapshots = find_snapshots(run_dir)
    if len(snapshots) < 2:
        raise ValueError(f"{run_dir} needs at least two snapshots")

    times = np.asarray([snapshot_number(path) * dt for path in snapshots])
    nu_asymmetry = np.empty(len(snapshots))
    antinu_asymmetry = np.empty(len(snapshots))
    # Soumya's global transverse diagnostic: the spatial mean of the sum of
    # the two beams' normalized off-diagonal magnitudes.  Since
    # 2|rho_ex|/Tr(rho) is the transverse polarization magnitude for one beam,
    # this is equivalently the average of that magnitude over the two beams.
    nu_coherence = np.empty(len(snapshots))
    antinu_coherence = np.empty(len(snapshots))
    # Magnitude of the spatially averaged transverse vector. This auxiliary
    # diagnostic is retained in the compact archive, but it is not the summed
    # local-magnitude Strans used in Soumya's supplied comparison figures.
    nu_coherent_transverse = np.empty(len(snapshots))
    antinu_coherent_transverse = np.empty(len(snapshots))
    antinu_tz = np.empty((len(snapshots), nz))
    antinu_plus_tz = np.empty((len(snapshots), nz))
    z_reference: np.ndarray | None = None

    for index, snapshot in enumerate(snapshots):
        data = load_table(snapshot, usecols=range(10))
        if data.shape != (nz * nvz, 10):
            raise ValueError(
                f"{snapshot} has shape {data.shape}; expected ({nz * nvz}, 10)"
            )
        shaped = data.reshape(nz, nvz, 10)
        z = shaped[:, 0, 0]
        velocities = shaped[0, :, 1]
        plus = int(np.argmax(velocities))
        if z_reference is None:
            z_reference = z.copy()

        nu_trace_plus = shaped[:, plus, 2] + shaped[:, plus, 3]
        antinu_trace_plus = shaped[:, plus, 6] + shaped[:, plus, 7]
        nu_trace_all = shaped[:, :, 2] + shaped[:, :, 3]
        antinu_trace_all = shaped[:, :, 6] + shaped[:, :, 7]
        nu_asymmetry[index] = np.mean(
            np.divide(
                shaped[:, plus, 2] - shaped[:, plus, 3],
                nu_trace_plus,
                out=np.zeros_like(nu_trace_plus),
                where=np.abs(nu_trace_plus) > np.finfo(float).tiny,
            )
        )
        antinu_asymmetry[index] = np.mean(
            np.divide(
                shaped[:, plus, 6] - shaped[:, plus, 7],
                antinu_trace_plus,
                out=np.zeros_like(antinu_trace_plus),
                where=np.abs(antinu_trace_plus) > np.finfo(float).tiny,
            )
        )
        nu_perp_all = np.divide(
            np.hypot(shaped[:, :, 4], shaped[:, :, 5]),
            np.abs(nu_trace_all),
            out=np.zeros_like(nu_trace_all),
            where=np.abs(nu_trace_all) > np.finfo(float).tiny,
        )
        antinu_perp_all = np.divide(
            np.hypot(shaped[:, :, 8], shaped[:, :, 9]),
            np.abs(antinu_trace_all),
            out=np.zeros_like(antinu_trace_all),
            where=np.abs(antinu_trace_all) > np.finfo(float).tiny,
        )
        nu_coherence[index] = np.mean(np.sum(nu_perp_all, axis=1))
        antinu_coherence[index] = np.mean(np.sum(antinu_perp_all, axis=1))
        nu_offdiag_normalized = np.divide(
            shaped[:, plus, 4] + 1j * shaped[:, plus, 5],
            nu_trace_plus,
            out=np.zeros(nz, dtype=np.complex128),
            where=np.abs(nu_trace_plus) > np.finfo(float).tiny,
        )
        antinu_offdiag_normalized = np.divide(
            shaped[:, plus, 8] + 1j * shaped[:, plus, 9],
            antinu_trace_plus,
            out=np.zeros(nz, dtype=np.complex128),
            where=np.abs(antinu_trace_plus) > np.finfo(float).tiny,
        )
        nu_coherent_transverse[index] = 2.0 * abs(
            np.mean(nu_offdiag_normalized)
        )
        antinu_coherent_transverse[index] = 2.0 * abs(
            np.mean(antinu_offdiag_normalized)
        )
        antinu_plus_tz[index] = np.divide(
            2.0 * np.hypot(shaped[:, plus, 8], shaped[:, plus, 9]),
            np.abs(antinu_trace_plus),
            out=np.zeros_like(antinu_trace_plus),
            where=np.abs(antinu_trace_plus) > np.finfo(float).tiny,
        )

        antinu_trace = dv * np.sum(
            shaped[:, :, 6] + shaped[:, :, 7], axis=1
        )
        antinu_offdiag_magnitude = dv * np.sum(
            np.hypot(shaped[:, :, 8], shaped[:, :, 9]), axis=1
        )
        antinu_tz[index] = np.divide(
            2.0 * antinu_offdiag_magnitude,
            np.abs(antinu_trace),
            out=np.zeros_like(antinu_trace),
            where=np.abs(antinu_trace) > np.finfo(float).tiny,
        )

    assert z_reference is not None
    scales = physical_scales(config)
    survival = load_table(run_dir / "survival_probability.dat")
    conservation = load_table(run_dir / "conserved_quantities.dat")
    invariant = conservation[:, 4]
    invariant_drift_percent = 100.0 * np.max(
        np.abs(invariant / invariant[0] - 1.0)
    )
    matter_slope = float(config.get("matter_slope", laptop.get("MATTER_SLOPE", "0")))
    matter_offset = float(config.get("matter_offset", laptop.get("MATTER_OFFSET", "0")))
    matter_endpoints = np.asarray(
        [matter_offset, matter_offset + matter_slope * (float(config["z1"]) - float(config["z0"]))]
    )
    final_time = float(times[-1])
    swept_wavenumber = abs(matter_slope) * final_time
    swept_points_per_wavelength = (
        np.inf if swept_wavenumber == 0.0
        else 2.0 * np.pi / (swept_wavenumber * dz)
    )
    decay_time = (
        np.inf
        if matter_slope == 0.0
        else nz / (abs(matter_slope) * (float(config["z1"]) - float(config["z0"])))
    )
    return {
        "run_dir": run_dir,
        "nz": nz,
        "angular_profile": config.get("angular_profile", "unknown"),
        "slope": matter_slope,
        "matter_offset": matter_offset,
        "cfl": float(config["CFL"]),
        "max_matter_phase": float(np.max(np.abs(matter_endpoints)) * dt),
        "swept_points_per_wavelength": float(swept_points_per_wavelength),
        "decay_time": float(decay_time),
        "times": times,
        "z": z_reference,
        "nu_asymmetry": nu_asymmetry,
        "antinu_asymmetry": antinu_asymmetry,
        "nu_coherence": nu_coherence,
        "antinu_coherence": antinu_coherence,
        "nu_coherent_transverse": nu_coherent_transverse,
        "antinu_coherent_transverse": antinu_coherent_transverse,
        "antinu_tz": antinu_tz,
        "antinu_plus_tz": antinu_plus_tz,
        "scales": scales,
        "final_pee": float(survival[-1, 1]),
        "final_pbee": float(survival[-1, 2]),
        "final_mean_norm_error": float(conservation[-1, 2]),
        "final_antinu_mean_norm_error": float(conservation[-1, 3]),
        "final_max_norm_error": float(conservation[-1, 1]),
        "peak_max_norm_error": float(np.max(conservation[:, 1])),
        "max_invariant_drift_percent": float(invariant_drift_percent),
    }


def plot_scan(results: list[dict[str, object]], output_dir: Path) -> None:
    colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, len(results)))

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for result, color in zip(results, colors):
        slope = float(result["slope"])
        times = np.asarray(result["times"])
        ax.plot(
            times,
            result["nu_asymmetry"],
            color=color,
            lw=1.5,
            label=rf"$m={slope:g}\,\mu^2$",
        )
        ax.plot(
            times,
            result["antinu_asymmetry"],
            color=color,
            lw=1.3,
            ls="--",
        )
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(r"Time [$\mu^{-1}$]")
    ax.set_ylabel(
        r"$\left\langle(\rho_{ee}-\rho_{xx})/"
        r"(\rho_{ee}+\rho_{xx})\right\rangle_{z,\,v_z=+1}$"
    )
    slope_legend = ax.legend(loc="lower left", ncol=2, fontsize=8)
    ax.add_artist(slope_legend)
    ax.legend(
        handles=[
            Line2D([0], [0], color="black", lw=1.5, label=r"$\nu$"),
            Line2D([0], [0], color="black", lw=1.3, ls="--", label=r"$\bar\nu$"),
        ],
        loc="upper right",
    )
    scales = results[0]["scales"]
    if scales is not None:
        add_physical_time_axis(ax, scales[3])
    save_figure(fig, output_dir / "soumya_flavor_asymmetry_scan.pdf")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for result, color in zip(results, colors):
        slope = float(result["slope"])
        times = np.asarray(result["times"])
        ax.semilogy(
            times,
            np.maximum(
                np.asarray(result["nu_coherence"]), 1.0e-10
            ),
            color=color,
            lw=1.5,
            label=rf"$m={slope:g}\,\mu^2$",
        )
        ax.semilogy(
            times,
            np.maximum(
                np.asarray(result["antinu_coherence"]), 1.0e-10
            ),
            color=color,
            lw=1.3,
            ls="--",
        )
    ax.set_xlim(left=0.0)
    ax.set_ylim(1.0e-8, 1.0)
    ax.set_xlabel(r"Time [$\mu^{-1}$]")
    ax.set_ylabel(
        r"$L^{-1}\!\int dz\sum_{v_z=\pm1}"
        r"|\rho_{ex}^{v_z}|/\mathrm{Tr}(\rho^{v_z})$"
    )
    slope_legend = ax.legend(loc="lower right", ncol=2, fontsize=8)
    ax.add_artist(slope_legend)
    ax.legend(
        handles=[
            Line2D([0], [0], color="black", lw=1.5, label=r"$\nu$"),
            Line2D([0], [0], color="black", lw=1.3, ls="--", label=r"$\bar\nu$"),
        ],
        loc="upper left",
    )
    if scales is not None:
        add_physical_time_axis(ax, scales[3])
    save_figure(fig, output_dir / "soumya_coherence_scan.pdf")


def plot_tz_maps(results: list[dict[str, object]], output_dir: Path) -> None:
    selected = [result for result in results if float(result["slope"]) >= 0.019]
    if not selected:
        return
    specifications = (
        (
            "antinu_tz",
            "Velocity-integrated antineutrino coherence",
            r"$\log_{10}\!\left[2\int dv_z\,|\bar\rho_{ex}|/"
            r"\int dv_z(\bar\rho_{ee}+\bar\rho_{xx})\right]$",
            "soumya_antineutrino_coherence_tz_scan.pdf",
        ),
        (
            "antinu_plus_tz",
            r"Antineutrino $v_z=+1$ coherence (Soumya diagnostic)",
            r"$\log_{10}\!\left[2|\bar\rho_{ex,+}|/"
            r"(\bar\rho_{ee,+}+\bar\rho_{xx,+})\right]$",
            "soumya_antineutrino_vplus_coherence_tz_scan.pdf",
        ),
    )
    for field, title, colorbar_label, filename in specifications:
        fig, axes = plt.subplots(
            1,
            len(selected),
            figsize=(5.2 * len(selected), 5.6),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        axes = axes[0]
        image = None
        for ax, result in zip(axes, selected):
            times = np.asarray(result["times"])
            z = np.asarray(result["z"])
            values = np.log10(np.maximum(np.asarray(result[field]), 1.0e-8))
            image = ax.pcolormesh(
                times,
                z,
                values.T,
                shading="nearest",
                cmap="rainbow",
                vmin=-8.0,
                vmax=0.0,
                rasterized=True,
            )
            ax.set_xlim(left=0.0)
            ax.set_xlabel(r"Time [$\mu^{-1}$]")
            ax.text(
                0.5,
                0.96,
                rf"$m={float(result['slope']):g}\,\mu^2$",
                transform=ax.transAxes,
                ha="center",
                va="top",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
            positive_end = min(float(times.max()), float(z.max()))
            negative_end = min(float(times.max()), -float(z.min()))
            ax.plot([0.0, positive_end], [0.0, positive_end], "k:", lw=1.0)
            ax.plot([0.0, negative_end], [0.0, -negative_end], "k:", lw=1.0)
            scales = result["scales"]
            if scales is not None:
                add_physical_time_axis(ax, scales[3])
                add_physical_distance_y_axis(ax, scales[2])
        axes[0].set_ylabel(r"$z$ [$\mu^{-1}$]")
        assert image is not None
        fig.subplots_adjust(left=0.055, right=0.82, bottom=0.12, top=0.76, wspace=0.38)
        fig.colorbar(image, ax=axes, pad=0.09, label=colorbar_label)
        fig.suptitle(title, y=0.98)
        save_figure(fig, output_dir / filename, use_tight_layout=False)


def write_compact_data(results: list[dict[str, object]], output_dir: Path) -> None:
    """Keep all time series, but only archive maps that appear in the figure."""
    arrays: dict[str, np.ndarray] = {}
    for index, result in enumerate(results):
        prefix = f"run_{index:02d}"
        arrays[f"{prefix}_slope"] = np.asarray([result["slope"]])
        for key in (
            "times",
            "nu_asymmetry",
            "antinu_asymmetry",
            "nu_coherence",
            "antinu_coherence",
            "nu_coherent_transverse",
            "antinu_coherent_transverse",
        ):
            arrays[f"{prefix}_{key}"] = np.asarray(result[key])
        if float(result["slope"]) >= 0.019:
            arrays[f"{prefix}_z"] = np.asarray(result["z"])
            # Single precision is ample for a plotted normalized coherence and
            # halves the student-facing archive without changing the analysis.
            arrays[f"{prefix}_antinu_tz"] = np.asarray(
                result["antinu_tz"], dtype=np.float32
            )
            arrays[f"{prefix}_antinu_plus_tz"] = np.asarray(
                result["antinu_plus_tz"], dtype=np.float32
            )
    np.savez_compressed(output_dir / "soumya_matter_scan_data.npz", **arrays)


def write_summary(results: list[dict[str, object]], output_dir: Path) -> None:
    lines = [
        "COSEnu Soumya matter-gradient production scan",
        "===============================================",
        "",
        "All runs: Soumya per-beam normalization (g_nue=1, g_anue=0.85",
        "in each v_z=-1,+1 beam), omega/MU=0.006, inverted ordering, and",
        "lambda(z)=lambda0+m(z+500).  lambda0 and NZ are shown per row.",
        "The supplied legacy figures and all plotted production runs use lambda0=1.",
        "",
        (
            "m        lambda0  NZ      CFL       max(lambda)dt  pts/lambda_sweep  t_decay   "
            "final <Pee>  final <Pbee>  dPmax final  dPmax peak  <dP> final  "
            "<dPbar> final  max |Q/Q0-1| [%]"
        ),
    ]
    for result in results:
        lines.append(
            f"{float(result['slope']):<8g} "
            f"{float(result['matter_offset']):<8g} "
            f"{int(result['nz']):<7d} "
            f"{float(result['cfl']):<9.6g} "
            f"{float(result['max_matter_phase']):<14.6g} "
            f"{float(result['swept_points_per_wavelength']):<17.6g} "
            f"{float(result['decay_time']):<9.6g} "
            f"{float(result['final_pee']):<12.8f} "
            f"{float(result['final_pbee']):<13.8f} "
            f"{float(result['final_max_norm_error']):<12.5g} "
            f"{float(result['peak_max_norm_error']):<11.5g} "
            f"{float(result['final_mean_norm_error']):<11.5g} "
            f"{float(result['final_antinu_mean_norm_error']):<15.5g} "
            f"{float(result['max_invariant_drift_percent']):.5g}"
        )
    lines.extend(["", "Input run mapping:"])
    for result in results:
        lines.append(
            f"  m={float(result['slope']):g}, NZ={int(result['nz'])}: "
            f"{Path(result['run_dir']).name}"
        )
    lines.extend(
        [
            "",
            "Global observables used for the Soumya comparison:",
            "m        NZ      <P3>nu final  <P3>anu final  Strans nu final/peak  Strans anu final/peak",
        ]
    )
    for result in results:
        nu_transverse = np.asarray(result["nu_coherence"])
        antinu_transverse = np.asarray(result["antinu_coherence"])
        lines.append(
            f"{float(result['slope']):<8g} "
            f"{int(result['nz']):<7d} "
            f"{float(np.asarray(result['nu_asymmetry'])[-1]):<14.8g} "
            f"{float(np.asarray(result['antinu_asymmetry'])[-1]):<15.8g} "
            f"{float(nu_transverse[-1]):.7g}/{float(np.max(nu_transverse)):<10.7g} "
            f"{float(antinu_transverse[-1]):.7g}/{float(np.max(antinu_transverse)):.7g}"
        )
    lines.extend(
        [
            "",
            "Definitions used in the plots:",
            "  flavor asymmetry = <(rho_ee-rho_xx)/(rho_ee+rho_xx)>_z for v_z=+1",
            "  Soumya transverse coherence = <sum_(v=+/-1) |rho_ex,v|/Tr(rho_v)>_z",
            "  coherent-vector residual = 2|<rho_ex/(rho_ee+rho_xx)>_z| (archived data)",
            "  t-z map = 2 integral(dv_z)|rhobar_ex| / integral(dv_z)(rhobar_ee+rhobar_xx)",
            "  Soumya t-z map = 2|rhobar_ex,+|/(rhobar_ee,++rhobar_xx,+) for v_z=+1",
            "",
            "Interpretation:",
            "  pts/lambda_sweep = 2*pi/(|m|*t_final*dz); values below about 10--12",
            "  do not resolve the matter-driven k_eff=k-m*t sweep quantitatively.",
            "  t_decay~NZ/(|m|*L) estimates when an unresolved high-k sweep can",
            "  produce a false decreasing tail. Resolved strong suppression should",
            "  hold the off-diagonal amplitude nearly constant instead of erasing it.",
            "  <dP> is the primary global accuracy measure for these box-averaged plots.",
            "  dPmax is retained to expose narrow high-k error hotspots, but a large local",
            "  maximum alone does not invalidate a result with a controlled mean error.",
            "  Compare nonlinear amplitudes together with both mean-error and NZ trends.",
            "  Shortening light-cone packets with increasing m is the signature of",
            "  a matter gradient sweeping a perturbation through a finite unstable k band,",
            "  but an under-resolved sweep can spuriously erase those packets.",
            "",
            "Supplied-figure mapping:",
            "  soumya_coherence_scan.pdf corresponds to SlowmodesStrans...NEW.pdf",
            "  (theta=1e-5, lambda0=1) using the spatial mean of the local magnitude.",
            "  soumya_flavor_asymmetry_scan.pdf is the rho-language counterpart of",
            "  SlowmodesSpara...pdf; that supplied PDF is labeled theta=0, whereas",
            "  the present COSEnu scan uses theta=1e-5.",
            "",
        ]
    )
    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/soumya_matter_scan"),
    )
    args = parser.parse_args()
    try:
        results = [run_diagnostics(path.resolve()) for path in args.runs]
        results.sort(key=lambda item: float(item["slope"]))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plot_scan(results, args.output_dir)
        plot_tz_maps(results, args.output_dir)
        write_compact_data(results, args.output_dir)
        write_summary(results, args.output_dir)
    except Exception as exc:
        print(f"Soumya scan analysis failed: {exc}", file=sys.stderr)
        return 1
    print(f"Soumya scan analysis complete: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
