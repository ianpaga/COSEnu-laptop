#!/usr/bin/env bash

# Laptop launcher for COSEnu. It never modifies anything in src/.

set -u

# Resolve the physical path so a case-only directory rename is reflected even
# when the parent shell still carries the previous logical $PWD spelling.
ROOT_DIR="$(cd "$(dirname "$0")" && /bin/pwd -P)"
SRC_DIR="$ROOT_DIR/src"
BUILD_DIR="$ROOT_DIR/build/laptop"
DEFAULT_CONFIG="$ROOT_DIR/laptop.conf"
ANALYSIS_DIR="$ROOT_DIR/analysis"
LSA_DIR="$ROOT_DIR/lsa"
PLOT_ENV="$ROOT_DIR/build/plot-env"

say() {
    printf '%s\n' "$*"
}

fail() {
    say ""
    say "Error: $*"
    exit 1
}

usage() {
    cat <<'EOF'
COSEnu laptop launcher

Usage:
  ./cosenu.sh run [config]       Build and run COSEnu (the usual command)
  ./cosenu.sh setup              Install/check the C++ compiler
  ./cosenu.sh build [config]     Build without running
  ./cosenu.sh restart runs/laptop-20260903-191021
                                Continue an interrupted run
  ./cosenu.sh plot runs/laptop-20260903-191021
                                Generate/regenerate a run's plots
  ./cosenu.sh lsa runs/laptop-20260903-191021
                                Run/regenerate linear stability analysis
  ./cosenu.sh help               Show this help

The default configuration is laptop.conf. Results are placed in a new
timestamped directory (for example, runs/laptop-20260903-191021), so an
earlier run is never overwritten.
EOF
}

find_compiler() {
    local compiler_path
    if [ -n "${CXX:-}" ] && command -v "$CXX" >/dev/null 2>&1; then
        compiler_path="$(command -v "$CXX")"
        if "$compiler_path" --version >/dev/null 2>&1; then
            printf '%s\n' "$compiler_path"
            return 0
        fi
    fi

    if [ "$(uname -s)" = "Darwin" ]; then
        # Prefer Apple's native compiler over compilers injected into PATH by
        # unrelated scientific environments.
        if [ -x /usr/bin/clang++ ] && /usr/bin/clang++ --version >/dev/null 2>&1; then
            printf '%s\n' /usr/bin/clang++
            return 0
        fi
        for candidate in clang++ c++ g++ g++-15 g++-14 g++-13 g++-12 g++-11; do
            if command -v "$candidate" >/dev/null 2>&1; then
                compiler_path="$(command -v "$candidate")"
                if "$compiler_path" --version >/dev/null 2>&1; then
                    printf '%s\n' "$compiler_path"
                    return 0
                fi
            fi
        done
    else
        for candidate in g++ c++ clang++; do
            if command -v "$candidate" >/dev/null 2>&1; then
                compiler_path="$(command -v "$candidate")"
                if "$compiler_path" --version >/dev/null 2>&1; then
                    printf '%s\n' "$compiler_path"
                    return 0
                fi
            fi
        done
    fi
    return 1
}

as_admin() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        fail "Installing the compiler requires an administrator account (sudo)."
    fi
}

setup_compiler() {
    local compiler
    compiler="$(find_compiler 2>/dev/null || true)"
    if [ -n "$compiler" ]; then
        say "[OK] C++ compiler found: $compiler"
        say "[OK] The simulation itself needs no GSL, Slurm, MPI, or Python packages."
        return 0
    fi

    say "A C++ compiler is the only missing requirement."
    case "$(uname -s)" in
        Darwin)
            say "Opening Apple's Command Line Tools installer..."
            xcode-select --install >/dev/null 2>&1 || true
            say "Finish the installer, then run ./cosenu.sh run again."
            return 1
            ;;
        Linux)
            if command -v apt-get >/dev/null 2>&1; then
                say "Installing build-essential..."
                as_admin apt-get update || return 1
                as_admin apt-get install -y build-essential || return 1
            elif command -v dnf >/dev/null 2>&1; then
                say "Installing gcc-c++..."
                as_admin dnf install -y gcc-c++ || return 1
            elif command -v yum >/dev/null 2>&1; then
                say "Installing gcc-c++..."
                as_admin yum install -y gcc-c++ || return 1
            elif command -v pacman >/dev/null 2>&1; then
                say "Installing base-devel..."
                as_admin pacman -S --needed base-devel || return 1
            elif command -v zypper >/dev/null 2>&1; then
                say "Installing gcc-c++..."
                as_admin zypper install -y gcc-c++ || return 1
            elif command -v apk >/dev/null 2>&1; then
                say "Installing build-base..."
                as_admin apk add build-base || return 1
            else
                fail "Unsupported Linux package manager. Install a C++ compiler and rerun ./cosenu.sh run."
            fi
            ;;
        *)
            fail "Use macOS, Linux, or Windows Subsystem for Linux (WSL)."
            ;;
    esac

    compiler="$(find_compiler 2>/dev/null || true)"
    [ -n "$compiler" ] || fail "The compiler installation did not complete."
    say "[OK] C++ compiler installed: $compiler"
    say "[OK] The simulation itself needs no GSL, Slurm, MPI, or Python packages."
}

find_python() {
    local candidate python_path
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            python_path="$(command -v "$candidate")"
            if "$python_path" -c 'import sys; raise SystemExit(sys.version_info < (3, 9))' >/dev/null 2>&1; then
                printf '%s\n' "$python_path"
                return 0
            fi
        fi
    done
    return 1
}

install_python_support() {
    say "Installing Python plotting support..."
    case "$(uname -s)" in
        Darwin)
            if command -v brew >/dev/null 2>&1; then
                brew install python || return 1
            else
                fail "Python 3.9+ is needed for plot/LSA analysis. Install it from python.org, then rerun ./cosenu.sh run."
            fi
            ;;
        Linux)
            if command -v apt-get >/dev/null 2>&1; then
                as_admin apt-get update || return 1
                as_admin apt-get install -y python3 python3-venv || return 1
            elif command -v dnf >/dev/null 2>&1; then
                as_admin dnf install -y python3 || return 1
            elif command -v yum >/dev/null 2>&1; then
                as_admin yum install -y python3 || return 1
            elif command -v pacman >/dev/null 2>&1; then
                as_admin pacman -S --needed python || return 1
            elif command -v zypper >/dev/null 2>&1; then
                as_admin zypper install -y python3 || return 1
            elif command -v apk >/dev/null 2>&1; then
                as_admin apk add python3 py3-pip || return 1
            else
                fail "Install Python 3.9+ for analysis, or disable plots and LSA in laptop.conf."
            fi
            ;;
        *)
            fail "Use macOS, Linux, or Windows Subsystem for Linux (WSL)."
            ;;
    esac
}

ensure_plot_environment() {
    local python
    python="$(find_python 2>/dev/null || true)"
    if [ -z "$python" ]; then
        install_python_support || return 1
        python="$(find_python 2>/dev/null || true)"
    fi
    [ -n "$python" ] || fail "Python 3.9+ could not be installed; disable Python analyses in laptop.conf."

    mkdir -p "$ROOT_DIR/build"
    if [ ! -x "$PLOT_ENV/bin/python" ]; then
        say "Creating an isolated plotting environment..."
        if ! "$python" -m venv --system-site-packages "$PLOT_ENV" >/dev/null 2>&1; then
            install_python_support || return 1
            "$python" -m venv --system-site-packages "$PLOT_ENV" || \
                fail "Could not create the Python analysis environment."
        fi
    fi

    PLOT_PYTHON="$PLOT_ENV/bin/python"
    if ! "$PLOT_PYTHON" -c 'import numpy, matplotlib' >/dev/null 2>&1; then
        say "Installing NumPy and Matplotlib (first analysis run only)..."
        "$PLOT_PYTHON" -m pip install --prefer-binary -r "$ANALYSIS_DIR/requirements.txt" || \
            fail "Analysis-package installation failed; check the internet connection."
    fi
    say "[OK] Python analysis support is ready."
}

load_config() {
    local config_file="$1"
    [ -f "$config_file" ] || fail "Configuration file not found: $config_file"

    # Defaults make older or shortened copies of laptop.conf easier to use.
    SCHEME="fd"
    BOUNDARY="periodic"
    VACUUM_OSCILLATIONS="true"
    MATTER_OSCILLATIONS="false"
    COLLECTIVE_OSCILLATIONS="true"
    ADVECTION="true"
    Z_MIN="-50"
    Z_MAX="50"
    NZ="500"
    V_MIN="-1.0"
    V_MAX="1.0"
    NVZ="256"
    CFL="0.4"
    END_TIME="500"
    PERTURBATION_SIZE="1e-2"
    MASS_ORDERING="1.0"
    OMEGA="1.0e-2"
    MIXING_ANGLE="0.0"
    MU="1.0"
    K_MAX="40.0"
    K_BINS="1000"
    T_STEADY="400"
    ANALYSES="100"
    SNAPSHOTS="50"
    THREADS="auto"
    ALPHA="0.85"
    NEUTRINO_WIDTH="0.6"
    ANTINEUTRINO_WIDTH="0.6"
    ANGULAR_PROFILE="gaussian"
    RANDOM_SEEDS="false"
    SEED_WIDTH="3.0"
    MATTER_SLOPE="0.0"
    MATTER_OFFSET="0.0"
    EXACT_MATTER_ROTATION="false"
    DAMPING_RATE="0.0"
    ANTINEUTRINO_COLLISION_RATIO="0.0"
    MULTIENERGY="false"
    ENERGY_GRID="energy"
    ENERGY_BINS="12"
    ENERGY_MIN="2.0"
    ENERGY_MAX="60.0"
    OMEGA_GRID_MIN="1.0e-3"
    OMEGA_GRID_MAX="0.25"
    REFERENCE_ENERGY="10.0"
    SPECTRAL_ALPHA="2.3"
    MEAN_ENERGY_NUE="9.6"
    MEAN_ENERGY_ANTINUE="11.8"
    MEAN_ENERGY_NUX="16.0"
    LUMINOSITY_NUE="44.0"
    LUMINOSITY_ANTINUE="45.0"
    LUMINOSITY_NUX="20.0"
    PLOTS="yes"
    LINEAR_STABILITY_ANALYSIS="false"
    LSA_NVZ="200"
    LSA_NUM="20"
    LSA_K_MIN="-1.5"
    LSA_K_MAX="1.5"

    # laptop.conf is a deliberately small shell-style KEY=value file.
    # shellcheck disable=SC1090
    . "$config_file"

    SCHEME="$(printf '%s' "$SCHEME" | tr '[:upper:]' '[:lower:]')"
    BOUNDARY="$(printf '%s' "$BOUNDARY" | tr '[:upper:]' '[:lower:]')"
    VACUUM_OSCILLATIONS="$(printf '%s' "$VACUUM_OSCILLATIONS" | tr '[:upper:]' '[:lower:]')"
    MATTER_OSCILLATIONS="$(printf '%s' "$MATTER_OSCILLATIONS" | tr '[:upper:]' '[:lower:]')"
    COLLECTIVE_OSCILLATIONS="$(printf '%s' "$COLLECTIVE_OSCILLATIONS" | tr '[:upper:]' '[:lower:]')"
    ADVECTION="$(printf '%s' "$ADVECTION" | tr '[:upper:]' '[:lower:]')"
    RANDOM_SEEDS="$(printf '%s' "$RANDOM_SEEDS" | tr '[:upper:]' '[:lower:]')"
    ANGULAR_PROFILE="$(printf '%s' "$ANGULAR_PROFILE" | tr '[:upper:]' '[:lower:]')"
    PLOTS="$(printf '%s' "$PLOTS" | tr '[:upper:]' '[:lower:]')"
    LINEAR_STABILITY_ANALYSIS="$(printf '%s' "$LINEAR_STABILITY_ANALYSIS" | tr '[:upper:]' '[:lower:]')"
    MULTIENERGY="$(printf '%s' "$MULTIENERGY" | tr '[:upper:]' '[:lower:]')"
    EXACT_MATTER_ROTATION="$(printf '%s' "$EXACT_MATTER_ROTATION" | tr '[:upper:]' '[:lower:]')"
    ENERGY_GRID="$(printf '%s' "$ENERGY_GRID" | tr '[:upper:]' '[:lower:]')"

    case "$SCHEME" in fd|fv) ;; *) fail "SCHEME must be fd or fv." ;; esac
    case "$BOUNDARY" in periodic|open) ;; *) fail "BOUNDARY must be periodic or open." ;; esac
    for value in "$VACUUM_OSCILLATIONS" "$MATTER_OSCILLATIONS" "$COLLECTIVE_OSCILLATIONS" "$ADVECTION"; do
        case "$value" in true|false) ;; *) fail "Configuration switches must be true or false." ;; esac
    done
    case "$RANDOM_SEEDS" in true|false) ;; *) fail "RANDOM_SEEDS must be true or false." ;; esac
    case "$ANGULAR_PROFILE" in gaussian|two_beam|symmetric_two_beam|soumya_two_beam) ;; *) fail "ANGULAR_PROFILE must be gaussian, two_beam, symmetric_two_beam, or soumya_two_beam." ;; esac
    case "$PLOTS" in yes|no) ;; *) fail "PLOTS must be yes or no." ;; esac
    case "$LINEAR_STABILITY_ANALYSIS" in true|false) ;; *) fail "LINEAR_STABILITY_ANALYSIS must be true or false." ;; esac
    case "$MULTIENERGY" in true|false) ;; *) fail "MULTIENERGY must be true or false." ;; esac
    case "$EXACT_MATTER_ROTATION" in true|false) ;; *) fail "EXACT_MATTER_ROTATION must be true or false." ;; esac
    case "$ENERGY_GRID" in energy|omega) ;; *) fail "ENERGY_GRID must be energy or omega." ;; esac

    case "$NZ" in ''|*[!0-9]*) fail "NZ must be a positive integer." ;; esac
    case "$NVZ" in ''|*[!0-9]*) fail "NVZ must be a positive integer." ;; esac
    case "$ANALYSES" in ''|*[!0-9]*) fail "ANALYSES must be zero or a positive integer." ;; esac
    case "$SNAPSHOTS" in ''|*[!0-9]*) fail "SNAPSHOTS must be zero or a positive integer." ;; esac
    case "$K_BINS" in ''|*[!0-9]*) fail "K_BINS must be a positive integer." ;; esac
    case "$LSA_NVZ" in ''|*[!0-9]*) fail "LSA_NVZ must be a positive integer." ;; esac
    case "$LSA_NUM" in ''|*[!0-9]*) fail "LSA_NUM must be a positive integer." ;; esac
    case "$ENERGY_BINS" in ''|*[!0-9]*) fail "ENERGY_BINS must be a positive integer." ;; esac
    [ "$NZ" -ge 10 ] || fail "NZ must be at least 10."
    [ "$NVZ" -ge 2 ] || fail "NVZ must be at least 2."
    [ "$K_BINS" -ge 2 ] || fail "K_BINS must be at least 2."
    [ "$LSA_NVZ" -ge 2 ] || fail "LSA_NVZ must be at least 2."
    [ "$LSA_NUM" -ge 2 ] || fail "LSA_NUM must be at least 2."
    [ "$ENERGY_BINS" -ge 2 ] || fail "ENERGY_BINS must be at least 2."
    if { [ "$ANGULAR_PROFILE" = "two_beam" ] || [ "$ANGULAR_PROFILE" = "symmetric_two_beam" ] || [ "$ANGULAR_PROFILE" = "soumya_two_beam" ]; } \
        && { [ "$MULTIENERGY" = "true" ] || [ "$NVZ" -ne 2 ]; }; then
        fail "Two-beam ANGULAR_PROFILE choices require MULTIENERGY=false and NVZ=2."
    fi
    awk -v width="$SEED_WIDTH" 'BEGIN {
        number = "^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$";
        exit !(width ~ number && width > 0);
    }' || fail "SEED_WIDTH must be a positive number in MU^-1."
    if [ "$MULTIENERGY" = "true" ] && [ "$SCHEME" != "fd" ]; then
        fail "MULTIENERGY=true currently requires SCHEME=fd."
    fi
    if [ "$MULTIENERGY" = "true" ] && [ "$RANDOM_SEEDS" = "true" ]; then
        fail "MULTIENERGY=true currently uses the deterministic article seed; set RANDOM_SEEDS=false."
    fi
    awk -v value="$K_MAX" 'BEGIN {
        number = "^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$";
        exit !(value ~ number && value > 0);
    }' || fail "K_MAX must be a positive number."
    awk -v value="$T_STEADY" 'BEGIN {
        number = "^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$";
        exit !(value ~ number && value >= 0);
    }' || fail "T_STEADY must be zero or a positive number."
    awk -v lower="$LSA_K_MIN" -v upper="$LSA_K_MAX" 'BEGIN {
        number = "^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$";
        exit !(lower ~ number && upper ~ number && upper > lower);
    }' || fail "LSA_K_MIN and LSA_K_MAX must define an increasing numeric interval."
    awk -v damping="$DAMPING_RATE" -v ratio="$ANTINEUTRINO_COLLISION_RATIO" 'BEGIN {
        number = "^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$";
        exit !(damping ~ number && ratio ~ number && damping >= 0 && ratio >= 0);
    }' || fail "DAMPING_RATE and ANTINEUTRINO_COLLISION_RATIO must be non-negative numbers."
    awk -v emin="$ENERGY_MIN" -v emax="$ENERGY_MAX" \
        -v omin="$OMEGA_GRID_MIN" -v omax="$OMEGA_GRID_MAX" \
        -v eref="$REFERENCE_ENERGY" -v alpha="$SPECTRAL_ALPHA" \
        -v enue="$MEAN_ENERGY_NUE" -v eanue="$MEAN_ENERGY_ANTINUE" \
        -v enux="$MEAN_ENERGY_NUX" -v lnue="$LUMINOSITY_NUE" \
        -v lanue="$LUMINOSITY_ANTINUE" -v lnux="$LUMINOSITY_NUX" 'BEGIN {
        number = "^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$";
        exit !(emin ~ number && emax ~ number && omin ~ number && omax ~ number &&
               eref ~ number && alpha ~ number && enue ~ number && eanue ~ number &&
               enux ~ number && lnue ~ number && lanue ~ number && lnux ~ number &&
               emin > 0 && emax > emin && omin >= 0 && omax > omin && eref > 0 &&
               alpha > -1 && enue > 0 && eanue > 0 && enux > 0 &&
               lnue > 0 && lanue > 0 && lnux > 0);
    }' || fail "Check the multi-energy bounds, alpha, mean energies, and luminosities."
    if [ "$MULTIENERGY" = "true" ] && [ "$ENERGY_GRID" = "omega" ]; then
        awk -v omega="$OMEGA" 'BEGIN { exit !(omega != 0) }' || \
            fail "ENERGY_GRID=omega needs nonzero OMEGA to map frequency to energy."
    fi

    if [ "$SCHEME" = "fd" ]; then
        GHOST_ZONES=2
    else
        GHOST_ZONES=4
    fi

    read -r N_ITER ANAL_EVERY ESTIMATED_MB WORK_UNITS NYQUIST_K MIN_NZ_FOR_KMAX <<EOF
$(awk -v z0="$Z_MIN" -v z1="$Z_MAX" -v nz="$NZ" -v v1="$V_MAX" \
       -v cfl="$CFL" -v end_time="$END_TIME" -v analyses="$ANALYSES" \
       -v nvz="$NVZ" -v gz="$GHOST_ZONES" -v kmax="$K_MAX" \
       -v multi="$MULTIENERGY" -v ne="$ENERGY_BINS" 'BEGIN {
    dz = (z1-z0)/nz;
    if (nz < 1 || nvz < 1 || dz <= 0 || cfl <= 0 || v1 == 0 || end_time <= 0) exit 2;
    dt = cfl*dz/v1;
    if (dt < 0) dt = -dt;
    n = int(end_time/dt);
    if (analyses > 0) { a = int(n/analyses); if (a < 1) a = 1; }
    else a = n+1;
    if (multi == "true") {
        mb = ((nz+2*gz)*nvz*(2*ne)*4*4*8)/(1024*1024);
        work = n*nz*nvz*(2*ne);
    } else {
        mb = ((nz+2*gz)*nvz*560)/(1024*1024);
        work = n*nz*nvz;
    }
    pi = 3.141592653589793;
    nyquist = pi/dz;
    required_nz = int(kmax*(z1-z0)/pi);
    if (required_nz*pi < kmax*(z1-z0)) required_nz++;
    printf "%d %d %.1f %.0f %.8g %d\n", n, a, mb, work, nyquist, required_nz;
}')
EOF
    [ -n "${N_ITER:-}" ] || fail "Check Z_MIN, Z_MAX, CFL, V_MAX, and END_TIME in $config_file."
    [ "$N_ITER" -ge 10 ] || fail "This setup gives only $N_ITER iterations; increase END_TIME to give at least 10."
}

write_presets() {
    local presets="$BUILD_DIR/source/presets.hpp"
    {
        say "#if not defined(__PRESETS__)"
        say "#define __PRESETS__"
        say ""
        if [ "$BOUNDARY" = "open" ]; then
            say "#define OPEN_BC"
        else
            say "#define PERIODIC_BC"
        fi
        if [ "$SCHEME" = "fd" ]; then
            say "#define FD"
            say "#define KO_ORD_3"
            say "#define ADVEC_CENTER_FD"
        else
            say "#define FV"
        fi
        [ "$VACUUM_OSCILLATIONS" = "true" ] && say "#define VAC_OSC_ON"
        [ "$MATTER_OSCILLATIONS" = "true" ] && say "#define MAT_OSC_ON"
        [ "$COLLECTIVE_OSCILLATIONS" = "true" ] && say "#define COLL_OSC_ON"
        [ "$ADVECTION" = "false" ] && say "#define ADVEC_OFF"
        say ""
        say "#endif // __PRESETS__"
    } > "$presets"
}

replace_source_line() {
    local file="$1"
    local pattern="$2"
    local replacement="$3"
    local temporary="$file.tmp"
    sed "s|$pattern|$replacement|" "$file" > "$temporary" || return 1
    mv "$temporary" "$file"
    grep -F "$replacement" "$file" >/dev/null 2>&1 || \
        fail "Could not apply laptop setting in $(basename "$file")."
}

prepare_source() {
    mkdir -p "$BUILD_DIR/source"
    find "$BUILD_DIR/source" -type f -delete

    for file in "$SRC_DIR"/*.hpp; do
        [ "$(basename "$file")" = "presets.hpp" ] && continue
        cp "$file" "$BUILD_DIR/source/"
    done

    # These are the only GSL references in the project, and no GSL API is used.
    # Omitting the unused includes in this disposable copy avoids a large,
    # unnecessary laptop dependency while keeping src/ untouched.
    sed \
        -e '/^#include <gsl\/gsl_fft_complex.h>$/d' \
        -e '/^#include <gsl\/gsl_complex.h>$/d' \
        -e '/^using std::cin;/i\
typedef unsigned int uint; // portable replacement for a non-standard GSL transitive typedef\
' \
        "$SRC_DIR/main.cpp" > "$BUILD_DIR/source/main.cpp"

    # Collision coefficients remain compile-time constants on the established
    # monoenergetic RHS. Other profile parameters are read from job.config.
    replace_source_line "$BUILD_DIR/source/rhs_fd.hpp" \
        'double gamma_damp = .*' "double gamma_damp = $DAMPING_RATE; // C in units of mu"
    replace_source_line "$BUILD_DIR/source/rhs_fd.hpp" \
        'double alf = .*' "double alf = $ANTINEUTRINO_COLLISION_RATIO; // Cbar/C"
    replace_source_line "$BUILD_DIR/source/rhs_fv.hpp" \
        'double gamma_damp = .*' "double gamma_damp = $DAMPING_RATE; // C in units of mu"
    replace_source_line "$BUILD_DIR/source/rhs_fv.hpp" \
        'double alf = .*' "double alf = $ANTINEUTRINO_COLLISION_RATIO; // Cbar/C"

    if grep -R "gsl_" "$BUILD_DIR/source" >/dev/null 2>&1; then
        fail "The source now uses GSL; update the laptop build before continuing."
    fi
    write_presets
}

detect_openmp_flags() {
    local compiler="$1"
    local probe_source="$BUILD_DIR/openmp-test.cpp"
    local probe_binary="$BUILD_DIR/openmp-test"
    OPENMP_FLAGS=""
    OPENMP_COMPATIBILITY_MODE="false"

    cat > "$probe_source" <<'EOF'
#include <cstdio>
#include <fstream>
#include <vector>
#include <omp.h>
int main() {
    std::vector<double> values(4, 1.0);
    return values.empty() || omp_get_max_threads() < 1;
}
EOF

    if "$compiler" -fopenmp "$probe_source" -o "$probe_binary" >/dev/null 2>&1; then
        OPENMP_FLAGS="-fopenmp"
        return 0
    fi

    if [ "$(uname -s)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
        local libomp
        libomp="$(brew --prefix libomp 2>/dev/null || true)"
        if [ -n "$libomp" ] && \
           "$compiler" -Xpreprocessor -fopenmp -I"$libomp/include" \
             -L"$libomp/lib" -Wl,-rpath,"$libomp/lib" -lomp \
             "$probe_source" -o "$probe_binary" >/dev/null 2>&1; then
            OPENMP_FLAGS="-Xpreprocessor -fopenmp -I$libomp/include -L$libomp/lib -Wl,-rpath,$libomp/lib -lomp"
            return 0
        fi
        # An Apple-Silicon Mac can have an older Intel Homebrew installation.
        # Apple Clang and the system SDK are universal, so an x86_64 build can
        # still use that libomp safely through Rosetta.
        if [ "$(uname -m)" = "arm64" ] && \
           "$compiler" -arch x86_64 -Xpreprocessor -fopenmp \
             -I"$libomp/include" -L"$libomp/lib" \
             -Wl,-rpath,"$libomp/lib" -lomp \
             "$probe_source" -o "$probe_binary" >/dev/null 2>&1; then
            OPENMP_FLAGS="-arch x86_64 -Xpreprocessor -fopenmp -I$libomp/include -L$libomp/lib -Wl,-rpath,$libomp/lib -lomp"
            OPENMP_COMPATIBILITY_MODE="true"
            return 0
        fi
    fi
    return 1
}

find_openmp_compiler() {
    local preferred="$1"
    local candidate compiler_path seen=""

    OPENMP_COMPILER=""
    # Try the normal compiler first, then common native Homebrew LLVM paths,
    # and finally versioned GCC installations. This lets Apple Clang fall back
    # automatically when it lacks a compatible libomp installation.
    for candidate in \
        "$preferred" \
        /opt/homebrew/opt/llvm/bin/clang++ \
        /usr/local/opt/llvm/bin/clang++ \
        g++-15 g++-14 g++-13 g++-12 g++-11 g++ c++ clang++
    do
        if [ -x "$candidate" ]; then
            compiler_path="$candidate"
        elif command -v "$candidate" >/dev/null 2>&1; then
            compiler_path="$(command -v "$candidate")"
        else
            continue
        fi
        case " $seen " in
            *" $compiler_path "*) continue ;;
        esac
        seen="$seen $compiler_path"
        if detect_openmp_flags "$compiler_path"; then
            OPENMP_COMPILER="$compiler_path"
            return 0
        fi
    done
    return 1
}

build_cosenu() {
    local config_file="$1"
    local compiler serial_compiler
    local -a compile_command

    load_config "$config_file"
    compiler="$(find_compiler 2>/dev/null || true)"
    if [ -z "$compiler" ]; then
        setup_compiler || exit 1
        compiler="$(find_compiler 2>/dev/null || true)"
    fi

    say "Preparing an untouched-source laptop build..."
    prepare_source
    mkdir -p "$BUILD_DIR"

    serial_compiler="$compiler"
    if find_openmp_compiler "$compiler"; then
        compiler="$OPENMP_COMPILER"
        MULTICORE_BUILD="true"
        if [ "$compiler" != "$serial_compiler" ]; then
            say "OpenMP compiler selected automatically: $compiler"
        fi
        if [ "$OPENMP_COMPATIBILITY_MODE" = "true" ]; then
            say "OpenMP compatibility mode: Intel binary through Rosetta (Intel libomp detected)."
        fi
        say "Compiling with OpenMP multicore support ($compiler)..."
    else
        MULTICORE_BUILD="false"
        OPENMP_FLAGS=""
        say "OpenMP is unavailable; compiling in single-core mode ($compiler)..."
    fi

    compile_command=("$compiler" -O3 -std=c++11 -Wno-unknown-pragmas)
    if [ -n "$OPENMP_FLAGS" ]; then
        # Intentional word splitting: OPENMP_FLAGS contains compiler arguments.
        # shellcheck disable=SC2206
        compile_command+=( $OPENMP_FLAGS )
    fi
    compile_command+=("$BUILD_DIR/source/main.cpp" -o "$BUILD_DIR/main")

    if ! "${compile_command[@]}"; then
        if [ "$MULTICORE_BUILD" = "true" ] && [ "$compiler" != "$serial_compiler" ]; then
            say "OpenMP compilation failed; retrying safely in single-core mode ($serial_compiler)..."
            compiler="$serial_compiler"
            MULTICORE_BUILD="false"
            OPENMP_FLAGS=""
            compile_command=("$compiler" -O3 -std=c++11 -Wno-unknown-pragmas)
            compile_command+=("$BUILD_DIR/source/main.cpp" -o "$BUILD_DIR/main")
            "${compile_command[@]}" || \
                fail "Compilation failed. Run ./cosenu.sh setup, then try again."
        else
            fail "Compilation failed. Run ./cosenu.sh setup, then try again."
        fi
    fi
    if [ "$MULTICORE_BUILD" = "true" ]; then
        say "[OK] Build mode: OpenMP parallel"
    else
        say "[OK] Build mode: single CPU core"
    fi
    say "[OK] Build complete: $BUILD_DIR/main"
}

write_job_config() {
    local destination="$1"
    cat > "$destination" <<EOF
scheme              : $(printf '%s' "$SCHEME" | tr '[:lower:]' '[:upper:]')
z0                  : $Z_MIN
z1                  : $Z_MAX
nvz                 : $NVZ
CFL                 : $CFL
gz                  : $GHOST_ZONES
v0                  : $V_MIN
v1                  : $V_MAX
nz                  : $NZ
pmo                 : $MASS_ORDERING
omega               : $OMEGA
theta               : $MIXING_ANGLE
mu                  : $MU
k_max               : $K_MAX
k_bins               : $K_BINS
perturbation_size   : $PERTURBATION_SIZE
N_ITER              : $N_ITER
ANAL_EVERY          : $ANAL_EVERY
n_dump_rho          : $SNAPSHOTS
multienergy         : $([ "$MULTIENERGY" = "true" ] && printf 1 || printf 0)
energy_grid         : $ENERGY_GRID
energy_bins         : $ENERGY_BINS
energy_min          : $ENERGY_MIN
energy_max          : $ENERGY_MAX
omega_grid_min      : $OMEGA_GRID_MIN
omega_grid_max      : $OMEGA_GRID_MAX
reference_energy    : $REFERENCE_ENERGY
spectral_alpha      : $SPECTRAL_ALPHA
mean_energy_nue     : $MEAN_ENERGY_NUE
mean_energy_anue    : $MEAN_ENERGY_ANTINUE
mean_energy_nux     : $MEAN_ENERGY_NUX
luminosity_nue      : $LUMINOSITY_NUE
luminosity_anue     : $LUMINOSITY_ANTINUE
luminosity_nux      : $LUMINOSITY_NUX
neutrino_width      : $NEUTRINO_WIDTH
antineutrino_width  : $ANTINEUTRINO_WIDTH
alpha               : $ALPHA
angular_profile     : $ANGULAR_PROFILE
random_seeds        : $([ "$RANDOM_SEEDS" = "true" ] && printf 1 || printf 0)
seed_width           : $SEED_WIDTH
matter_slope        : $MATTER_SLOPE
matter_offset       : $MATTER_OFFSET
exact_matter_rotation : $([ "$EXACT_MATTER_ROTATION" = "true" ] && printf 1 || printf 0)
damping_rate        : $DAMPING_RATE
antineutrino_collision_ratio : $ANTINEUTRINO_COLLISION_RATIO
EOF
}

resolved_threads() {
    local available
    if [ "$THREADS" != "auto" ]; then
        case "$THREADS" in ''|*[!0-9]*) fail "THREADS must be auto or a positive integer." ;; esac
        [ "$THREADS" -ge 1 ] || fail "THREADS must be at least 1."
        printf '%s\n' "$THREADS"
        return
    fi

    if command -v getconf >/dev/null 2>&1; then
        available="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
    elif command -v sysctl >/dev/null 2>&1; then
        available="$(sysctl -n hw.logicalcpu 2>/dev/null || true)"
    else
        available="1"
    fi
    case "$available" in ''|*[!0-9]*) available=1 ;; esac
    # Twelve threads is a near-optimal, conservative default on the development
    # Mac. Avoid oversubscribing smaller laptops by using the lower value.
    [ "$available" -gt 12 ] && available=12
    printf '%s\n' "$available"
}

confirm_large_run() {
    if awk -v kmax="$K_MAX" -v nyquist="$NYQUIST_K" 'BEGIN { exit !(kmax > nyquist) }'; then
        say ""
        say "Warning: K_MAX=$K_MAX exceeds the grid Nyquist limit $NYQUIST_K MU."
        say "The spectrum above that limit will be aliased. For this domain, use NZ=$MIN_NZ_FOR_KMAX or higher."
    fi
    if awk -v work="$WORK_UNITS" 'BEGIN { exit !(work > 100000000) }'; then
        say ""
        if [ "$MULTIENERGY" = "true" ]; then
            say "Warning: this is a large laptop run ($NZ z x $NVZ vz x $((2 * ENERGY_BINS)) energy grid, $N_ITER iterations)."
            say "         ENERGY_BINS=$ENERGY_BINS per sector ($ENERGY_BINS neutrino + $ENERGY_BINS antineutrino)."
        else
            say "Warning: this is a large laptop run ($NZ x $NVZ grid, $N_ITER iterations)."
        fi
        say "It may take a long time and create large output files."
        if [ -t 0 ]; then
            printf 'Continue? [y/N] '
            read -r answer
            case "$answer" in y|Y|yes|YES) ;; *) say "Cancelled."; exit 0 ;; esac
        else
            fail "Large non-interactive run cancelled for safety."
        fi
    fi
}

new_run_directory() {
    local base="$ROOT_DIR/runs/laptop-$(date '+%Y%m%d-%H%M%S')"
    local candidate="$base"
    local suffix=2
    while [ -e "$candidate" ]; do
        candidate="$base-$suffix"
        suffix=$((suffix + 1))
    done
    mkdir -p "$candidate"
    printf '%s\n' "$candidate"
}

generate_plots() {
    local run_dir="$1"
    ensure_plot_environment || return 1
    "$ANALYSIS_DIR/run_plotting.sh" "$run_dir" "$PLOT_PYTHON"
}

generate_lsa() {
    local run_dir="$1"
    ensure_plot_environment || return 1
    "$LSA_DIR/run_lsa.sh" "$run_dir" "$PLOT_PYTHON" \
        "$LSA_NVZ" "$LSA_NUM" "$LSA_K_MIN" "$LSA_K_MAX"
}

plot_existing_run() {
    local run_dir="$1"
    [ -d "$run_dir" ] || fail "Run directory not found: $run_dir"
    if [ -f "$run_dir/laptop.conf" ]; then
        load_config "$run_dir/laptop.conf"
    fi
    generate_plots "$run_dir" || fail "Could not generate plots for $run_dir"
}

lsa_existing_run() {
    local run_dir="$1"
    [ -d "$run_dir" ] || fail "Run directory not found: $run_dir"
    run_dir="$(cd "$run_dir" && /bin/pwd -P)"
    [ -f "$run_dir/laptop.conf" ] || fail "Cannot run LSA: $run_dir/laptop.conf is missing."
    # The completed run supplies its original physics. Only the independent
    # LSA controls come from the current top-level laptop.conf.
    load_config "$run_dir/laptop.conf"
    load_config "$DEFAULT_CONFIG"
    say "LSA controls from laptop.conf: NVZ=$LSA_NVZ, NUM=$LSA_NUM, k=$LSA_K_MIN to $LSA_K_MAX"
    generate_lsa "$run_dir" || fail "Could not complete LSA for $run_dir"
    if [ "$PLOTS" = "yes" ]; then
        say "Refreshing plots with the LSA dispersion overlay."
        generate_plots "$run_dir" || fail "LSA completed, but plots could not be refreshed for $run_dir"
    fi
}

setup_all() {
    setup_compiler || return 1
    load_config "$DEFAULT_CONFIG"
    if [ "$PLOTS" = "yes" ] || [ "$LINEAR_STABILITY_ANALYSIS" = "true" ]; then
        ensure_plot_environment
    fi
}

run_cosenu() {
    local config_file="$1"
    local run_dir threads status

    build_cosenu "$config_file"
    confirm_large_run
    if [ "$PLOTS" = "yes" ] || [ "$LINEAR_STABILITY_ANALYSIS" = "true" ]; then
        ensure_plot_environment || exit 1
    fi
    run_dir="$(new_run_directory)"
    cp "$BUILD_DIR/main" "$run_dir/main"
    cp "$config_file" "$run_dir/laptop.conf"
    write_job_config "$run_dir/job.config"

    threads="$(resolved_threads)"
    if [ "$MULTICORE_BUILD" != "true" ]; then
        threads=1
    fi

    say ""
    say "Running COSEnu"
    if [ "$MULTIENERGY" = "true" ]; then
        say "  grid:       $NZ x $NVZ x $((2 * ENERGY_BINS)) energy modes"
    else
        say "  grid:       $NZ x $NVZ (monoenergetic)"
    fi
    say "  iterations: $N_ITER"
    say "  memory:     about $ESTIMATED_MB MB"
    say "  spectrum:   -$K_MAX <= k <= $K_MAX ($K_BINS bins)"
    if [ "$EXACT_MATTER_ROTATION" = "true" ]; then
        say "  matter:     exact half-phase / RK4 / exact half-phase"
    fi
    if [ "$MULTICORE_BUILD" = "true" ] && [ "$threads" -gt 1 ]; then
        say "  execution:  OpenMP parallel ($threads CPU threads)"
    elif [ "$MULTICORE_BUILD" = "true" ]; then
        say "  execution:  OpenMP enabled (1 CPU thread)"
    else
        say "  execution:  single CPU thread (OpenMP unavailable)"
    fi
    say "  results:    $run_dir"
    say ""

    (
        cd "$run_dir" || exit 1
        OMP_DYNAMIC=FALSE OMP_NUM_THREADS="$threads" ./main --conf job.config
    )
    status=$?
    if [ "$status" -ne 0 ]; then
        fail "COSEnu stopped with exit code $status. Partial results are in $run_dir"
    fi

    say ""
    say "[OK] Simulation finished."
    say "Results: $run_dir"
    # Run the LSA first so the standard power-spectrum figure can overlay its
    # dispersion curve and most-unstable mode during the same launcher run.
    if [ "$LINEAR_STABILITY_ANALYSIS" = "true" ]; then
        say ""
        if generate_lsa "$run_dir"; then
            say "LSA:     $run_dir/lsa"
        else
            say "Warning: the simulation finished, but linear stability analysis failed."
            say "You can retry with: ./cosenu.sh lsa '$run_dir'"
        fi
    else
        say "LSA:     disabled (LINEAR_STABILITY_ANALYSIS=false)"
    fi
    if [ "$PLOTS" = "yes" ]; then
        say ""
        if generate_plots "$run_dir"; then
            say "Plots:   $run_dir/plots"
        else
            say "Warning: the simulation finished, but plotting failed."
            say "You can retry with: ./cosenu.sh plot '$run_dir'"
        fi
    else
        say "Plots:   disabled (PLOTS=no)"
    fi
}

restart_cosenu() {
    local run_dir="$1"
    local threads
    [ -d "$run_dir" ] || fail "Run directory not found: $run_dir"
    run_dir="$(cd "$run_dir" && /bin/pwd -P)"
    load_config "$run_dir/laptop.conf"
    if [ "$MULTIENERGY" = "true" ]; then
        restart_files="main job.config state.bin"
    else
        restart_files="main job.config state.bin G0.bin state0.bin"
    fi
    for needed in $restart_files; do
        [ -f "$run_dir/$needed" ] || fail "Cannot restart: $run_dir/$needed is missing."
    done

    threads="$(resolved_threads)"
    say "Restarting from the last checkpoint in $run_dir"
    (
        cd "$run_dir" || exit 1
        OMP_DYNAMIC=FALSE OMP_NUM_THREADS="$threads" ./main --conf job.config --ff
    ) || fail "Restart failed. Existing results were kept in $run_dir"
    if [ "${LINEAR_STABILITY_ANALYSIS:-false}" = "true" ]; then
        generate_lsa "$run_dir" || \
            say "Warning: restart completed, but LSA could not be refreshed."
    fi
    if [ "${PLOTS:-no}" = "yes" ]; then
        generate_plots "$run_dir" || \
            say "Warning: restart completed, but plots could not be refreshed."
    fi
}

command_name="${1:-help}"
case "$command_name" in
    run)
        run_cosenu "${2:-$DEFAULT_CONFIG}"
        ;;
    setup)
        setup_all
        ;;
    build)
        build_cosenu "${2:-$DEFAULT_CONFIG}"
        ;;
    restart)
        [ -n "${2:-}" ] || fail "Example: ./cosenu.sh restart runs/laptop-20260903-191021"
        restart_cosenu "$2"
        ;;
    plot)
        [ -n "${2:-}" ] || fail "Example: ./cosenu.sh plot runs/laptop-20260903-191021"
        plot_existing_run "$2"
        ;;
    lsa)
        [ -n "${2:-}" ] || fail "Example: ./cosenu.sh lsa runs/laptop-20260903-191021"
        lsa_existing_run "$2"
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        usage
        fail "Unknown command: $command_name"
        ;;
esac
