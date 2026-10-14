#if !defined(__MULTIENERGY__)
#define __MULTIENERGY__

/*
 * Optional multi-energy solver.
 *
 * The original NuOsc implementation remains the complete monoenergetic code
 * path.  This class uses the signed-frequency (flavor-isospin) convention:
 * omega > 0 labels neutrinos and omega < 0 labels antineutrinos.  It therefore
 * needs four real arrays per mode instead of duplicating eight arrays, and it
 * accumulates the energy moments once per RHS evaluation.
 */

#include <algorithm>
#include <cstdint>
#include <limits>

struct MEField
{
    std::vector<double> ee, xx, ex_re, ex_im;
    explicit MEField(std::size_t n = 0) : ee(n, 0.0), xx(n, 0.0), ex_re(n, 0.0), ex_im(n, 0.0) {}
};

class MultiEnergyNuOsc
{
public:
    explicit MultiEnergyNuOsc(const Params &p);

    void initialize();
    void step_rk4();
    void analyse(int iteration, bool first);
    void dump(int iteration);
    void write_checkpoint(int iteration) const;
    int read_checkpoint();

    double dt;
    int n_iter;
    int analysis_every;
    int dump_interval;
    double average_p_error() const { return latest_average_p_error; }

private:
    const Params pars;
    int nz, nv, ne, nm, gz, stride;
    std::size_t total_size;
    double z0, z1, v0, v1, dz, dv;
    double theta, cos2theta, sin2theta;
    std::vector<double> z, velocity, matter;
    std::vector<double> energy, omega, mode_weight;
    std::vector<double> initial_e_spectrum, initial_x_spectrum;
    std::vector<double> initial_e_bin, initial_x_bin;
    std::vector<double> angular_profile;
    MEField state, work, rhs, accumulator;
    std::vector<double> Iee, Ixx, Ire, Iim, Jee, Jxx, Jre, Jim;
    double latest_average_p_error = 0.0;

    inline std::size_t idx(int e, int v, int j) const
    {
        return (static_cast<std::size_t>(e) * nv + v) * stride + (j + gz);
    }
    double spectrum(double E, double mean, double luminosity) const;
    void build_spectrum();
    void update_boundaries(MEField &f);
    void calculate_rhs(MEField &out, const MEField &in);
    void combine(MEField &out, const MEField &a, double scale, const MEField &b);
    void integrated_cell(const MEField &f, int v, int j, double values[8]) const;
    void write_survival(int iteration, bool first);
    void write_conservation(int iteration, bool first);
    void write_power(int iteration, bool first);
    void write_energy_diagnostics(int iteration) const;
    void write_energy_angle(int iteration) const;
    double conserved_q(const MEField &f) const;
};

double MultiEnergyNuOsc::spectrum(double E, double mean, double luminosity) const
{
    const double a = pars.spectral_alpha;
    const double number_ratio = (luminosity / mean) /
                                (pars.luminosity_nue / pars.mean_energy_nue);
    const double norm = number_ratio * std::pow(a + 1.0, a + 1.0) /
                        std::tgamma(a + 1.0) / std::pow(mean, a + 1.0);
    return norm * std::pow(E, a) * std::exp(-(a + 1.0) * E / mean);
}

MultiEnergyNuOsc::MultiEnergyNuOsc(const Params &p)
    : pars(p), nz(p.nz), nv(p.nvz), ne(p.energy_bins), nm(2 * p.energy_bins), gz(p.gz),
      stride(p.nz + 2 * p.gz),
      total_size(static_cast<std::size_t>(2 * p.energy_bins) * p.nvz * (p.nz + 2 * p.gz)),
      z0(p.z0), z1(p.z1), v0(p.v0), v1(p.v1), dz((p.z1 - p.z0) / p.nz),
      dv((p.v1 - p.v0) / p.nvz), theta(p.theta * M_PI / 180.0),
      cos2theta(std::cos(2.0 * theta)), sin2theta(std::sin(2.0 * theta)),
      z(nz), velocity(nv), matter(nz), energy(nm), omega(nm), mode_weight(nm),
      initial_e_spectrum(nm), initial_x_spectrum(nm), initial_e_bin(nm), initial_x_bin(nm),
      angular_profile(static_cast<std::size_t>(nm) * nv), state(total_size), work(total_size),
      rhs(total_size), accumulator(total_size), Iee(nz), Ixx(nz), Ire(nz), Iim(nz),
      Jee(nz), Jxx(nz), Jre(nz), Jim(nz)
{
    dt = p.CFL * dz / std::max(std::fabs(v0), std::fabs(v1));
    n_iter = p.N_ITER;
    analysis_every = p.ANAL_EVERY;
    dump_interval = p.dump_interval;
    for (int j = 0; j < nz; ++j)
    {
        z[j] = z0 + (j + 0.5) * dz;
        matter[j] = p.matter_offset + p.matter_slope * (z[j] - z0);
    }
    for (int v = 0; v < nv; ++v)
        velocity[v] = v0 + (v + 0.5) * dv;
    build_spectrum();

    // Four MEField objects (state, work, rhs, accumulator), each containing
    // four double arrays, dominate the Runge--Kutta memory footprint.
    const double gib = total_size * 16.0 * sizeof(double) / (1024.0 * 1024.0 * 1024.0);
    std::cout << draw(50, "#") << "\n"
              << std::setw(35) << "Multi-energy state created" << "\n"
              << draw(50, ".") << "\n"
              << "\tgrid (Nenergy/sector, Nvz, Nz): (" << ne << ", " << nv << ", " << nz << ")\n"
              << "\ttotal energy modes: " << nm << "\n"
              << "\tenergy grid: " << pars.energy_grid << "\n"
              << "\tRK storage: about " << std::setprecision(3) << gib << " GiB\n"
              << draw(50, "#") << std::endl;
}

void MultiEnergyNuOsc::build_spectrum()
{
    const double C = std::fabs(pars.omega) * pars.reference_energy;
    const double dcoord = pars.energy_grid == "energy"
                              ? (pars.energy_max - pars.energy_min) / ne
                              : (pars.omega_grid_max - pars.omega_grid_min) / ne;
    for (int sector = 0; sector < 2; ++sector)
    {
        for (int q = 0; q < ne; ++q)
        {
            const int e = sector * ne + q;
            double jacobian = 1.0;
            if (pars.energy_grid == "energy")
            {
                energy[e] = pars.energy_min + (q + 0.5) * dcoord;
                omega[e] = (sector == 0 ? 1.0 : -1.0) * C / energy[e];
            }
            else
            {
                const double abs_omega = pars.omega_grid_min + (q + 0.5) * dcoord;
                omega[e] = (sector == 0 ? 1.0 : -1.0) * abs_omega;
                energy[e] = C / abs_omega;
                jacobian = energy[e] / abs_omega; // |dE/domega|
            }

            const double fnue = spectrum(energy[e], pars.mean_energy_nue, pars.luminosity_nue);
            const double fanue = spectrum(energy[e], pars.mean_energy_anue, pars.luminosity_anue);
            const double fnux = spectrum(energy[e], pars.mean_energy_nux, pars.luminosity_nux);
            const double integration_weight = dcoord * jacobian;
            if (sector == 0)
            {
                initial_e_spectrum[e] = fnue;
                initial_x_spectrum[e] = fnux;
                initial_e_bin[e] = fnue * integration_weight;
                initial_x_bin[e] = fnux * integration_weight;
                mode_weight[e] = (fnue - fnux) * integration_weight;
            }
            else
            {
                initial_e_spectrum[e] = fanue;
                initial_x_spectrum[e] = fnux;
                initial_e_bin[e] = fanue * integration_weight;
                initial_x_bin[e] = fnux * integration_weight;
                mode_weight[e] = (fnux - fanue) * integration_weight;
            }
            const double width = sector == 0 ? pars.neutrino_width : pars.antineutrino_width;
            for (int v = 0; v < nv; ++v)
                angular_profile[static_cast<std::size_t>(e) * nv + v] = g(velocity[v], 1.0, width);
        }
    }

    std::ofstream out("energy_grid.dat", std::ios::out | std::ios::trunc);
    out << "# sector E_MeV omega mode_weight f_e(E) f_x(E) bin_e bin_x\n";
    out << std::setprecision(16);
    for (int e = 0; e < nm; ++e)
        out << (e < ne ? 1 : -1) << '\t' << energy[e] << '\t' << omega[e] << '\t'
            << mode_weight[e] << '\t' << initial_e_spectrum[e] << '\t'
            << initial_x_spectrum[e] << '\t' << initial_e_bin[e] << '\t'
            << initial_x_bin[e] << '\n';
}

void MultiEnergyNuOsc::initialize()
{
#pragma omp parallel for collapse(2)
    for (int e = 0; e < nm; ++e)
        for (int v = 0; v < nv; ++v)
#pragma omp simd
            for (int j = 0; j < nz; ++j)
            {
                const std::size_t n = idx(e, v, j);
                const double epsilon = pars.perturbation_size *
                    std::exp(-z[j] * z[j] / (2.0 * pars.seed_width * pars.seed_width));
                const double p3 = std::sqrt(std::max(0.0, 1.0 - epsilon * epsilon));
                const double trace = mode_weight[e] * angular_profile[static_cast<std::size_t>(e) * nv + v];
                state.ee[n] = 0.5 * trace * (1.0 + p3);
                state.xx[n] = 0.5 * trace * (1.0 - p3);
                state.ex_re[n] = 0.5 * trace * epsilon;
                state.ex_im[n] = 0.0;
            }
    update_boundaries(state);

    std::cout << "Multi-energy simulation state initialized." << std::endl;
}

void MultiEnergyNuOsc::update_boundaries(MEField &f)
{
#pragma omp parallel for collapse(2)
    for (int e = 0; e < nm; ++e)
        for (int v = 0; v < nv; ++v)
            for (int j = 0; j < gz; ++j)
            {
#ifdef PERIODIC_BC
                const int lower_source = nz - j - 1;
                const int upper_source = j;
#else
                const int lower_source = 0;
                const int upper_source = nz - 1;
#endif
                const std::size_t lo = idx(e, v, -j - 1), ls = idx(e, v, lower_source);
                const std::size_t hi = idx(e, v, nz + j), hs = idx(e, v, upper_source);
                f.ee[lo] = f.ee[ls]; f.xx[lo] = f.xx[ls]; f.ex_re[lo] = f.ex_re[ls]; f.ex_im[lo] = f.ex_im[ls];
                f.ee[hi] = f.ee[hs]; f.xx[hi] = f.xx[hs]; f.ex_re[hi] = f.ex_re[hs]; f.ex_im[hi] = f.ex_im[hs];
            }
}

void MultiEnergyNuOsc::calculate_rhs(MEField &out, const MEField &in)
{
#pragma omp parallel for
    for (int j = 0; j < nz; ++j)
    {
        double iee = 0.0, ixx = 0.0, ire = 0.0, iim = 0.0;
        double jee = 0.0, jxx = 0.0, jre = 0.0, jim = 0.0;
        for (int e = 0; e < nm; ++e)
            for (int v = 0; v < nv; ++v)
            {
                const std::size_t n = idx(e, v, j);
                iee += in.ee[n]; ixx += in.xx[n]; ire += in.ex_re[n]; iim += in.ex_im[n];
                jee += velocity[v] * in.ee[n]; jxx += velocity[v] * in.xx[n];
                jre += velocity[v] * in.ex_re[n]; jim += velocity[v] * in.ex_im[n];
            }
        Iee[j] = iee * dv; Ixx[j] = ixx * dv; Ire[j] = ire * dv; Iim[j] = iim * dv;
        Jee[j] = jee * dv; Jxx[j] = jxx * dv; Jre[j] = jre * dv; Jim[j] = jim * dv;
    }

#pragma omp parallel for collapse(2)
    for (int e = 0; e < nm; ++e)
        for (int v = 0; v < nv; ++v)
#pragma omp simd
            for (int j = 0; j < nz; ++j)
            {
                const std::size_t n = idx(e, v, j);
                const double *ee = &in.ee[n], *xx = &in.xx[n];
                const double *xr = &in.ex_re[n], *xi = &in.ex_im[n];
                double hee_minus_hxx = 0.0, hex_re = 0.0, hex_im = 0.0;
#ifdef VAC_OSC_ON
                hee_minus_hxx += -pars.pmo * omega[e] * cos2theta;
                hex_re += 0.5 * pars.pmo * omega[e] * sin2theta;
#endif
#ifdef MAT_OSC_ON
                hee_minus_hxx += matter[j];
#endif
#ifdef COLL_OSC_ON
                hee_minus_hxx += pars.mu * ((Iee[j] - Ixx[j]) - velocity[v] * (Jee[j] - Jxx[j]));
                hex_re += pars.mu * (Ire[j] - velocity[v] * Jre[j]);
                hex_im += pars.mu * (Iim[j] - velocity[v] * Jim[j]);
#endif
                const double population_delta = xx[0] - ee[0];
                out.ee[n] = 2.0 * (hex_im * xr[0] - hex_re * xi[0]);
                out.xx[n] = -out.ee[n];
                out.ex_re[n] = hee_minus_hxx * xi[0] + hex_im * population_delta;
                out.ex_im[n] = -hee_minus_hxx * xr[0] - hex_re * population_delta;

                const double damping = e < ne ? pars.damping_rate
                                               : pars.damping_rate * pars.antineutrino_collision_ratio;
                out.ex_re[n] -= damping * xr[0];
                out.ex_im[n] -= damping * xi[0];

#ifndef ADVEC_OFF
                const double factor = -velocity[v] / (12.0 * dz);
#define ME_ADV(x) ((x[-2] - x[2]) - 8.0 * (x[-1] - x[1]))
                out.ee[n] += factor * ME_ADV(ee); out.xx[n] += factor * ME_ADV(xx);
                out.ex_re[n] += factor * ME_ADV(xr); out.ex_im[n] += factor * ME_ADV(xi);
#undef ME_ADV
#endif
                const double ko = -0.1 / (16.0 * dz);
#define ME_KO(x) (x[-2] + x[2] - 4.0 * (x[-1] + x[1]) + 6.0 * x[0])
                out.ee[n] += ko * ME_KO(ee); out.xx[n] += ko * ME_KO(xx);
                out.ex_re[n] += ko * ME_KO(xr); out.ex_im[n] += ko * ME_KO(xi);
#undef ME_KO
            }
}

void MultiEnergyNuOsc::combine(MEField &out, const MEField &a, double scale, const MEField &b)
{
#pragma omp parallel for
    for (std::int64_t n = 0; n < static_cast<std::int64_t>(total_size); ++n)
    {
        out.ee[n] = a.ee[n] + scale * b.ee[n]; out.xx[n] = a.xx[n] + scale * b.xx[n];
        out.ex_re[n] = a.ex_re[n] + scale * b.ex_re[n]; out.ex_im[n] = a.ex_im[n] + scale * b.ex_im[n];
    }
    update_boundaries(out);
}

void MultiEnergyNuOsc::step_rk4()
{
    calculate_rhs(rhs, state);
#pragma omp parallel for
    for (std::int64_t n = 0; n < static_cast<std::int64_t>(total_size); ++n)
    {
        accumulator.ee[n] = rhs.ee[n]; accumulator.xx[n] = rhs.xx[n];
        accumulator.ex_re[n] = rhs.ex_re[n]; accumulator.ex_im[n] = rhs.ex_im[n];
    }
    combine(work, state, 0.5 * dt, rhs);
    calculate_rhs(rhs, work);
#pragma omp parallel for
    for (std::int64_t n = 0; n < static_cast<std::int64_t>(total_size); ++n)
    {
        accumulator.ee[n] += 2.0 * rhs.ee[n]; accumulator.xx[n] += 2.0 * rhs.xx[n];
        accumulator.ex_re[n] += 2.0 * rhs.ex_re[n]; accumulator.ex_im[n] += 2.0 * rhs.ex_im[n];
    }
    combine(work, state, 0.5 * dt, rhs);
    calculate_rhs(rhs, work);
#pragma omp parallel for
    for (std::int64_t n = 0; n < static_cast<std::int64_t>(total_size); ++n)
    {
        accumulator.ee[n] += 2.0 * rhs.ee[n]; accumulator.xx[n] += 2.0 * rhs.xx[n];
        accumulator.ex_re[n] += 2.0 * rhs.ex_re[n]; accumulator.ex_im[n] += 2.0 * rhs.ex_im[n];
    }
    combine(work, state, dt, rhs);
    calculate_rhs(rhs, work);
#pragma omp parallel for
    for (std::int64_t n = 0; n < static_cast<std::int64_t>(total_size); ++n)
    {
        state.ee[n] += dt * (accumulator.ee[n] + rhs.ee[n]) / 6.0;
        state.xx[n] += dt * (accumulator.xx[n] + rhs.xx[n]) / 6.0;
        state.ex_re[n] += dt * (accumulator.ex_re[n] + rhs.ex_re[n]) / 6.0;
        state.ex_im[n] += dt * (accumulator.ex_im[n] + rhs.ex_im[n]) / 6.0;
    }
    update_boundaries(state);
}

void MultiEnergyNuOsc::integrated_cell(const MEField &f, int v, int j, double a[8]) const
{
    std::fill(a, a + 8, 0.0);
    for (int e = 0; e < ne; ++e)
    {
        const std::size_t n = idx(e, v, j);
        a[0] += f.ee[n]; a[1] += f.xx[n]; a[2] += f.ex_re[n]; a[3] += f.ex_im[n];
    }
    for (int e = ne; e < nm; ++e)
    {
        const std::size_t n = idx(e, v, j);
        a[4] -= f.ee[n]; a[5] -= f.xx[n]; a[6] -= f.ex_re[n]; a[7] += f.ex_im[n];
    }
}

void MultiEnergyNuOsc::dump(int iteration)
{
    const std::string file_name = "rho_" + std::to_string(iteration) + ".npy";
    const std::size_t rows = static_cast<std::size_t>(nz) * nv;
    std::vector<double> snapshot(rows * 10);
#pragma omp parallel for collapse(2)
    for (int j = 0; j < nz; ++j)
        for (int v = 0; v < nv; ++v)
        {
            double a[8]; integrated_cell(state, v, j, a);
            const std::size_t n = (static_cast<std::size_t>(j) * nv + v) * 10;
            snapshot[n] = z[j];
            snapshot[n + 1] = velocity[v];
            for (int q = 0; q < 8; ++q)
                snapshot[n + 2 + q] = a[q];
        }
    if (!write_numpy_matrix(file_name, snapshot, rows, 10))
        std::cerr << "Unable to write " << file_name << std::endl;
    write_energy_diagnostics(iteration);
    write_energy_angle(iteration);
}

void MultiEnergyNuOsc::write_survival(int iteration, bool first)
{
    double nu = 0.0, anu = 0.0, nu_norm = 0.0, anu_norm = 0.0;
    double nu_coherence = 0.0, anu_coherence = 0.0;
    for (int e = 0; e < nm; ++e)
        for (int v = 0; v < nv; ++v)
            for (int j = 0; j < nz; ++j)
            {
                const double profile = angular_profile[static_cast<std::size_t>(e) * nv + v];
                const double trace = mode_weight[e] * profile;
                const double psur = state.ee[idx(e, v, j)] / trace;
                const double probability_weight = initial_e_bin[e] * profile * dz * dv;
                const double coherence = std::hypot(
                    state.ex_re[idx(e, v, j)], state.ex_im[idx(e, v, j)]) * dz * dv;
                if (e < ne)
                {
                    nu += probability_weight * psur;
                    nu_norm += probability_weight;
                    nu_coherence += coherence;
                }
                else
                {
                    anu += probability_weight * psur;
                    anu_norm += probability_weight;
                    anu_coherence += coherence;
                }
            }
    std::ofstream out("survival_probability.dat", first ? std::ios::trunc : std::ios::app);
    if (first) out << "# [iteration, energy-integrated <Pee>, <Pbee>]\n";
    out << iteration << '\t' << std::setprecision(16) << nu / nu_norm << '\t'
        << anu / anu_norm << '\n';
    std::ofstream coherence_out(
        "multienergy_coherence.dat", first ? std::ios::trunc : std::ios::app);
    if (first)
        coherence_out << "# iteration sum_{omega>0,v,z}|rho_ex| sum_{omega<0,v,z}|rho_ex|\n";
    coherence_out << iteration << '\t' << std::setprecision(16) << nu_coherence
                  << '\t' << anu_coherence << '\n';
}

double MultiEnergyNuOsc::conserved_q(const MEField &f) const
{
    // Global projection on the vacuum mass direction.  Unlike the local
    // self-interaction energy, this linear lepton-number invariant remains
    // conserved for an inhomogeneous, periodically streamed gas (when matter
    // and collisions are absent).  Do not weight by omega: every energy mode
    // contributes to the same global polarization vector.
    double mass_lepton_projection = 0.0;
    for (int e = 0; e < nm; ++e)
        for (int v = 0; v < nv; ++v)
            for (int j = 0; j < nz; ++j)
            {
                const std::size_t n = idx(e, v, j);
                mass_lepton_projection +=
                    (2.0 * sin2theta * f.ex_re[n] -
                     cos2theta * (f.ee[n] - f.xx[n])) * dz * dv;
            }
    return mass_lepton_projection;
}

void MultiEnergyNuOsc::write_conservation(int iteration, bool first)
{
    double maximum = 0.0, average_nu = 0.0, average_anu = 0.0;
    double weight_nu = 0.0, weight_anu = 0.0;
    double moments[11] = {0.0};
    double moment_vectors[11][3] = {{0.0}};
    for (int e = 0; e < nm; ++e)
        for (int v = 0; v < nv; ++v)
        {
            double integrated_vector[3] = {0.0, 0.0, 0.0};
            for (int j = 0; j < nz; ++j)
            {
                const std::size_t n = idx(e, v, j);
                const double trace = mode_weight[e] * angular_profile[static_cast<std::size_t>(e) * nv + v];
                if (std::fabs(trace) < 1.0e-15) continue;
                const double p1 = 2.0 * state.ex_re[n] / trace;
                const double p2 = -2.0 * state.ex_im[n] / trace;
                const double p3 = (state.ee[n] - state.xx[n]) / trace;
                const double deviation = std::fabs(std::sqrt(p1 * p1 + p2 * p2 + p3 * p3) - 1.0);
                maximum = std::max(maximum, deviation);
                if (e < ne) { average_nu += deviation * std::fabs(trace); weight_nu += std::fabs(trace); }
                else { average_anu += deviation * std::fabs(trace); weight_anu += std::fabs(trace); }
                integrated_vector[0] += 2.0 * state.ex_re[n] * dz;
                integrated_vector[1] += -2.0 * state.ex_im[n] * dz;
                integrated_vector[2] += (state.ee[n] - state.xx[n]) * dz;
            }
            for (int l = 0; l <= 10; ++l)
                for (int component = 0; component < 3; ++component)
                    moment_vectors[l][component] +=
                        L(velocity[v], l) * integrated_vector[component] * dv;
        }
    for (int l = 0; l <= 10; ++l)
        moments[l] = std::sqrt(
            moment_vectors[l][0] * moment_vectors[l][0] +
            moment_vectors[l][1] * moment_vectors[l][1] +
            moment_vectors[l][2] * moment_vectors[l][2]);
    latest_average_p_error =
        (average_nu + average_anu) /
        std::max(weight_nu + weight_anu, 1.0e-300);
    std::ofstream out("conserved_quantities.dat", first ? std::ios::trunc : std::ios::app);
    if (first) out << "# iteration dPmax <dPnu> <dPanu> Lmass M0 ... M10\n";
    out << iteration << '\t' << std::setprecision(16) << maximum << '\t'
        << average_nu / std::max(weight_nu, 1.0e-300) << '\t'
        << average_anu / std::max(weight_anu, 1.0e-300) << '\t' << conserved_q(state);
    for (int l = 0; l <= 10; ++l) out << '\t' << moments[l];
    out << '\n';
}

void MultiEnergyNuOsc::write_power(int iteration, bool first)
{
    std::vector<double> signal_re(nz, 0.0), signal_im(nz, 0.0);
    for (int j = 0; j < nz; ++j)
        for (int e = 0; e < ne; ++e)
            for (int v = 0; v < nv; ++v)
            {
                const std::size_t n = idx(e, v, j);
                signal_re[j] += 2.0 * state.ex_re[n] * dv;
                signal_im[j] += 2.0 * state.ex_im[n] * dv;
            }
    std::ofstream out("power_spectrum.dat", first ? std::ios::trunc : std::ios::app);
    if (first) out << "# t k Re[F(k)] Im[F(k)] Power Phase\n";
    for (int m = 0; m < pars.k_bins; ++m)
    {
        const double k = -pars.k_max + 2.0 * pars.k_max * m / (pars.k_bins - 1.0);
        double re = 0.0, im = 0.0;
        for (int j = 0; j < nz; ++j)
        {
            const double x = static_cast<double>(j);
            const double window = 0.355768 - 0.487396 * std::cos(2.0 * M_PI * x / (nz - 1.0))
                                  + 0.144232 * std::cos(4.0 * M_PI * x / (nz - 1.0))
                                  - 0.012604 * std::cos(6.0 * M_PI * x / (nz - 1.0));
            const double c = std::cos(k * z[j]), s = std::sin(k * z[j]);
            re += window * (signal_re[j] * c + signal_im[j] * s) * dz;
            im += window * (signal_im[j] * c - signal_re[j] * s) * dz;
        }
        out << iteration << '\t' << std::setprecision(16) << k << '\t' << re << '\t' << im
            << '\t' << re * re + im * im << '\t' << std::atan2(im, re) << '\n';
    }
    out << '\n';
}

void MultiEnergyNuOsc::write_energy_diagnostics(int iteration) const
{
    std::ofstream out(("energy_" + std::to_string(iteration) + ".dat").c_str(), std::ios::trunc);
    out << "# sector E omega Psur |S| initial_e initial_x final_e final_x G(E)\n";
    out << std::setprecision(16);
    for (int e = 0; e < nm; ++e)
    {
        double ee = 0.0, xr = 0.0, xi = 0.0;
        for (int v = 0; v < nv; ++v)
            for (int j = 0; j < nz; ++j)
            {
                const std::size_t n = idx(e, v, j);
                ee += state.ee[n] * dz * dv;
                xr += state.ex_re[n] * dz * dv;
                xi += state.ex_im[n] * dz * dv;
            }
        const double spatial_length = z1 - z0;
        const double psur = ee / (mode_weight[e] * spatial_length);
        const double coh = 2.0 * std::hypot(xr, xi) /
                           (std::max(std::fabs(mode_weight[e]), 1.0e-300) * spatial_length);
        const double fe = initial_e_spectrum[e], fx = initial_x_spectrum[e];
        const double final_e = fx + (fe - fx) * psur;
        const double final_x = fe - (fe - fx) * psur;
        const double gE = e < ne ? fe - fx : fx - fe;
        out << (e < ne ? 1 : -1) << '\t' << energy[e] << '\t' << omega[e] << '\t'
            << psur << '\t' << coh << '\t' << fe << '\t' << fx << '\t'
            << final_e << '\t' << final_x << '\t' << gE << '\n';
    }
}

void MultiEnergyNuOsc::write_energy_angle(int iteration) const
{
    std::ofstream out(("energy_angle_" + std::to_string(iteration) + ".dat").c_str(), std::ios::trunc);
    out << "# sector E omega vz P3 |S|\n" << std::setprecision(16);
    for (int e = 0; e < nm; ++e)
        for (int v = 0; v < nv; ++v)
        {
            double p3 = 0.0, xr = 0.0, xi = 0.0;
            const double trace = mode_weight[e] * angular_profile[static_cast<std::size_t>(e) * nv + v];
            for (int j = 0; j < nz; ++j)
            {
                const std::size_t n = idx(e, v, j);
                p3 += (state.ee[n] - state.xx[n]) / trace;
                xr += state.ex_re[n] / trace;
                xi += state.ex_im[n] / trace;
            }
            out << (e < ne ? 1 : -1) << '\t' << energy[e] << '\t' << omega[e] << '\t'
                << velocity[v] << '\t' << p3 / nz << '\t' << 2.0 * std::hypot(xr, xi) / nz << '\n';
        }
}

void MultiEnergyNuOsc::analyse(int iteration, bool first)
{
    write_survival(iteration, first);
    write_conservation(iteration, first);
    write_power(iteration, first);
}

void MultiEnergyNuOsc::write_checkpoint(int iteration) const
{
    std::ofstream out("state.tmp", std::ios::binary | std::ios::trunc);
    const std::uint64_t magic = 0x434f53454e554d45ULL; // "COSENUME"
    out.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char *>(&iteration), sizeof(iteration));
    out.write(reinterpret_cast<const char *>(&nz), sizeof(nz));
    out.write(reinterpret_cast<const char *>(&nv), sizeof(nv));
    out.write(reinterpret_cast<const char *>(&ne), sizeof(ne));
    std::vector<double> row(static_cast<std::size_t>(4) * nz);
    for (int e = 0; e < nm; ++e)
        for (int v = 0; v < nv; ++v)
        {
            for (int j = 0; j < nz; ++j)
            {
                const std::size_t n = idx(e, v, j);
                const std::size_t q = static_cast<std::size_t>(4) * j;
                row[q] = state.ee[n]; row[q + 1] = state.xx[n];
                row[q + 2] = state.ex_re[n]; row[q + 3] = state.ex_im[n];
            }
            out.write(reinterpret_cast<const char *>(row.data()),
                      static_cast<std::streamsize>(row.size() * sizeof(double)));
        }
    if (!out)
    {
        std::cerr << "Could not write the complete multi-energy checkpoint." << std::endl;
        return;
    }
    out.close();
    std::rename("state.tmp", "state.bin");
}

int MultiEnergyNuOsc::read_checkpoint()
{
    std::ifstream in("state.bin", std::ios::binary);
    std::uint64_t magic = 0; int iteration = 0, file_nz = 0, file_nv = 0, file_ne = 0;
    in.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char *>(&iteration), sizeof(iteration));
    in.read(reinterpret_cast<char *>(&file_nz), sizeof(file_nz));
    in.read(reinterpret_cast<char *>(&file_nv), sizeof(file_nv));
    in.read(reinterpret_cast<char *>(&file_ne), sizeof(file_ne));
    if (!in || magic != 0x434f53454e554d45ULL || file_nz != nz || file_nv != nv || file_ne != ne)
    {
        std::cerr << "state.bin is not a compatible multi-energy checkpoint." << std::endl;
        std::exit(1);
    }
    std::vector<double> row(static_cast<std::size_t>(4) * nz);
    for (int e = 0; e < nm; ++e)
        for (int v = 0; v < nv; ++v)
        {
            in.read(reinterpret_cast<char *>(row.data()),
                    static_cast<std::streamsize>(row.size() * sizeof(double)));
            for (int j = 0; j < nz; ++j)
            {
                const std::size_t n = idx(e, v, j);
                const std::size_t q = static_cast<std::size_t>(4) * j;
                state.ee[n] = row[q]; state.xx[n] = row[q + 1];
                state.ex_re[n] = row[q + 2]; state.ex_im[n] = row[q + 3];
            }
        }
    if (!in)
    {
        std::cerr << "state.bin is incomplete." << std::endl;
        std::exit(1);
    }
    update_boundaries(state);
    return iteration + 1;
}

inline int run_multienergy(const Params &pars, bool restart)
{
#if defined(FV)
    std::cerr << "Multi-energy evolution currently uses the finite-difference scheme; set SCHEME=fd." << std::endl;
    return 1;
#else
    MultiEnergyNuOsc simulation(pars);
    int first_iteration = 1;
    // Reconstruct the deterministic initial normalization without storing a
    // second full four-dimensional state on disk or in memory.
    simulation.initialize();
    if (restart)
        first_iteration = simulation.read_checkpoint();
    else
    {
        simulation.analyse(0, true);
        simulation.dump(0);
    }
    std::cout << "Starting time = " << first_iteration << "\n";
    const auto start = std::chrono::steady_clock::now();
    int next_percent = restart ? static_cast<int>(first_iteration * 100.0 / (simulation.n_iter - 1)) + 1 : 1;
    for (int iteration = first_iteration; iteration < simulation.n_iter; ++iteration)
    {
        simulation.step_rk4();
        if (iteration % simulation.analysis_every == 0)
            simulation.analyse(iteration, false);
        if (iteration % simulation.dump_interval == 0)
            simulation.dump(iteration);
        const int completed = static_cast<int>(iteration * 100.0 / (simulation.n_iter - 1));
        while (next_percent <= completed && next_percent < 100)
        {
            std::cout << " " << next_percent++ << " %";
            if (simulation.average_p_error() > 0.1)
                std::cout << "  [WARNING: total box-averaged <dP> = "
                          << std::setprecision(4) << simulation.average_p_error()
                          << " exceeds 0.1 (10%)]";
            std::cout << std::endl;
        }
        if (iteration % std::max(1, simulation.n_iter / 10) == 0)
            simulation.write_checkpoint(iteration);
    }
    if ((simulation.n_iter - 1) % simulation.analysis_every != 0)
        simulation.analyse(simulation.n_iter - 1, false);
    simulation.dump(simulation.n_iter - 1);
    simulation.write_checkpoint(simulation.n_iter - 1);
    std::cout << " 100 %";
    if (simulation.average_p_error() > 0.1)
        std::cout << "  [WARNING: total box-averaged <dP> = "
                  << std::setprecision(4) << simulation.average_p_error()
                  << " exceeds 0.1 (10%)]";
    std::cout << "\n\nSIMULATION COMPLETED\n";
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    std::ofstream time_file("time.txt", std::ios::trunc);
    time_file << elapsed.count() << '\n';
    std::cout << "elapsed time: " << elapsed.count() << "s\n";
    return 0;
#endif
}

#endif // __MULTIENERGY__
