#if not defined(__NUOSC__)
#define __NUOSC__
/*---------------------------------------------------------------------------*/

typedef double real;

class NuOsc
{
public:
    real phy_time, dt;

    // all field variables...
    real *Z, *vw, *vz;
    real *Hm;
    FieldVar *v_stat, *v_rhs, *v_pre, *v_cor;
#ifdef COLL_OSC_ON
    // Reused by every RHS evaluation.  Keeping these moments with the solver
    // avoids allocating (and formerly leaking) sixteen nz-sized arrays during
    // each Runge--Kutta stage.
    FieldVar *Irho, *Jrho;
#endif
    Flux *flux;
    Profile *G0;

    int nz;  // Dim of z  (the last dimension, the one with derivatives. Cell-center grid used.)
    int nvz; // Dim of vz (Vertex-center grid used.)
    int gz;  // Width of z-buffer zone.
    int size;
    real ko;

    real dz, dv;
    real vz0, vz1;
    real z0, z1;
    real CFL;

    real signu;
    real siganu;
    real alpha;
    real eln_0;
    real beln_0;

    real theta;
    real ct;
    real st;
    real mu;
    real pmo; // 1.0 for normal mass ordering, -1.0 for inverted mass ordering, 0.0 for no vacuum term

    real perturbation_size;
    real seed_width;
    real k_max;
    int k_bins;
    bool random_seeds;
    bool exact_matter_rotation;
    std::string angular_profile;

    std::string SCHEME;
    std::ofstream outf;

    NuOsc(struct Params pars) : phy_time(0.)
    {
        SCHEME = pars.SCHEME;
        vz0 = pars.v0;
        vz1 = pars.v1;
        z0 = pars.z0;
        z1 = pars.z1;

        nz = pars.nz;
        nvz = pars.nvz;
        CFL = pars.CFL;
        gz = pars.gz; // Width of z-buffer zone. 4 for 2nd-order of d/dz.

        ko = 1.0e-1;

        perturbation_size = pars.perturbation_size;
        seed_width = pars.seed_width;
        k_max = pars.k_max;
        k_bins = pars.k_bins;
        signu = pars.neutrino_width;
        siganu = pars.antineutrino_width;
        alpha = pars.alpha;
        random_seeds = pars.random_seeds;
        exact_matter_rotation = pars.exact_matter_rotation;
        angular_profile = pars.angular_profile;

        size = (nz + 2 * gz) * (nvz);
        Z = new double[nz];
        vz = new double[nvz];
        vw = new real[nvz]; // quadrature of v

        v_stat = new FieldVar(size);
        v_rhs = new FieldVar(size);
        v_pre = new FieldVar(size);
        v_cor = new FieldVar(size);
#ifdef COLL_OSC_ON
        Irho = new FieldVar(nz);
        Jrho = new FieldVar(nz);
#endif
        flux = new Flux(size);
        G0 = new Profile(size);

        Hm = new double[nz];

        dz = (z1 - z0) / nz;
        dv = (vz1 - vz0) / (nvz); // cell-center
        dt = CFL * dz; // / vz1;


        for (int i = 0; i < nz; i++)
        {
            Z[i] = z0 + (i + 0.5) * dz;
        }

        for (int i = 0; i < nvz; i++)
        {
            vz[i] = vz0 + (i + 0.5) * dv;
            vw[i] = 1.0;
        }
        // The validation profile is an exact two-stream quadrature at v_z=+/-1.
        // All ordinary student configurations retain the cell-centred grid.
        if (angular_profile == "two_beam" || angular_profile == "symmetric_two_beam"
            || angular_profile == "soumya_two_beam")
        {
            vz[0] = -1.0;
            vz[1] = 1.0;
        }

        for (int i = 0; i < nz; i++)
        {
            // Hm[i] = -1.0e-3 * (Z[i] + 600.0); // non-homogeneous choice

            //// units of mu2 i.e. cm-2
            double mslope = pars.matter_slope;
            //// choice of lam0 should be such that any at any z, Hmat is comparable to ~mu and >> omega
            double lam0 = pars.matter_offset;
            // Hm[i] = mslope * (Z[i] + 600.0/2.0); // non-homogeneous choice
            Hm[i] = lam0 + mslope * (Z[i] - z0); // more transparent form, Hm \in [lam0, lam0 + m L]

            // Hm[i] = 0.0 * 2.0; // homogeneous choice with lambda=2mu
            // Hm[i] = -1.0;   // homogeneous choice with lambda=mu 
        }

        std::cout << draw(50, "#") << std::endl;
        std::cout << std::setw(35)
                  << "Simulation state created" << std::endl
                  << draw(50, ".") << std::endl
                  << "\t"
                  << "SCHEME: " << SCHEME << std::endl
                  << "\t"
                  << "z : (" << z0 << ", " << z1 << ")" << std::endl
                  << "\t"
                  << "vz: (" << vz[0] << ", " << vz[nvz - 1] << ")" << std::endl
                  << "\t"
                  << "(Nvz, Nz): (" << nvz << ", " << nz << ")" << std::endl
                  << "\t"
                  << "gz: " << gz << std::endl
                  << "\t"
                  << "CFL: " << CFL
                  << "\n\t"
                  << "perturbation_size: " << perturbation_size
                  << "\n\t"
                  << "power spectrum: -" << k_max << " <= k <= " << k_max
                  << " (" << k_bins << " bins)"
                  << std::endl;
        std::cout << draw(50, "#") << std::endl;
    }
    ~NuOsc()
    {
        delete[] vz;
        delete[] Z;
        delete v_stat;
        delete v_rhs;
        delete v_pre;
        delete v_cor;
#ifdef COLL_OSC_ON
        delete Irho;
        delete Jrho;
#endif
        delete flux;
        delete G0;
        delete[] Hm;
    }

    void set_vac_pars(double pmo_, double omega_, double theta_)
    {
        pmo = pmo_; // 1.0 for normal mass ordering, -1.0 for inverted mass ordering, 0.0 for no vacuum term
        theta = theta_ * M_PI / 180.0;
        ct = (1.0 / 2.0) * omega_ * cos(2.0 * theta);
        st = (1.0 / 2.0) * omega_ * sin(2.0 * theta);
    }

    void set_collective_pars(double mu_)
    {
        mu = mu_;
    }

    inline unsigned int idx(const int i, const int j)
    {
        return i * (nz + 2 * gz) + (j + gz);
    }

    inline int v_idx(double v)
    {
        return (int)((v - vz0) / dv - 0.5);
    }

    inline int z_idx(double z)
    {
        return (int)((z - z0) / dz - 0.5);
    }

    // Simulation specific methods.
    // Edit as per requirement
    void initialize();

    // Editing these methods are not recomended
    void updateBufferZone(FieldVar *in);
    void step_rk4();
    void rotate_matter(FieldVar *, real);
    void WENO7(const double *, double *, double *);
    void calFlux(const FieldVar *, Flux *);
    void calRHS(FieldVar *out, const FieldVar *in);
    void vectorize(FieldVar *v0, const FieldVar *v1, const real a, const FieldVar *v2);
    void vectorize(FieldVar *v0, const FieldVar *v1, const real a, const FieldVar *v2, const FieldVar *v3);
    void copy_state(const FieldVar *, FieldVar *);

    // State outputs
    void dump_rho(const FieldVar *, const uint);

    // Binary file ops
    void write_state(unsigned int);
    void write_state0(const FieldVar *);
    int read_state();
    void read_state0(FieldVar *);
    void read_G0();

    // Analysis related methods.
    // Edit these functions as per the requirement.
    void cal_pol(const FieldVar *, Pol *);
    void cal_Mn(M *, const Pol *, unsigned int);
    void cal_Mn_moments(M *, const Pol *, unsigned int); // added
    void cal_PowerSpectrum(M *, const Pol *, int); // added
    void dcon(const Pol *, const Pol *, M *, int);
    void dcon_copy(const Pol *, const Pol *, M *, M *, M *, M *, M *, M *, M *, M *, M *, M *, M *, M *, int); // added
    // void analyse(const FieldVar *, const Pol *, uint, uint);
    void analyse(const FieldVar *, const FieldVar *, const Pol *, uint, uint);
    void surv_prob(const FieldVar *, const FieldVar *, const uint);
    void v_distr_of_surv_prob(const FieldVar *, const FieldVar *, const uint); // added

};

void NuOsc::copy_state(const FieldVar *ivstate, FieldVar *cpvstate)
{
    for (int i = 0; i < nvz; i++)
    {
        for (int j = 0; j < nz; j++)
        {
            cpvstate->ee[idx(i, j)] = ivstate->ee[idx(i, j)];
            cpvstate->xx[idx(i, j)] = ivstate->xx[idx(i, j)];
            cpvstate->ex_re[idx(i, j)] = ivstate->ex_re[idx(i, j)];
            cpvstate->ex_im[idx(i, j)] = ivstate->ex_im[idx(i, j)];

            cpvstate->bee[idx(i, j)] = ivstate->bee[idx(i, j)];
            cpvstate->bxx[idx(i, j)] = ivstate->bxx[idx(i, j)];
            cpvstate->bex_re[idx(i, j)] = ivstate->bex_re[idx(i, j)];
            cpvstate->bex_im[idx(i, j)] = ivstate->bex_im[idx(i, j)];
        }
    }
}

void NuOsc::updateBufferZone(FieldVar *in)
{

#ifdef PERIODIC_BC
#pragma omp parallel for
#pragma acc parallel loop collapse(2) independent //  default(present)
    for (int i = 0; i < nvz; i++)
    {
        for (int j = 0; j < gz; j++)
        {
            // lower side
            in->ee[idx(i, -j - 1)] = in->ee[idx(i, nz - j - 1)];
            in->xx[idx(i, -j - 1)] = in->xx[idx(i, nz - j - 1)];
            in->ex_re[idx(i, -j - 1)] = in->ex_re[idx(i, nz - j - 1)];
            in->ex_im[idx(i, -j - 1)] = in->ex_im[idx(i, nz - j - 1)];

            in->bee[idx(i, -j - 1)] = in->bee[idx(i, nz - j - 1)];
            in->bxx[idx(i, -j - 1)] = in->bxx[idx(i, nz - j - 1)];
            in->bex_re[idx(i, -j - 1)] = in->bex_re[idx(i, nz - j - 1)];
            in->bex_im[idx(i, -j - 1)] = in->bex_im[idx(i, nz - j - 1)];

            // upper side
            in->ee[idx(i, nz + j)] = in->ee[idx(i, j)];
            in->xx[idx(i, nz + j)] = in->xx[idx(i, j)];
            in->ex_re[idx(i, nz + j)] = in->ex_re[idx(i, j)];
            in->ex_im[idx(i, nz + j)] = in->ex_im[idx(i, j)];

            in->bee[idx(i, nz + j)] = in->bee[idx(i, j)];
            in->bxx[idx(i, nz + j)] = in->bxx[idx(i, j)];
            in->bex_re[idx(i, nz + j)] = in->bex_re[idx(i, j)];
            in->bex_im[idx(i, nz + j)] = in->bex_im[idx(i, j)];
        }
    }
#endif

#ifdef OPEN_BC
#pragma omp parallel for
#pragma acc parallel loop collapse(2) indedependent // default(present)
    for (int i = 0; i < nvz; i++)
    {
        for (int j = 0; j < gz; j++)
        {
            // lower side
            in->ee[idx(i, -j - 1)] = in->ee[idx(i, 0)];
            in->xx[idx(i, -j - 1)] = in->xx[idx(i, 0)];
            in->ex_re[idx(i, -j - 1)] = in->ex_re[idx(i, 0)];
            in->ex_im[idx(i, -j - 1)] = in->ex_im[idx(i, 0)];

            in->bee[idx(i, -j - 1)] = in->bee[idx(i, 0)];
            in->bxx[idx(i, -j - 1)] = in->bxx[idx(i, 0)];
            in->bex_re[idx(i, -j - 1)] = in->bex_re[idx(i, 0)];
            in->bex_im[idx(i, -j - 1)] = in->bex_im[idx(i, 0)];

            // upper side, estimated simply by extrapolation
            in->ee[idx(i, nz + j)] = in->ee[idx(i, nz + j - 1)] * 2 - in->ee[idx(i, nz + j - 2)];
            in->xx[idx(i, nz + j)] = in->xx[idx(i, nz + j - 1)] * 2 - in->xx[idx(i, nz + j - 2)];
            in->ex_re[idx(i, nz + j)] = in->ex_re[idx(i, nz + j - 1)] * 2 - in->ex_re[idx(i, nz + j - 2)];
            in->ex_im[idx(i, nz + j)] = in->ex_im[idx(i, nz + j - 1)] * 2 - in->ex_im[idx(i, nz + j - 2)];

            in->bee[idx(i, nz + j)] = in->bee[idx(i, nz + j - 1)] * 2 - in->bee[idx(i, nz + j - 2)];
            in->bxx[idx(i, nz + j)] = in->bxx[idx(i, nz + j - 1)] * 2 - in->bxx[idx(i, nz + j - 2)];
            in->bex_re[idx(i, nz + j)] = in->bex_re[idx(i, nz + j - 1)] * 2 - in->bex_re[idx(i, nz + j - 2)];
            in->bex_im[idx(i, nz + j)] = in->bex_im[idx(i, nz + j - 1)] * 2 - in->bex_im[idx(i, nz + j - 2)];
        }
    }
#endif
}

/* v0 = v1 + a * v2 */
void NuOsc::vectorize(FieldVar *v0, const FieldVar *v1, const real a, const FieldVar *v2)
{
#pragma omp parallel for collapse(2)
#pragma acc parallel loop collapse(2) independent // default(present)
    for (int i = 0; i < nvz; i++)
        for (int j = 0; j < nz; j++)
        {
            int k = idx(i, j);
            v0->ee[k] = v1->ee[k] + a * v2->ee[k];
            v0->xx[k] = v1->xx[k] + a * v2->xx[k];
            v0->ex_re[k] = v1->ex_re[k] + a * v2->ex_re[k];
            v0->ex_im[k] = v1->ex_im[k] + a * v2->ex_im[k];
            v0->bee[k] = v1->bee[k] + a * v2->bee[k];
            v0->bxx[k] = v1->bxx[k] + a * v2->bxx[k];
            v0->bex_re[k] = v1->bex_re[k] + a * v2->bex_re[k];
            v0->bex_im[k] = v1->bex_im[k] + a * v2->bex_im[k];
        }
}

// v0 = v1 + a * ( v2 + v3 )
void NuOsc::vectorize(FieldVar *v0, const FieldVar *v1, const real a, const FieldVar *v2, const FieldVar *v3)
{
#pragma omp parallel for collapse(2)
#pragma acc parallel loop collapse(2) independent // default(present)
    for (int i = 0; i < nvz; i++)
        for (int j = 0; j < nz; j++)
        {
            int k = idx(i, j);
            v0->ee[k] = v1->ee[k] + a * (v2->ee[k] + v3->ee[k]);
            v0->xx[k] = v1->xx[k] + a * (v2->xx[k] + v3->xx[k]);
            v0->ex_re[k] = v1->ex_re[k] + a * (v2->ex_re[k] + v3->ex_re[k]);
            v0->ex_im[k] = v1->ex_im[k] + a * (v2->ex_im[k] + v3->ex_im[k]);
            v0->bee[k] = v1->bee[k] + a * (v2->bee[k] + v3->bee[k]);
            v0->bxx[k] = v1->bxx[k] + a * (v2->bxx[k] + v3->bxx[k]);
            v0->bex_re[k] = v1->bex_re[k] + a * (v2->bex_re[k] + v3->bex_re[k]);
            v0->bex_im[k] = v1->bex_im[k] + a * (v2->bex_im[k] + v3->bex_im[k]);
        }
}

void NuOsc::step_rk4()
{
#if defined(MAT_OSC_ON)
    // Strang-split the stiff diagonal matter Hamiltonian.  Its action is a
    // pure complex phase, so applying it analytically avoids the artificial
    // polarization-length damping produced by explicit RK4 when lambda*dt
    // is moderately large.  The option is off by default and is intended
    // for strongly inhomogeneous-matter runs.
    if (exact_matter_rotation)
        rotate_matter(v_stat, 0.5 * dt);
#endif

    // Step-1
    updateBufferZone(v_stat);
    calRHS(v_rhs, v_stat);
    vectorize(v_pre, v_stat, 0.5 * dt, v_rhs);

    // Step-2
    updateBufferZone(v_pre);
    calRHS(v_cor, v_pre);
    vectorize(v_rhs, v_rhs, 2.0, v_cor);
    vectorize(v_cor, v_stat, 0.5 * dt, v_cor);
    swap(&v_pre, &v_cor);

    // Step-3
    updateBufferZone(v_pre);
    calRHS(v_cor, v_pre);
    vectorize(v_rhs, v_rhs, 2.0, v_cor);
    vectorize(v_cor, v_stat, dt, v_cor);
    swap(&v_pre, &v_cor);

    // Step-4
    updateBufferZone(v_pre);
    calRHS(v_cor, v_pre);
    vectorize(v_pre, v_stat, 1.0 / 6.0 * dt, v_cor, v_rhs);
    swap(&v_pre, &v_stat);

#if defined(MAT_OSC_ON)
    if (exact_matter_rotation)
        rotate_matter(v_stat, 0.5 * dt);
#endif

    phy_time += dt;
}

void NuOsc::rotate_matter(FieldVar *state, real interval)
{
#pragma omp parallel for collapse(2)
    for (int i = 0; i < nvz; i++)
        for (int j = 0; j < nz; j++)
        {
            const int k = idx(i, j);
            const real phase = Hm[j] * interval;
            const real c = std::cos(phase);
            const real s = std::sin(phase);

            const real re = state->ex_re[k];
            const real im = state->ex_im[k];
            state->ex_re[k] = c * re + s * im;
            state->ex_im[k] = c * im - s * re;

            const real bre = state->bex_re[k];
            const real bim = state->bex_im[k];
            state->bex_re[k] = c * bre - s * bim;
            state->bex_im[k] = c * bim + s * bre;
        }
}

/*---------------------------------------------------------------------------*/

#endif // __NUOSC__

/*---------------------------------------------------------------------------*/
