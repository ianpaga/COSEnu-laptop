#if not defined(__INITIALIZE__)
#define __INITIALIZE__

/*---------------------------------------------------------------------------*/

void NuOsc::initialize()
{
/*
    Initialize the components of \rho and \bar{\rho}
    here.
*/
    // int amax=nz/2/10;
    // real *phi = new real[nz/10+1];
    // int amax=nz/2/250; // adjust later
    // real *phi = new real[nz/250+1]; // adjust later
    int amax=nz/2/5;
    real *phi = new real[nz/5+1]; 
    real L=z0-z1;
    for(int k=-amax;k<=amax;k++){
        phi[k+amax]=2.0*M_PI*rand()/RAND_MAX;
    }

    std::ofstream g_file("G0.bin",std::ofstream::out | std::ofstream::binary);
    if(!g_file)
    {
        std::cout << "Unable to open " << "G0.bin" << " file from NuOsc::initialise." 
        << "Will not be storing initial angular profiles.\n";
    }

    for (int j = 0; j < nz; j++)
    {
        double e_rand = random_amp(perturbation_size);
        double e_rand_ = sqrt(1.0 - e_rand * e_rand);

        real tmpr=0.0;
        real tmpi=0.0;
        for(int k=-amax;k<amax;k++)
        {
            if(k!=0)
            {
                double B = 1e-3; //1e-7;
                tmpr = tmpr + B / abs(k) * cos(2.0*M_PI*k*Z[j]/L+phi[k+amax]);
                tmpi = tmpi + B / abs(k) * sin(2.0*M_PI*k*Z[j]/L+phi[k+amax]);
            }
        }
        real tmp2=sqrt(1.0-tmpr*tmpr-tmpi*tmpi);

        for (int i = 0; i < nvz; i++)
        {
            if (angular_profile == "two_beam")
            {
                G0->G[idx(i, j)] = (i == 1) ? 1.0 / dv : 0.0;
                G0->bG[idx(i, j)] = (i == 0) ? alpha / dv : 0.0;
            }
            else if (angular_profile == "symmetric_two_beam")
            {
                // Equal populations in both v_z=+/-1 directions.  The 1/2
                // weights keep integral G dv=1 and integral Gbar dv=alpha,
                // exactly matching the normalization of the continuous
                // angular profiles used by COSEnu.
                G0->G[idx(i, j)] = 0.5 / dv;
                G0->bG[idx(i, j)] = 0.5 * alpha / dv;
            }
            else if (angular_profile == "soumya_two_beam")
            {
                // Soumya's convention quotes g_nue=1 and g_anue=alpha for
                // each of the v_z=-1 and +1 modes.  Keep that per-beam
                // normalization explicit rather than absorbing its factor
                // of two into mu.
                G0->G[idx(i, j)] = 1.0 / dv;
                G0->bG[idx(i, j)] = alpha / dv;
            }
            else
            {
                G0->G[idx(i, j)] = g(vz[i], 1.0, signu);
                G0->bG[idx(i, j)] = alpha * g(vz[i], 1.0, siganu);
            }

            if(random_seeds)
            {
                // // random seeds
                // v_stat->ee[idx(i, j)]    = 0.5 * G0->G[idx(i, j)] * (1.0 + e_rand_); 
                // v_stat->xx[idx(i, j)]    = 0.5 * G0->G[idx(i, j)] * (1.0 - e_rand_);
                // v_stat->ex_re[idx(i, j)] = 0.5 * G0->G[idx(i, j)] * (0.0 + e_rand);
                // v_stat->ex_im[idx(i, j)] = -0.0;

                // v_stat->bee[idx(i, j)]    = 0.5 * G0->bG[idx(i, j)] * (1.0 + e_rand_); 
                // v_stat->bxx[idx(i, j)]    = 0.5 * G0->bG[idx(i, j)] * (1.0 - e_rand_);
                // v_stat->bex_re[idx(i, j)] = 0.5 * G0->bG[idx(i, j)] * (0.0 + e_rand);
                // v_stat->bex_im[idx(i, j)] = 0.0;

                // // k-superposition of Fourier modes
                v_stat->ee[idx(i,j)]    = 0.5 * G0->G[idx(i, j)] * (1.0 + tmp2);
                v_stat->xx[idx(i,j)]    = 0.5 * G0->G[idx(i, j)] * (1.0 - tmp2);
                v_stat->ex_re[idx(i,j)] = 0.5 * G0->G[idx(i, j)] * tmpr;
                v_stat->ex_im[idx(i,j)] = 0.5 * G0->G[idx(i, j)] * tmpi;

                v_stat->bee[idx(i,j)]    = 0.5 * G0->bG[idx(i, j)] * (1.0 + tmp2);
                v_stat->bxx[idx(i,j)]    = 0.5 * G0->bG[idx(i, j)] * (1.0 - tmp2);
                v_stat->bex_re[idx(i,j)] = 0.5 * G0->bG[idx(i, j)] * tmpr;
                v_stat->bex_im[idx(i,j)] = -0.5 * G0->bG[idx(i, j)] * tmpi;
            }
            else
            {
                // // central seeds
                // v_stat->ee[idx(i, j)]    = 0.5 * G0->G[idx(i, j)] * (1.0 + eps_(Z[j], 0.0, perturbation_size)); 
                // v_stat->xx[idx(i, j)]    = 0.5 * G0->G[idx(i, j)] * (1.0 - eps_(Z[j], 0.0, perturbation_size));
                // v_stat->ex_re[idx(i, j)] = 0.5 * G0->G[idx(i, j)] * (0.0 + eps(Z[j], 0.0, perturbation_size));
                // v_stat->ex_im[idx(i, j)] = -0.0;

                // v_stat->bee[idx(i, j)]    = 0.5 * G0->bG[idx(i, j)] * (1.0 + eps_(Z[j], 0.0, perturbation_size)); 
                // v_stat->bxx[idx(i, j)]    = 0.5 * G0->bG[idx(i, j)] * (1.0 - eps_(Z[j], 0.0, perturbation_size));
                // v_stat->bex_re[idx(i, j)] = 0.5 * G0->bG[idx(i, j)] * (0.0 + eps(Z[j], 0.0, perturbation_size));
                // v_stat->bex_im[idx(i, j)] = 0.0;

                double perc = 1.0; //0.70;

                // central seeds
                v_stat->ee[idx(i, j)]    = 0.5 * G0->G[idx(i, j)] * (1.0 + perc * eps_(Z[j], 0.0, perturbation_size, seed_width)); 
                v_stat->xx[idx(i, j)]    = 0.5 * G0->G[idx(i, j)] * (1.0 - perc * eps_(Z[j], 0.0, perturbation_size, seed_width));
                v_stat->ex_re[idx(i, j)] = 0.5 * G0->G[idx(i, j)] * (0.0 + eps(Z[j], 0.0, perturbation_size, seed_width));
                v_stat->ex_im[idx(i, j)] = -0.0;

                v_stat->bee[idx(i, j)]    = 0.5 * G0->bG[idx(i, j)] * (1.0 + perc * eps_(Z[j], 0.0, perturbation_size, seed_width)); 
                v_stat->bxx[idx(i, j)]    = 0.5 * G0->bG[idx(i, j)] * (1.0 - perc * eps_(Z[j], 0.0, perturbation_size, seed_width));
                v_stat->bex_re[idx(i, j)] = 0.5 * G0->bG[idx(i, j)] * (0.0 + eps(Z[j], 0.0, perturbation_size, seed_width));
                v_stat->bex_im[idx(i, j)] = 0.0;

            }
            
            g_file.write((char *)&G0->G [idx(i, j)], sizeof(double)); 
            g_file.write((char *)&G0->bG[idx(i, j)], sizeof(double));
        }
    }
    updateBufferZone(v_stat);
    std::cout << "Simulation state initialized." << std::endl;
    g_file.close();
}

/*---------------------------------------------------------------------------*/

#endif // __INITIALIZE__

/*---------------------------------------------------------------------------*/
