#if not defined(__ANALYSIS__)
#define __ANALYSIS__
// void NuOsc::analyse(const FieldVar *ivstate, const Pol *P0, uint n, uint t)
void NuOsc::analyse(const FieldVar *ivstate, const FieldVar *ivstate0, const Pol *P0, uint n, uint t)
{
    /****************************************************************************************/
    /* Subroutine to check the deviation of conserved quantities.                           */
    /*                                                                                      */
    /* (1)---                                                                               */
    /*   Estimate maximal deviation of polarization,max{ep_i} for all ep_i belongs to       */
    /*   {P(z_j, v_i) - P0(z_j, v_i)}, i:0->nvz, j:0->nz.                                   */
    /*                                                                                      */
    /* (2)---                                                                               */
    /*   Evaluate M_0                                                                       */
    /****************************************************************************************/
    Pol *P = new Pol(size);
    M *M_0 = new M(0);

    cal_pol(ivstate, P); // Calculating the components of polarization.
    cal_Mn(M_0, P, 0);   // Calculating the components of M_0.Hvac

    M *MOM_0 = new M(0);
    M *MOM_1 = new M(0);
    M *MOM_2 = new M(0);
    M *MOM_3 = new M(0);
    M *MOM_4 = new M(0);
    M *MOM_5 = new M(0);
    M *MOM_6 = new M(0);
    M *MOM_7 = new M(0);
    M *MOM_8 = new M(0);
    M *MOM_9 = new M(0);
    M *MOM_10 = new M(0);

    // Calculating the components of MOM_i
    cal_Mn_moments(MOM_0, P, 0);   
    cal_Mn_moments(MOM_1, P, 1);   
    cal_Mn_moments(MOM_2, P, 2);   
    cal_Mn_moments(MOM_3, P, 3);   
    cal_Mn_moments(MOM_4, P, 4);   
    cal_Mn_moments(MOM_5, P, 5);   
    cal_Mn_moments(MOM_6, P, 6);   
    cal_Mn_moments(MOM_7, P, 7);   
    cal_Mn_moments(MOM_8, P, 8);   
    cal_Mn_moments(MOM_9, P, 9);   
    cal_Mn_moments(MOM_10, P, 10);

    // dcon(P, P0, M_0, t);
    dcon_copy(P, P0, M_0, MOM_0, MOM_1, MOM_2, MOM_3, MOM_4, MOM_5, MOM_6, MOM_7, MOM_8, MOM_9, MOM_10, t);

    cal_PowerSpectrum(M_0, P, t); // power spectrum 
    //v_distr_of_surv_prob(ivstate, ivstate0, t); // <Pee>(v), <bPee>(v)

    delete P;
    delete M_0;

    delete MOM_0;
    delete MOM_1;
    delete MOM_2;
    delete MOM_3;
    delete MOM_4;
    delete MOM_5;
    delete MOM_6;
    delete MOM_7;
    delete MOM_8;
    delete MOM_9;
    delete MOM_10;

}

void NuOsc::cal_pol(const FieldVar *inField, Pol *inP)
{
    /* Calculate the polarization vector(\vec {P}) of nu and anu and correspondig |\vec{P}|*/

    for (int i = 0; i < nvz; i++)
    {
        for (int j = 0; j < nz; j++)
        {
            const unsigned int ij = idx(i, j);
            const double neutrino_density = G0->G[ij];
            const double antineutrino_density = G0->bG[ij];
            if (fabs(neutrino_density) > 1.0e-300)
            {
                inP->P1[ij] = 2.0 * inField->ex_re[ij] / neutrino_density;
                inP->P2[ij] = -2.0 * inField->ex_im[ij] / neutrino_density;
                inP->P3[ij] = (inField->ee[ij] - inField->xx[ij]) / neutrino_density;
            }
            else
            {
                inP->P1[ij] = inP->P2[ij] = inP->P3[ij] = 0.0;
            }
            inP->normP[ij] = fabs(sqrt(pow(inP->P1[ij], 2) + pow(inP->P2[ij], 2) + pow(inP->P3[ij], 2)));

            if (fabs(antineutrino_density) > 1.0e-300)
            {
                inP->bP1[ij] = 2.0 * inField->bex_re[ij] / antineutrino_density;
                inP->bP2[ij] = 2.0 * inField->bex_im[ij] / antineutrino_density;
                inP->bP3[ij] = (inField->bee[ij] - inField->bxx[ij]) / antineutrino_density;
            }
            else
            {
                inP->bP1[ij] = inP->bP2[ij] = inP->bP3[ij] = 0.0;
            }
            inP->normbP[ij] = fabs(sqrt(pow(inP->bP1[ij], 2) + pow(inP->bP2[ij], 2) + pow(inP->bP3[ij], 2)));
        }
    }
}

/*---------------------------------------------------------------------------*/

void NuOsc::cal_Mn(M *inMn, const Pol *inP, unsigned int n)
{
    /*
        Subroutine to calculate the moments. L(vz[i], n) returns the legendre polynomial
        of order n (up to n=5) for the mode vz[i]
    */
    for (int i = 0; i < nvz; i++)
    {
        for (int j = 0; j < nz; j++)
        {
            inMn->e1 += L(vz[i], n) * (G0->G[idx(i, j)] * inP->P1[idx(i, j)] - G0->bG[idx(i, j)] * inP->bP1[idx(i, j)]) * dz * dv;
            inMn->e2 += L(vz[i], n) * (G0->G[idx(i, j)] * inP->P2[idx(i, j)] - G0->bG[idx(i, j)] * inP->bP2[idx(i, j)]) * dz * dv;
            inMn->e3 += L(vz[i], n) * (G0->G[idx(i, j)] * inP->P3[idx(i, j)] - G0->bG[idx(i, j)] * inP->bP3[idx(i, j)]) * dz * dv;
        }
    }

    // inMn->norm = sqrt(pow(inMn->e1, 2) + pow(inMn->e2, 2) + pow(inMn->e3, 2));
    inMn->norm = - inMn->e1 * sin(2.0*theta) + inMn->e3 * cos(2.0*theta) ; //MRW: store M0 cdot Hvac
}

/*------------------------------------ testing moments ---------------------------------------*/

void NuOsc::cal_Mn_moments(M *inMn, const Pol *inP, unsigned int n)
{
    /*
        Subroutine to calculate the moments. L(vz[i], n) returns the legendre polynomial
        of order n (up to n=5) for the mode vz[i]
    */
    for (int i = 0; i < nvz; i++)
    {
        for (int j = 0; j < nz; j++)
        {
            inMn->e1 += L(vz[i], n) * (G0->G[idx(i, j)] * inP->P1[idx(i, j)] - G0->bG[idx(i, j)] * inP->bP1[idx(i, j)]) * dz * dv;
            inMn->e2 += L(vz[i], n) * (G0->G[idx(i, j)] * inP->P2[idx(i, j)] - G0->bG[idx(i, j)] * inP->bP2[idx(i, j)]) * dz * dv;
            inMn->e3 += L(vz[i], n) * (G0->G[idx(i, j)] * inP->P3[idx(i, j)] - G0->bG[idx(i, j)] * inP->bP3[idx(i, j)]) * dz * dv;
        }
    }

    inMn->norm = sqrt(pow(inMn->e1, 2) + pow(inMn->e2, 2) + pow(inMn->e3, 2));
}

/*------------------------------------- cal_PowerSpectrum--------------------------------------*/

// PUT BACK!!!

// void NuOsc::cal_PowerSpectrum(M *inMn, const Pol *inP, int t)
// {

//     /*
//         Computes the power spectrum using the Nuttall window.
//     */

//     // Periodicity is an artifact of the resolution, adjust for i t
//     double L = z0 - z1;
//     double k_start = - (nz/abs(L)) * M_PI; // -500/100 = -5, e.g. -5pi
//     double k_end   = + (nz/abs(L)) * M_PI; // +500/100 = +5, e.g. +5pi
//     //int k_bins = nz;
//     int k_bins = 1000;

//     std::vector<double> k_array(k_bins);
//     for (int m = 0; m < k_bins; m++) 
//     {
//         k_array[m] = k_start + m * (k_end - k_start) / (k_bins - 1);
//     }

//     std::ofstream outfile("power_spectrum.dat", std::ios::app);
//     if (outfile.tellp() == 0) 
//     {
//         outfile << "# [time, wavenumber k, Power Spectrum <tildeS>]" << std::endl;
//     }

//     // Nuttall window parameters
//     double a0 = 0.355768;
//     double a1 = 0.487396;
//     double a2 = 0.144232;
//     double a3 = 0.012604;

//     // Loop over each k value and calculate the power spectrum for each
//     for (double k : k_array) 
//     {
//         double re_total_integral = 0.0; 
//         double im_total_integral = 0.0;

//         for (int i = 0; i < nvz; i++) 
//         {
//             for (int j = 0; j < nz; j++) 
//             {
//                 // Compute (P1 - i * P2)
//                 double re_P1_minus_iP2 = inP->P1[idx(i, j)];
//                 double im_P1_minus_iP2 = -inP->P2[idx(i, j)];

//                 // Compute Nuttall window value
//                 double n = static_cast<double>(j);
//                 double nuttall_window = a0 
//                                       - a1 * cos((2 * M_PI * n) / (nz - 1))
//                                       + a2 * cos((4 * M_PI * n) / (nz - 1))
//                                       - a3 * cos((6 * M_PI * n) / (nz - 1));

//                 // Apply Nuttall window
//                 re_P1_minus_iP2 *= nuttall_window;
//                 im_P1_minus_iP2 *= nuttall_window;

//                 // Compute exp(-i * k * Z) for padded array Z (assuming Z has been zero-padded)
//                 double re_exp_term = cos(-k * (j < nz ? Z[j] : Z[nz - 1] + (j - nz + 1) * dz));
//                 double im_exp_term = sin(-k * (j < nz ? Z[j] : Z[nz - 1] + (j - nz + 1) * dz));

//                 // Update total integral for this k value
//                 re_total_integral += (G0->G[idx(i, j)] * re_P1_minus_iP2) * dz * dv * re_exp_term;
//                 im_total_integral += (G0->G[idx(i, j)] * im_P1_minus_iP2) * dz * dv * im_exp_term;
//             }
//         }

//         // Compute the norm of the result for this k
//         double norm_result = sqrt(re_total_integral * re_total_integral + im_total_integral * im_total_integral);

//         // Write t, k, and norm_result to the file using tabs
//         outfile << t << "\t" 
//                 << std::fixed << std::setprecision(20)
//                 << k << "\t"
//                 << norm_result << std::endl; 
//     }
//     outfile << std::endl;
//     outfile.close();
// }

void NuOsc::cal_PowerSpectrum(M *inMn, const Pol *inP, int t)
{
    /*
        Computes the complex Fourier spectrum using a Nuttall window.
        Outputs Re[F(k)], Im[F(k)], |F(k)|^2, and phase.
    */


    // Periodicity is an artifact of the resolution, adjust for i t
    // double L = z0 - z1;
    // double k_start = - (nz/abs(L)) * M_PI; // -500/100 = -5, e.g. -5pi
    // double k_end   = + (nz/abs(L)) * M_PI; // +500/100 = +5, e.g. +5pi
    // //int k_bins = nz; // very slow for high res
    // int k_bins = 500;

    double k_start = -k_max;
    double k_end   = +k_max;

    std::vector<double> k_array(k_bins);
    for (int m = 0; m < k_bins; m++)
        k_array[m] = k_start + m * (k_end - k_start) / (k_bins - 1);

    std::ofstream outfile("power_spectrum.dat", std::ios::app);
    if (outfile.tellp() == 0)
    {
        outfile << "# t   k   Re[F(k)]   Im[F(k)]   Power   Phase\n";
    }

    // Nuttall window coefficients
    const double a0 = 0.355768;
    const double a1 = 0.487396;
    const double a2 = 0.144232;
    const double a3 = 0.012604;

    // Loop over k values
    for (double k : k_array)
    {
        double re_total = 0.0;
        double im_total = 0.0;

        for (int i = 0; i < nvz; i++)
        {
            for (int j = 0; j < nz; j++)
            {
                // ---- Signal S = a + i b ----
                double a = G0->G[idx(i,j)] * inP->P1[idx(i,j)];
                double b = -G0->G[idx(i,j)] * inP->P2[idx(i,j)];

                // ---- Nuttall window ----
                double n = double(j);
                double W = a0
                         - a1 * cos(2.0 * M_PI * n / (nz - 1))
                         + a2 * cos(4.0 * M_PI * n / (nz - 1))
                         - a3 * cos(6.0 * M_PI * n / (nz - 1));

                a *= W;
                b *= W;

                // ---- exp(-i k z) ----
                double c = cos(k * Z[j]);
                double s = sin(k * Z[j]);

                // i left out zero-padding, ok, but maybe implement later

                // ---- Complex multiply: (a + ib)(c - is) ----
                re_total += ( a * c + b * s ) * dz * dv;
                im_total += ( b * c - a * s ) * dz * dv;
            }
        }

        // ---- Derived quantities ----
        double power = re_total * re_total + im_total * im_total; // should i take sqrt?
        double phase = atan2(im_total, re_total);

        // ---- Output ----
        outfile << t << "\t"
                << std::fixed << std::setprecision(16)
                << k << "\t"
                << re_total << "\t"
                << im_total << "\t"
                << power << "\t"
                << phase << "\n";
    }

    outfile << "\n";
    outfile.close();
}




// void NuOsc::cal_PowerSpectrum(M *inMn, const Pol *inP, int t)
// {
//     /*
//         Subroutine to calculate the power spectrum for a range of k from k_start to k_end.
//     */

//     // std::complex<double> i_complex(0.0e0, 1.0e0); // Imaginary unit
//     double k_start = -5.0e0 * M_PI;
//     double k_end = 5.0e0 * M_PI;
//     double L=z0-z1; 
//     int k_bins = 1000;
//     // int k_bins = int(5.0*abs(L)); // 10pi/k_bins = 2pi/L -> k_bins = 5L
//     std::vector<double> k_array(k_bins);

//     for (int m = 0; m < k_bins; m++) 
//     {
//         k_array[m] = k_start + m * (k_end - k_start) / (k_bins - 1);
//     }

//     std::ofstream outfile("power_spectrum.dat", std::ios::app);
//     if (outfile.tellp() == 0) 
//     {
//         outfile << "# [time, wavenumber k, Power Spectrum <tildeS>]" << std::endl;
//     }

//     // Loop over each k value and calculate the power spectrum for each
//     for (double k : k_array) 
//     {
//         // std::complex<double> total_integral(0.0e0, 0.0e0); // Reset for each k
//         double re_total_integral = 0.0e0; 
//         double im_total_integral = 0.0e0;

//         for (int i = 0; i < nvz; i++) 
//         {
//             for (int j = 0; j < nz; j++) 
//             {
//                 // Compute (P1 - i * P2)
//                 // std::complex<double> P1_minus_iP2 = inP->P1[idx(i, j)] - i_complex * inP->P2[idx(i, j)];
//                 double re_P1_minus_iP2 = + inP->P1[idx(i, j)];
//                 double im_P1_minus_iP2 = - inP->P2[idx(i, j)];

//                 // Compute exp(-i * k * Z)
//                 // std::complex<double> exp_term = std::exp(-i_complex * k * Z[j]);
//                 double re_exp_term = cos(-k * Z[j]);
//                 double im_exp_term = sin(-k * Z[j]);

//                 // Update total integral for this k value
//                 // total_integral += (G0->G[idx(i, j)] * P1_minus_iP2) * dz * dv * exp_term;
//                 re_total_integral += (G0->G[idx(i, j)] * re_P1_minus_iP2) * dz * dv * re_exp_term;
//                 im_total_integral += (G0->G[idx(i, j)] * im_P1_minus_iP2) * dz * dv * im_exp_term;

//             }
//         }

//         // Compute the norm of the result for this k
//         // double norm_result = std::abs(total_integral);
//         double norm_result = sqrt(re_total_integral*re_total_integral + im_total_integral*im_total_integral);

//         // Write t, k, and norm_result to the file using tabs
//         outfile << t << "\t" 
//                 << std::fixed << std::setprecision(20)
//                 << k << "\t"
//                 << norm_result << std::endl; 

//     }
//     outfile << std::endl;
//     outfile.close();
// }



/*---------------------------------------------------------------------------*/


void NuOsc::dcon_copy(const Pol *P, const Pol *P0, M *M_0, M *MOM_0, M *MOM_1, M *MOM_2, M *MOM_3, M *MOM_4, M *MOM_5, M *MOM_6, M *MOM_7, M *MOM_8, M *MOM_9, M *MOM_10, int t)
{
    /*
        Estimates the deviation of conserved quantities from their initial values and write them ot the
        file.
    */
    std::ofstream con_qty_ofstream; // To store deviation of conserved qtys.
    std::string con_qty_fname = "conserved_quantities.dat";
    if (t == 0)
    {
        con_qty_ofstream.open(con_qty_fname, std::ofstream::out | std::ofstream::trunc);
    }
    else
    {
        con_qty_ofstream.open(con_qty_fname, std::ofstream::out | std::ofstream::app);
    }
    if (!con_qty_ofstream)
    {
        std::cout << "Unable to open " << con_qty_fname << std::endl;
        return;
    }
    else
    {
        if (t == 0)
        {
            con_qty_ofstream << "# [time, dP_max(t), <dP>(t), <dbP(t), M0.Hvac, |MOM_0|, |MOM_1|, |MOM_2|, |MOM_3|, |MOM_4|, |MOM_5|, |MOM_6|, |MOM_7|, |MOM_8|, |MOM_9|, |MOM_10|]\n";
        }
        double dP = 0.0;
        double dbP = 0.0;
        double Nee0 = 0.0;
        double Nbee0 = 0.0;
        double avdP = 0.0;
        double avbdP = 0.0;

        for (int i = 0; i < nvz; i++)
        {
            for (int j = 0; j < nz; j++)
            {
                int ij = idx(i, j);

                dP = (dP >= fabs(P->normP[ij] - P0->normP[ij])) ? dP : fabs(P->normP[ij] - P0->normP[ij]);
                dbP = (dbP >= fabs(P->normbP[ij] - P0->normbP[ij])) ? dbP : fabs(P->normbP[ij] - P0->normbP[ij]);

                avdP += fabs(P->normP[ij] - P0->normP[ij]) * G0->G[ij];
                avbdP += fabs(P->normbP[ij] - P0->normbP[ij]) * G0->bG[ij];

                Nee0 += G0->G[ij];
                Nbee0 += G0->bG[ij];
            }
        }
        con_qty_ofstream << t << "\t"
                         << std::fixed << std::setprecision(20)
                         << ((dP >= dbP) ? dP : dbP) << "\t"
                         << avdP / Nee0 << "\t"
                         << avbdP / Nbee0 << "\t"
                         << M_0->norm << "\t"
                         << MOM_0->norm << "\t"
                         << MOM_1->norm << "\t"
                         << MOM_2->norm << "\t"
                         << MOM_3->norm << "\t"
                         << MOM_4->norm << "\t"
                         << MOM_5->norm << "\t"
                         << MOM_6->norm << "\t"
                         << MOM_7->norm << "\t"
                         << MOM_8->norm << "\t"
                         << MOM_9->norm << "\t"
                         << MOM_10->norm << endl;

        con_qty_ofstream.close();
    }
}

/*---------------------------------------------------------------------------*/


void NuOsc::dcon(const Pol *P, const Pol *P0, M *M_0, int t)
{
    /*
        Estimates the deviation of conserved quantities from their initial values and write them ot the
        file.
    */
    std::ofstream con_qty_ofstream; // To store deviation of conserved qtys.
    std::string con_qty_fname = "conserved_quantities.dat";
    if (t == 0)
    {
        con_qty_ofstream.open(con_qty_fname, std::ofstream::out | std::ofstream::trunc);
    }
    else
    {
        con_qty_ofstream.open(con_qty_fname, std::ofstream::out | std::ofstream::app);
    }
    if (!con_qty_ofstream)
    {
        std::cout << "Unable to open " << con_qty_fname << std::endl;
        return;
    }
    else
    {
        if (t == 0)
        {
            con_qty_ofstream << "# [time, dP_max(t), <dP>(t), <dbP(t), |M0|]\n";
        }
        double dP = 0.0;
        double dbP = 0.0;
        double Nee0 = 0.0;
        double Nbee0 = 0.0;
        double avdP = 0.0;
        double avbdP = 0.0;

        for (int i = 0; i < nvz; i++)
        {
            for (int j = 0; j < nz; j++)
            {
                int ij = idx(i, j);

                dP = (dP >= fabs(P->normP[ij] - P0->normP[ij])) ? dP : fabs(P->normP[ij] - P0->normP[ij]);
                dbP = (dbP >= fabs(P->normbP[ij] - P0->normbP[ij])) ? dbP : fabs(P->normbP[ij] - P0->normbP[ij]);

                avdP += fabs(P->normP[ij] - P0->normP[ij]) * G0->G[ij];
                avbdP += fabs(P->normbP[ij] - P0->normbP[ij]) * G0->bG[ij];

                Nee0 += G0->G[ij];
                Nbee0 += G0->bG[ij];
            }
        }
        con_qty_ofstream << t << "\t"
                         << std::fixed << std::setprecision(20)
                         << ((dP >= dbP) ? dP : dbP) << "\t"
                         << avdP / Nee0 << "\t"
                         << avbdP / Nbee0 << "\t"
                         << M_0->norm << endl;

        con_qty_ofstream.close();
    }
}

/*---------------------------------------------------------------------------*/

void NuOsc::surv_prob(const FieldVar *ivstate, const FieldVar *ivstate0, uint t)
{
    /*
        Total survival probabilities of \nu and \bar\nu.
    */

    std::ofstream surv_prob_ofstream;
    std::string surv_prob_fname = "survival_probability.dat";
    if (t == 0)
    {
        surv_prob_ofstream.open(surv_prob_fname, std::ofstream::out | std::ofstream::trunc);
        
        if (!surv_prob_ofstream)
        {
            std::cout << "Unable to open " << surv_prob_fname << std::endl;
            return;
        }
        surv_prob_ofstream << "# [time, <Pee>, <Pbee>]" << std::endl;
    }
    else
    {
        surv_prob_ofstream.open(surv_prob_fname, std::ofstream::out | std::ofstream::app);
    }

    if (!surv_prob_ofstream)
    {
        std::cout << "Unable to open " << surv_prob_fname << std::endl;
        return;
    }

    double num_Pee = 0;
    double num_Pbee = 0;
    double dnom_Pee = 0;
    double dnom_Pbee = 0;

    for (int i = 0; i < nvz; i++)
    {
        for (int j = 0; j < nz; j++)
        {
            num_Pee += ivstate->ee[idx(i, j)] * dz * dv;
            num_Pbee += ivstate->bee[idx(i, j)] * dz * dv;
            dnom_Pee += ivstate0->ee[idx(i, j)] * dz * dv;
            dnom_Pbee += ivstate0->bee[idx(i, j)] * dz * dv;
        }
    }
    surv_prob_ofstream << t << "\t"
                       << std::scientific
                       << num_Pee / dnom_Pee << "\t"
                       << num_Pbee / dnom_Pbee << std::endl;

    surv_prob_ofstream.close();
}

/*------------------------------------testing ---------------------------------------*/

void NuOsc::v_distr_of_surv_prob(const FieldVar *ivstate, const FieldVar *ivstate0, const uint t)
{
    /*
      Survival probabilities of each mode averaged over the domain at a give time.
    */

    std::ofstream av_spv_ofstream;
    std::string av_spv_fname = "dom_avrgd_surv_prob_" + std::to_string(t) + ".dat";
    av_spv_ofstream.open(av_spv_fname, std::ofstream::out | std::ofstream::trunc);
    if (!av_spv_ofstream)
    {
        std::cout << "Unable to open " << av_spv_fname << std::endl;
        return;
    }
    else
    {
        av_spv_ofstream << "# [vz, <P_ee>(v), <P_bee>(v)]" << std::endl;
    }
    for (int i = 0; i < nvz; i++)
    {
        double P_ee   = 0;
        double P_bee  = 0;
        double P_ee0  = 0;
        double P_bee0 = 0;

        for (int j = 0; j < nz; j++)
        {
            P_ee   += ivstate->ee[idx(i, j)] * dz;
            P_bee  += ivstate->bee[idx(i, j)] * dz;
            P_ee0  += ivstate0->ee[idx(i, j)] * dz;
            P_bee0 += ivstate0->bee[idx(i, j)] * dz;
        }
        av_spv_ofstream << std::scientific << vz[i] << "\t"
                        << (P_ee / P_ee0) << "\t"
                        << (P_bee / P_bee0) << std::endl;
    }
    av_spv_ofstream.close();
}

/*---------------------------------------------------------------------------*/

#endif // __ANALYSIS__

/*------------------------------- EOF ------------------------------------*/
