#if not defined(__PARSER__)
#define __PARSER__
/*---------------------------------------------------------------------------*/
template <typename T>
void string_to_type(std::string value, T &var)
{
    /*Convert string to type T and assign it to var*/
    std::stringstream svalue(value);
    svalue >> var;
}

/*---------------------------------------------------------------------------*/

template <typename T>
void cssl_to_vec(std::string cssl, std::vector<T> &vec) // cssl -> comma-separated-string-list
{
    /*
        Convert comma sepparated list of strings(if the fiorm [a, b, c, ...]) and convert it to a 
        vector of type T.
    */
    int start = cssl.find_first_of("[");
    int end = cssl.find_first_of("]");
    cssl = cssl.substr(start + 1, end - 1);

    // In case any accidantal space in the begining of cssl.
    cssl = cssl.substr(cssl.find_first_not_of(" "), end - 1) + ",;";
    std::string c = "";

    while (true)
    {
        c = cssl.substr(0, cssl.find_first_of(","));
        cssl = cssl.substr(cssl.find_first_of(","), cssl.find_first_of("]"));
        cssl = cssl.substr(cssl.find_first_not_of(" ") + 1, cssl.find_first_of("]"));
        std::stringstream svalue(c);
        T tval;
        svalue >> tval;
        vec.push_back(tval);

        if (cssl == ";")
            break;
    }
}

/*---------------------------------------------------------------------------*/

class Params
{
public:
    std::string SCHEME;

    // Simulation specific configs.
    int N_ITER = 0;
    int DUMP_EVERY = 0;
    int ANAL_EVERY = 0;

    // Grid specific configs.
    int nz = 0;
    int nvz = 0;
    int gz = 0;

    double z0 = 0.0;
    double z1 = 0.0;
    double v0;
    double v1;
    double CFL = 0.0;
    double pmo = 0.0;
    double omega = 0.0;
    double theta = 0.0;
    double mu = 1.0;
    double k_max = 0.1;
    int k_bins = 500;

    // Optional multi-energy calculation.  These values are ignored by the
    // original monoenergetic path, which is intentionally left unchanged.
    bool multienergy = false;
    int energy_bins = 12; // per sector: N neutrino + N antineutrino modes
    std::string energy_grid = "energy";
    double energy_min = 2.0;
    double energy_max = 60.0;
    double omega_grid_min = 1.0e-3;
    double omega_grid_max = 0.25;
    double reference_energy = 10.0;
    double spectral_alpha = 2.3;
    double mean_energy_nue = 9.6;
    double mean_energy_anue = 11.8;
    double mean_energy_nux = 16.0;
    double luminosity_nue = 44.0;
    double luminosity_anue = 45.0;
    double luminosity_nux = 20.0;
    double neutrino_width = 0.6;
    double antineutrino_width = 0.6;
    double alpha = 0.85;
    std::string angular_profile = "gaussian";
    bool random_seeds = false;
    double seed_width = 3.0;
    double matter_slope = 0.0;
    double matter_offset = 0.0;
    bool exact_matter_rotation = false;
    double damping_rate = 0.0;
    double antineutrino_collision_ratio = 0.0;

    double perturbation_size = 0.0;

    //Analysis related

    int n_dump = 0;
    int dump_interval = 0;


    bool is_scheme = false;
    bool is_nz = false;
    bool is_nvz = false;
    bool is_z0 = false;
    bool is_z1 = false;
    bool is_v0 = false;
    bool is_v1 = false;
    bool is_CFL = false;
    bool is_gz = false;
    bool is_N_ITER = false;
    bool is_ANAL_EVERY = false;
    bool is_pmo = false;
    bool is_omega = false;
    bool is_theta = false;
    bool is_mu = false;

    Params(std::string CONFIG_FILE);
    ~Params() {}
};

Params::Params(std::string CONFIG_FILE)
{
    std::string line;
    std::string key;
    std::string value;

    std::ifstream config;
    config.open(CONFIG_FILE.c_str(), std::ifstream::in);
    if (!config)
    {
        std::cout << "[ FAIL ]...Unable to open "
                  << CONFIG_FILE
                  << "exitting.\n";
        exit(0);
    }
    while (config)
    {
        std::getline(config, line);
        if (line.length() > 0)
        {
            int pos = line.find_first_of(":");
            std::string left = line.substr(0, pos);
            std::string right = line.substr(pos + 1);
            if (left.length() > 0)
            {
                key = left.substr(0, left.find_first_of(" "));
                value = right.substr(right.find_first_not_of(" "), right.find_first_of("\n"));
            }
            if (key == "scheme")
            {
                string_to_type(value, SCHEME);
                is_scheme = true;
            }
            else if (key == "nz")
            {
                string_to_type(value, nz);
                is_nz = true;
            }
            else if (key == "nvz")
            {
                string_to_type(value, nvz);
                is_nvz = true;
            }
            else if (key == "CFL")
            {
                string_to_type(value, CFL);
                is_CFL = true;
            }
            else if (key == "gz")
            {
                string_to_type(value, gz);
                is_gz = true;
            }
            else if (key == "z0")
            {
                string_to_type(value, z0);
                is_z0 = true;
            }
            else if (key == "z1")
            {
                string_to_type(value, z1);
                is_z1 = true;
            }
            else if (key == "v0")
            {
                string_to_type(value, v0);
                is_v0 = true;
            }
            else if (key == "v1")
            {
                string_to_type(value, v1);
                is_v1 = true;
            }

            else if (key == "perturbation_size")
            {
                string_to_type(value, perturbation_size);
            }
            else if (key == "N_ITER")
            {
                string_to_type(value, N_ITER);
                is_N_ITER = true;
            }
            else if (key == "ANAL_EVERY")
            {
                string_to_type(value, ANAL_EVERY);
                is_ANAL_EVERY = true;
            }
            else if (key == "pmo")
            {
                string_to_type(value, pmo);
                is_pmo = true;
            }
            else if (key == "omega")
            {
                string_to_type(value, omega);
                is_omega = true;
            }
            else if (key == "theta")
            {
                string_to_type(value, theta);
                is_theta = true;
            }
            else if (key == "mu")
            {
                string_to_type(value, mu);
                is_mu = true;
            }
            else if (key == "k_max")
            {
                string_to_type(value, k_max);
            }
            else if (key == "k_bins")
            {
                string_to_type(value, k_bins);
            }
            else if (key == "multienergy")
            {
                int enabled = 0;
                string_to_type(value, enabled);
                multienergy = (enabled != 0);
            }
            else if (key == "energy_bins")
                string_to_type(value, energy_bins);
            else if (key == "energy_grid")
                string_to_type(value, energy_grid);
            else if (key == "energy_min")
                string_to_type(value, energy_min);
            else if (key == "energy_max")
                string_to_type(value, energy_max);
            else if (key == "omega_grid_min")
                string_to_type(value, omega_grid_min);
            else if (key == "omega_grid_max")
                string_to_type(value, omega_grid_max);
            else if (key == "reference_energy")
                string_to_type(value, reference_energy);
            else if (key == "spectral_alpha")
                string_to_type(value, spectral_alpha);
            else if (key == "mean_energy_nue")
                string_to_type(value, mean_energy_nue);
            else if (key == "mean_energy_anue")
                string_to_type(value, mean_energy_anue);
            else if (key == "mean_energy_nux")
                string_to_type(value, mean_energy_nux);
            else if (key == "luminosity_nue")
                string_to_type(value, luminosity_nue);
            else if (key == "luminosity_anue")
                string_to_type(value, luminosity_anue);
            else if (key == "luminosity_nux")
                string_to_type(value, luminosity_nux);
            else if (key == "neutrino_width")
                string_to_type(value, neutrino_width);
            else if (key == "antineutrino_width")
                string_to_type(value, antineutrino_width);
            else if (key == "alpha")
                string_to_type(value, alpha);
            else if (key == "angular_profile")
                string_to_type(value, angular_profile);
            else if (key == "random_seeds")
            {
                int enabled = 0;
                string_to_type(value, enabled);
                random_seeds = (enabled != 0);
            }
            else if (key == "seed_width")
                string_to_type(value, seed_width);
            else if (key == "matter_slope")
                string_to_type(value, matter_slope);
            else if (key == "matter_offset")
                string_to_type(value, matter_offset);
            else if (key == "exact_matter_rotation")
            {
                int enabled = 0;
                string_to_type(value, enabled);
                exact_matter_rotation = (enabled != 0);
            }
            else if (key == "damping_rate")
                string_to_type(value, damping_rate);
            else if (key == "antineutrino_collision_ratio")
                string_to_type(value, antineutrino_collision_ratio);
            else if (key == "n_dump_rho")
            {
                string_to_type(value, n_dump);
            }
            else
            {
                std::cout << "Redundant or unknown key: " << key << std::endl;
                continue;
            }
        }
    }
    //----------------------------------------------------------

    if (!(is_scheme && is_nz && is_nvz && is_z0 && is_z1 && is_v0 && is_v1 && is_CFL && is_gz &&
          is_N_ITER && is_ANAL_EVERY))
    {
        std::cout << "[ FAIL ]...Incomplete config file" << std::endl
                  << "Exiting." << std::endl;
        exit(0);
    }

#ifdef VAC_OSC_ON
    if (!(is_pmo && is_omega && is_theta))
    {
        std::cout << "[ FAIL ]...Incomplete config file" << std::endl
                  << "Exiting." << std::endl;
        exit(0);
    }
#endif
#ifdef COLL_OSC_ON
    if (!(is_mu))
    {
        std::cout << "[ FAIL ]...Incomplete config file" << std::endl
                  << "Exiting." << std::endl;
        exit(0);
    }
#endif
    if (k_max <= 0.0 || k_bins < 2)
    {
        std::cout << "[ FAIL ]...k_max must be positive and k_bins must be at least 2."
                  << std::endl;
        exit(1);
    }
    if (angular_profile != "gaussian" && angular_profile != "two_beam"
        && angular_profile != "symmetric_two_beam"
        && angular_profile != "soumya_two_beam")
    {
        std::cout << "[ FAIL ]...angular_profile must be gaussian, two_beam, symmetric_two_beam, or soumya_two_beam."
                  << std::endl;
        exit(1);
    }
    if (seed_width <= 0.0)
    {
        std::cout << "[ FAIL ]...seed_width must be positive." << std::endl;
        exit(1);
    }
    if ((angular_profile == "two_beam" || angular_profile == "symmetric_two_beam"
         || angular_profile == "soumya_two_beam")
        && (multienergy || nvz != 2))
    {
        std::cout << "[ FAIL ]...two-beam profiles require monoenergetic mode with nvz=2."
                  << std::endl;
        exit(1);
    }
    if (multienergy)
    {
        if (energy_bins < 2 || reference_energy <= 0.0 || spectral_alpha <= -1.0 ||
            mean_energy_nue <= 0.0 || mean_energy_anue <= 0.0 || mean_energy_nux <= 0.0 ||
            luminosity_nue <= 0.0 || luminosity_anue <= 0.0 || luminosity_nux <= 0.0)
        {
            std::cout << "[ FAIL ]...Invalid multi-energy spectral configuration." << std::endl;
            exit(1);
        }
        if (!((energy_grid == "energy" && energy_min > 0.0 && energy_max > energy_min) ||
              (energy_grid == "omega" && omega_grid_min >= 0.0 && omega_grid_max > omega_grid_min)))
        {
            std::cout << "[ FAIL ]...energy_grid must be energy or omega, with increasing positive bounds."
                      << std::endl;
            exit(1);
        }
    }
if (n_dump > 0)
    {   
        dump_interval = (int)(N_ITER/n_dump);
        if (dump_interval < 1)
        {
            dump_interval = 1;
        }
    }   
    else
    {   
        dump_interval = N_ITER + 1;
    } 
}

/*---------------------------------------------------------------------------*/

#endif // __PARSER__

/*---------------------------------------------------------------------------*/
