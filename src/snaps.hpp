#if not defined(__SNAPS__)
#define __SNAPS__

/*---------------------------------------------------------------------------*/

// Write a C-contiguous float64 matrix in NumPy's simple, dependency-free NPY
// format.  Plotting can memory-map this directly, avoiding the size and parse
// cost of millions of decimal strings.
inline bool write_numpy_matrix(const std::string &file_name,
                               const std::vector<double> &data,
                               std::size_t rows,
                               std::size_t columns)
{
    if (data.size() != rows * columns)
        return false;
    std::ofstream out(file_name, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!out)
        return false;

    std::ostringstream description;
    description << "{'descr': '<f8', 'fortran_order': False, 'shape': ("
                << rows << ", " << columns << "), }";
    std::string header = description.str();
    const std::size_t preamble_size = 10; // magic + version + uint16 header length
    const std::size_t remainder = (preamble_size + header.size() + 1) % 64;
    if (remainder != 0)
        header.append(64 - remainder, ' ');
    header.push_back('\n');
    if (header.size() > 65535)
        return false;

    const char magic[] = "\x93NUMPY";
    const char version[2] = {1, 0};
    const unsigned int header_size = static_cast<unsigned int>(header.size());
    const char encoded_size[2] = {
        static_cast<char>(header_size & 0xffU),
        static_cast<char>((header_size >> 8U) & 0xffU)};
    out.write(magic, 6);
    out.write(version, 2);
    out.write(encoded_size, 2);
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(reinterpret_cast<const char *>(data.data()),
              static_cast<std::streamsize>(data.size() * sizeof(double)));
    return static_cast<bool>(out);
}

/*---------------------------------------------------------------------------*/

void NuOsc::write_state(unsigned int t)
{
    std::string file_name = "state.bin";
    std::string file_name_temp = "state" + std::to_string(t) + ".bin";
    std::ofstream fpw(file_name_temp, std::ios::out | std::ios::binary);
    if (!fpw)
    {
        std::cout << "Failed call to open " << file_name_temp << " in NuOsc::write_state. \n";
        return;
    }

    fpw.write((char *)&t, sizeof(int)); // First entry is the current iteration
    std::vector<double> row(static_cast<std::size_t>(8) * nz);
    for (int i = 0; i < nvz; i++)
    {
        for (int j = 0; j < nz; j++)
        {
            int ij = idx(i, j);
            const std::size_t n = static_cast<std::size_t>(8) * j;
            row[n] = v_stat->ee[ij]; row[n + 1] = v_stat->xx[ij];
            row[n + 2] = v_stat->ex_re[ij]; row[n + 3] = v_stat->ex_im[ij];
            row[n + 4] = v_stat->bee[ij]; row[n + 5] = v_stat->bxx[ij];
            row[n + 6] = v_stat->bex_re[ij]; row[n + 7] = v_stat->bex_im[ij];
        }
        fpw.write(reinterpret_cast<const char *>(row.data()),
                  static_cast<std::streamsize>(row.size() * sizeof(double)));
    }
    fpw.close();

    std::rename(file_name_temp.c_str(), file_name.c_str());
}

/*---------------------------------------------------------------------------*/

void NuOsc::write_state0(const FieldVar *stat0)
{
    std::string file_name = "state0.bin";
    std::ofstream fpw(file_name, std::ios::out | std::ios::binary);
    if (!fpw)
    {
        std::cout << "Failed call to open " << file_name << " in NuOsc::write_state0. \n";
        return;
    }

    std::vector<double> row(static_cast<std::size_t>(8) * nz);
    for (int i = 0; i < nvz; i++)
    {
        for (int j = 0; j < nz; j++)
        {
            int ij = idx(i, j);
            const std::size_t n = static_cast<std::size_t>(8) * j;
            row[n] = stat0->ee[ij]; row[n + 1] = stat0->xx[ij];
            row[n + 2] = stat0->ex_re[ij]; row[n + 3] = stat0->ex_im[ij];
            row[n + 4] = stat0->bee[ij]; row[n + 5] = stat0->bxx[ij];
            row[n + 6] = stat0->bex_re[ij]; row[n + 7] = stat0->bex_im[ij];
        }
        fpw.write(reinterpret_cast<const char *>(row.data()),
                  static_cast<std::streamsize>(row.size() * sizeof(double)));
    }
    fpw.close();
}

/*---------------------------------------------------------------------------*/

int NuOsc::read_state()
{
    std::string file_name = "state.bin";
    bool is_loading_0 = false;
    if (!file_exists(file_name))
    {
        std::cout << "Unable to find " << file_name << "\n";
        file_name = "state0.bin";
        is_loading_0 = true;
        std::cout << "Loading from " << file_name << ". \n";
    }

    std::ifstream fpr(file_name, std::ios::in | std::ios::binary);
    if (!fpr)
    {
        std::cout << file_name << " did not open. Exiting.\n";
        exit(1);
    }

    int t;
    if (!is_loading_0)
    {
        fpr.read((char *)&t, sizeof(int));
    }
    else
    {
        t = 1;
    }

    std::cout << "Loading from " << file_name << ". \n";
    std::vector<double> row(static_cast<std::size_t>(8) * nz);
    for (int i = 0; i < nvz; i++)
    {
        fpr.read(reinterpret_cast<char *>(row.data()),
                 static_cast<std::streamsize>(row.size() * sizeof(double)));
        for (int j = 0; j < nz; j++)
        {
            int ij = idx(i, j);
            const std::size_t n = static_cast<std::size_t>(8) * j;
            v_stat->ee[ij] = row[n]; v_stat->xx[ij] = row[n + 1];
            v_stat->ex_re[ij] = row[n + 2]; v_stat->ex_im[ij] = row[n + 3];
            v_stat->bee[ij] = row[n + 4]; v_stat->bxx[ij] = row[n + 5];
            v_stat->bex_re[ij] = row[n + 6]; v_stat->bex_im[ij] = row[n + 7];
        }
    }
    if (!fpr)
    {
        std::cout << file_name << " is incomplete. Exiting.\n";
        exit(1);
    }
    fpr.close();
    updateBufferZone(v_stat);
    // Checkpoints are written after completing iteration t, so resume at the
    // following iteration.  state0.bin represents the unevolved state.
    return is_loading_0 ? 1 : t + 1;
}

/*---------------------------------------------------------------------------*/

void NuOsc::read_state0(FieldVar *stat0)
{
    std::string file_name = "state0.bin";
    std::ifstream fpr(file_name, std::ios::in | std::ios::binary);
    if (!fpr)
    {
        std::cout << file_name << " does not exist. Exiting.\n";
        exit(1);
    }

    std::vector<double> row(static_cast<std::size_t>(8) * nz);
    for (int i = 0; i < nvz; i++)
    {
        fpr.read(reinterpret_cast<char *>(row.data()),
                 static_cast<std::streamsize>(row.size() * sizeof(double)));
        for (int j = 0; j < nz; j++)
        {
            int ij = idx(i, j);
            const std::size_t n = static_cast<std::size_t>(8) * j;
            stat0->ee[ij] = row[n]; stat0->xx[ij] = row[n + 1];
            stat0->ex_re[ij] = row[n + 2]; stat0->ex_im[ij] = row[n + 3];
            stat0->bee[ij] = row[n + 4]; stat0->bxx[ij] = row[n + 5];
            stat0->bex_re[ij] = row[n + 6]; stat0->bex_im[ij] = row[n + 7];
        }
    }
    if (!fpr)
    {
        std::cout << file_name << " is incomplete. Exiting.\n";
        exit(1);
    }
    fpr.close();
    return;
}

/*---------------------------------------------------------------------------*/

void NuOsc::read_G0()
{
    std::string file_name = "G0.bin";
    std::ifstream g_state_in(file_name, std::ios::in | std::ios::binary);
    if (!g_state_in)
    {
        std::cout << file_name << " does not exist. Exiting.\n";
        exit(EXIT_FAILURE);
    }

    long int buffer_size = sizeof(double);
    for (int i = 0; i < nvz; i++)
    {
        for (int j = 0; j < nz; j++)
        {
            int ij = idx(i, j);
            g_state_in.read((char *)&G0->G[ij], buffer_size);
            g_state_in.read((char *)&G0->bG[ij], buffer_size);
        }
    }
    g_state_in.close();
    return;
}

/*---------------------------------------------------------------------------*/

void NuOsc::dump_rho(const FieldVar *ivstate, const uint t_)
{
    const std::string rdump_filename = "rho_" + std::to_string(t_) + ".npy";
    const std::size_t rows = static_cast<std::size_t>(nz) * nvz;
    std::vector<double> snapshot(rows * 10);
#pragma omp parallel for collapse(2)
    for (int j = 0; j < nz; ++j)
        for (int i = 0; i < nvz; ++i)
        {
            const int ij = idx(i, j);
            const std::size_t n = (static_cast<std::size_t>(j) * nvz + i) * 10;
            snapshot[n] = Z[j]; snapshot[n + 1] = vz[i];
            snapshot[n + 2] = ivstate->ee[ij]; snapshot[n + 3] = ivstate->xx[ij];
            snapshot[n + 4] = ivstate->ex_re[ij]; snapshot[n + 5] = ivstate->ex_im[ij];
            snapshot[n + 6] = ivstate->bee[ij]; snapshot[n + 7] = ivstate->bxx[ij];
            snapshot[n + 8] = ivstate->bex_re[ij]; snapshot[n + 9] = ivstate->bex_im[ij];
        }
    if (!write_numpy_matrix(rdump_filename, snapshot, rows, 10))
        std::cout << "Unable to write " << rdump_filename << "\n";
}

/*---------------------------------------------------------------------------*/

#endif // __SNAPS__

/*---------------------------------------------------------------------------*/
