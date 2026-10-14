#if not defined(__MISCELLANEOUS_FUNCS__)
#define __MISCELLANEOUS_FUNCS__
/*---------------------------------------------------------------------------*/

bool file_exists(std::string path)
{
    std::ifstream f(path.c_str(), std::ios::in);
    if (!f)
    {
        return false;
    }
    else
    {
        f.close();
        return true;
    }
}

/*---------------------------------------------------------------------------*/

float roundoff(float value, unsigned char prec)
{
  float pow_10 = pow(10.0f, (float)prec);
  return round(value * pow_10) / pow_10;
}

/*---------------------------------------------------------------------------*/

std::string draw(const int l, const string c)
{
    // write a string l times on to std::out
    std::string s = "";
    for (int i = 0; i < l; i++)
    {
        s += c;
    }
    return s;
}
/*---------------------------------------------------------------------------*/

template <typename T>
int sign(T val)
{
    return (T(0) < val) - (val < T(0));
}
/*---------------------------------------------------------------------------*/

inline void swap(FieldVar **a, FieldVar **b)
{
    FieldVar *tmp = *a;
    *a = *b;
    *b = tmp;
}

/*---------------------------------------------------------------------------*/

double g(double v, double v0, double sigma)
{
    double exponant = (v - v0) * (v - v0) / (2.0 * sigma * sigma);
    double N = sigma * sqrt(M_PI / 2.0) * (erf((1.0 + v0) / sigma / sqrt(2.0)) + erf((1.0 - v0) / sigma / sqrt(2.0)));
    return exp(-exponant) / N;
}

/*---------------------------------------------------------------------------*/

inline double eps(double z, double z0, double amp, double width)
{
    return amp * exp(-(z - z0) * (z - z0) / (2.0 * width * width));
}

/*---------------------------------------------------------------------------*/

inline double eps_(double z, double z0, double amp, double width)
{
    double e = eps(z, z0, amp, width);
    return sqrt(1.0 - e * e);
}

/*---------------------------------------------------------------------------*/

inline double random_amp(double a)
{
    return a * rand() / RAND_MAX;
}
/*---------------------------------------------------------------------------*/

double hev(double x, double x0)
{
    /* Heviside theta function */
    return 0.5 * (1.0 + sign(x - x0));
}

/*---------------------------------------------------------------------------*/

double gauss(double x, double x0, double sigma)
{
    double exponant = (x - x0) * (x - x0) / (2.0 * sigma * sigma);
    return exp(-exponant);
}

/*---------------------------------------------------------------------------*/

double L(double x, unsigned int order)
{
    // Legendre polynomials.
    double value;
    switch (order)
    {
    case 0:
        value = 1;
        break;
    case 1:
        value = x;
        break;
    case 2:
        value = 0.5 * (3 * x * x - 1);
        break;
    case 3:
        value = 0.5 * (5 * pow(x, 3) - 3 * x);
        break;
    case 4:
        value = 0.125 * (35 * pow(x, 4) - 30 * x * x + 3);
        break;
    case 5:
        value = 0.125 * (63 * pow(x, 5) - 70 * pow(x, 3) + 15 * x);
        break;
    case 6:
        value = (1.0 / 16) * (231 * pow(x, 6) - 315 * pow(x, 4) + 105 * x * x - 5);
        break;
    case 7:
        value = (1.0 / 16) * (429 * pow(x, 7) - 693 * pow(x, 5) + 315 * pow(x, 3) - 35 * x);
        break;
    case 8:
        value = (1.0 / 128) * (6435 * pow(x, 8) - 12012 * pow(x, 6) + 6930 * pow(x, 4) - 1260 * x * x + 35);
        break;
    case 9:
        value = (1.0 / 128) * (12155 * pow(x, 9) - 25740 * pow(x, 7) + 18018 * pow(x, 5) - 4620 * pow(x, 3) + 315 * x);
        break;
    case 10:
        value = (1.0 / 256) * (46189 * pow(x, 10) - 109395 * pow(x, 8) + 90090 * pow(x, 6) - 30030 * pow(x, 4) + 3465 * x * x - 63);
        break;
    default:
        value = 0;
    }
    return value;
}

/*---------------------------------------------------------------------------*/

#endif // __MISCELLANEOUS_FUNCS__

/*---------------------------------------------------------------------------*/
