#include <mkl_vsl.h>

#define INTERVAL_LEN 1
#define DELTAT       1e-4
#define LINE_LEN     100

extern VSLStreamStatePtr stream;
extern const int stream_seed;

int bd(int npos, double *pos_orig, double *buf, const int *types, double L, double *pos, int *next, double *forces, double f_const, double *au, double *rad, double xi, int nr, int nk, double * hd_vec);
