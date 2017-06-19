#include <stdlib.h>
#include <stdio.h>
#include <unistd.h> // access
#include <math.h>
#include <assert.h>
#include <mkl_vsl.h>
#include "timer.h"

#define M_PI 3.14159265358979323846

static const int iterations = 3;

static const int nr = 2;
static const int nk = 3;
double radius = 1;

int populate_radii(int npos, double *rad, double val);
int traj_read_npos(const char *fn);
int traj_read(const char *fn, double *box_width, int npos, double *pos);
int gold_read(const char *fn, int npos, double *a);
double compare_gold(int npos, double *a,double *gold);
int rpy_write(const char *fn, int npos, const double *a);


int rpy_ewald(int np, double *a, const double *pos, double L, const double *rad, double xi, int nr, int nk);
void scalar_rpy_ewald_recip(double k, double xi, double *m2);
void scalar_rpy_ewald_real(double r, double xi, double a3, double *m11, double *m12);

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: hd in.xyz out.xyz gold.xyz\n");
        return -1;
    }

    char *input_filename  = argv[1];
    char *output_filename = argv[2];
    char *gold_filename   = argv[3];

    int npos = traj_read_npos(input_filename);
    printf("Number of particles: %d\n", npos);

    if (access(output_filename, F_OK) != -1) {
        printf("Output file already exists: %s\nExiting...\n", output_filename);
        return 0;
    }

    double *pos   = (double *) _mm_malloc(3*npos*sizeof(double), 64);
    double *a     = (double *) _mm_malloc((3*npos) * (3*npos) * sizeof(double), 64);
    double *gold  = (double *) _mm_malloc((3*npos) * (3*npos) * sizeof(double), 64);
    double *rad   = (double *) _mm_malloc(npos * sizeof(double), 64);

    assert(pos);
    assert(a);
    assert(gold);
    assert(rad);

    if (access(gold_filename, F_OK) == -1) {
        printf("[WARNING] Unable to access gold file \"%s\"; comparison will not proceed.\n", gold_filename);
    } else {
        gold_read(gold_filename, npos, gold);
    }

    double start_time, box_width;

    populate_radii(npos, rad, 1.0);
    traj_read(input_filename, &box_width, npos, pos);
    printf("Simulation box width: %f\n", box_width);
    double xi = (1.5 * M_PI) / box_width;

    rpy_ewald(npos, a, pos, box_width, rad, xi, nr, nk);// Warming up the cache
    double t1, t0 = time_in_seconds();
    for(int i=0; i<iterations; ++i) {
        for (int p=0; p<3*npos*3*npos; p++){
            a[p] = 0;
	}	
//    a     = (double *) _mm_malloc((3*npos) * (3*npos) * sizeof(double), 64);
        rpy_ewald(npos, a, pos, box_width, rad, xi, nr, nk);
    }
    t1 = time_in_seconds();

    printf("%d iterations took %f seconds.\n Average time per iteration: %f\n", iterations, (t1-t0), (t1-t0)/(double)iterations);

    double error = compare_gold(npos, a, gold);
    printf("Squared Error: %f\n", error);
    printf("Speedup: %f\n", 19.317257/(t1-t0)*(double)iterations);

    rpy_write(output_filename, npos, a);

    _mm_free(pos);
    _mm_free(a);
    _mm_free(gold);
    _mm_free(rad);
    return 0;
}

int populate_radii(int npos, double *rad, double val) {
    for(int r=0; r<npos; r++) {
        rad[r] = val;
    }
    return 0;
}


int traj_read_npos(const char *filename)
{
    int npos;
    FILE *fp = fopen(filename, "r");
    fscanf(fp, "%d\n", &npos);
    assert(npos > 0);
    fclose(fp);
    return npos;
}

int traj_read(const char *filename, double *box_width,
              int npos, double *pos)
{
    int npos_read;
    FILE *fp = fopen(filename, "r");
    assert(fp);

    fscanf(fp, "%d\n", &npos_read);
    char label[100];
    double start_time;
    fgets(label, 100, fp);
    sscanf(label, "%lf %lf", &start_time, box_width);

    assert(npos == npos_read);

    for (int i=0; i<npos; i++, pos+=3)
        fscanf(fp, "%*d %lf %lf %lf\n", &pos[0], &pos[1], &pos[2]);

    fclose(fp);
    return 0;
}

int gold_read(const char *filename, int npos, double *gold)
{
    int npos_read;
    FILE *fp = fopen(filename, "r");
    assert(fp);

    fscanf(fp, "%d\n", &npos_read);
    char label[100];
    fgets(label, 100, fp);

    assert(npos == npos_read);

    for (int i=0; i<3*npos; i++) {
        for (int j=0; j<3*npos; j++) {
            fscanf(fp, "%lf\n", &gold[i*(3*npos) + j]);
        }
    }
    fclose(fp);
    return 0;
}

double compare_gold(int npos, double *a,double *gold) {
    double err = 0.0;
    for (int i=0; i<npos; i++) {
        for (int j=0; j<npos; j++) {
            double diff = a[i*(npos*3) + j] - gold[i*(npos*3) +j];
            err += diff*diff;
//            if(err>0){printf("error at position: i=%d j=%d and err = %lf\n", i, j, err);}
        }
    }
    return err;
}

int rpy_write(const char *filename, int npos, const double *a)
{
    FILE *fp = fopen(filename, "a");
    assert(fp);

    fprintf(fp, "%d x %d xyz particles\n", npos, npos);

    int idx=0;
    for (int i=0; i<3*npos; i++) {
        for (int j=0; j<npos; j++) {
            fprintf(fp, "%f %f %f ", a[idx], a[idx+1], a[idx+2]);
            idx+=3;
        }
        fprintf(fp, "\n");
    }

    fclose(fp);

    return 0;
}
