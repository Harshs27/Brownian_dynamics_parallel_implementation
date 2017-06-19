#include <stdlib.h>
#include <stdio.h>
#include <unistd.h> // access
#include <math.h>
#include <assert.h>
#include <mkl_vsl.h>
#include "timer.h"
#include "bd.h"

VSLStreamStatePtr stream;
const int stream_seed = 777;

// return number of particles in trajectory file
int traj_read_npos(const char *filename)
{
    int npos;
    FILE *fp = fopen(filename, "r");
    fscanf(fp, "%d\n", &npos);
    assert(npos > 0);
    fclose(fp);
    return npos;
}

// read first frame of trajectory file
int traj_read(
  const char *filename, 
  char *label,
  int npos,
  double *pos, 
  int *types)
{
    int npos_read;
    FILE *fp = fopen(filename, "r");
    assert(fp);

    fscanf(fp, "%d\n", &npos_read);
    fgets(label, LINE_LEN, fp);

    assert(npos == npos_read);

    for (int i=0; i<npos; i++, pos+=3)
        fscanf(fp, "%d %lf %lf %lf\n", &types[i], &pos[0], &pos[1], &pos[2]);

    fclose(fp);

    return 0;
}

// append positions to trajectory file
int traj_write(
  const char *filename, 
  const char *label,
  int npos, 
  const double *pos, 
  const int *types)
{
    FILE *fp = fopen(filename, "a");
    assert(fp);

    fprintf(fp, "%d\n", npos);
    fprintf(fp, "%s\n", label);

    for (int i=0; i<npos; i++, pos+=3)
        fprintf(fp, "%d %f %f %f\n", types[i], pos[0], pos[1], pos[2]);

    fclose(fp);

    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        fprintf(stderr, "usage: bd in.xyz out.xyz num_intervals\n");
        return -1;
    }

    char *input_filename  = argv[1];
    char *output_filename = argv[2];
    int num_intervals = atoi(argv[3]);

    if (access(output_filename, F_OK) != -1)
    {
        printf("Output file already exists: %s\nExiting...\n", output_filename);
        return -1;
    }

    int npos = traj_read_npos(input_filename);
    printf("Number of particles: %d\n", npos);
    printf("Number of intervals to simulate: %d\n", num_intervals);

    double *pos   = (double *) _mm_malloc(3*npos*sizeof(double), 64);
    double *buf   = (double *) _mm_malloc(3*npos*sizeof(double), 64);
    int    *types = (int *)   _mm_malloc(  npos*sizeof(int), 64);
    assert(pos);
    assert(buf);
    assert(types);

    double *pos_copy = (double *) _mm_malloc(npos*3*sizeof(double), 64);
    int *next = (int *) _mm_malloc(npos*sizeof(int), 64);
    double *forces = (double *) _mm_malloc(3*npos*sizeof(double), 64); //NOTE: Replace by __mm__malloc and 64 bit align
    assert(pos_copy);
    assert(next);
    assert(forces);
    double f_const = sqrt(2.*DELTAT);// updation of random function in brownian pos update


    // initialize random number stream
    vslNewStream(&stream, VSL_BRNG_SFMT19937, stream_seed);

    char label[LINE_LEN]; char init_label[LINE_LEN];
    double start_time, box_width;

    traj_read(input_filename, label, npos, pos, types);
    sscanf(label, "%lf %lf", &start_time, &box_width);
    sprintf(init_label, "%lf %lf", start_time, box_width);
    printf("Simulation box width: %f\n", box_width);

    double t1, t0 = time_in_seconds();
	// WRITING the initial state to the output file    
//        traj_write(output_filename, init_label, npos, pos, types);

    // simulate for num_intervals, writing frame after each interval
    for (int interval=1; interval<=num_intervals; interval++)
    {
        bd(npos, pos, buf, types, box_width, pos_copy, next, forces, f_const);
        sprintf(label, "%f %f",
            start_time+interval*INTERVAL_LEN*DELTAT, box_width);
        traj_write(output_filename, label, npos, pos, types);

        if (interval % 100 == 0)
            printf("Done interval: %d\n", interval);
    }
    t1 = time_in_seconds();
    printf("Time: %f for %d intervals\n", t1-t0, num_intervals);
    printf("Time per time step: %g\n", (t1-t0)/num_intervals/INTERVAL_LEN);

    _mm_free(pos);
    _mm_free(buf);
    _mm_free(types);
    _mm_free(pos_copy);
    _mm_free(next);
    _mm_free(forces);

    vslDeleteStream(&stream);

    return 0;
}
