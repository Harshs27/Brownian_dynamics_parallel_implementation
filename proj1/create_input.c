#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<assert.h>
//#include<mkl_vsl.h>

#define M_PI 3.14159265358979323846

//VSLStreamStatePtr stream;
//const int stream_seed = 777;

double get_rand(int a, int b){//gen random num between (a, b)
        return a+(b-a)*(double)rand() / (double)((unsigned)RAND_MAX + 1);
}


int traj_write(const char *filename, const char *label, int npos, const double *pos, const int *types)
{
    FILE *fp = fopen(filename, "w");
    assert(fp);

    fprintf(fp, "%d\n", npos);
    fprintf(fp, "%s\n", label);

    for (int i=0; i<npos; i++, pos+=3)
        fprintf(fp, "%d %f %f %f\n", types[i], pos[0], pos[1], pos[2]);

    fclose(fp);

    return 0;
}

int main(){
    int npos = 10000; //NOTE: CHANGE INPUT HERE!!!
    double a=1, a_sq, phi=0.2, L;
    a_sq = a*a;
    L = pow(4.0/3.0*M_PI*a_sq*a*npos/phi, 1.0/3.0);
    double *pos = (double *) malloc(3*npos*sizeof(double));
    int    *types = (int *)    malloc(  npos*sizeof(int));
    int i;
    char label[15], name[50];
    // Initialising with uniform random functions
    for(i=0; i<npos; i++){
        types[i] = 0;
        pos[3*i] = get_rand(0, L);
        pos[3*i+1] = get_rand(0, L);
        pos[3*i+2] = get_rand(0, L);
    }
    // initialize random number stream
 //   vslNewStream(&stream, VSL_BRNG_SFMT19937, stream_seed);
 //   vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, 3*npos, pos, 0.0, L);

    sprintf(label, "0.0 %lf", L);    

    sprintf(name, "my_input_npos_%d.xyz", npos);
    traj_write(name, label, npos, pos, types);
    return 0;
}
