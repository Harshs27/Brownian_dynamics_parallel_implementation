#include <stdlib.h>
#include <stdio.h>
#include <unistd.h> // access
#include <math.h>
#include <assert.h>
#include "timer.h"
#include "bd.h"

#define M_PI 3.14159265358979323846

struct box
{
    int head;
};

// it is possible to use smaller boxes and more complex neighbor patterns
#define NUM_BOX_NEIGHBORS 13
int box_neighbors[NUM_BOX_NEIGHBORS][3] =
{
    {-1,-1,-1},
    {-1,-1, 0},
    {-1,-1,+1},
    {-1, 0,-1},
    {-1, 0, 0},
    {-1, 0,+1},
    {-1,+1,-1},
    {-1,+1, 0},
    {-1,+1,+1},
    { 0,-1,-1},
    { 0,-1, 0},
    { 0,-1,+1},
    { 0, 0,-1}
};


int bd(int npos, double *pos_orig, double *buf, const int *types, double L)
{
    double *pos;
//    printf("check fmod = %lf and remainder = %lf", fmod(27.027540,20.180992), remainder(17.027540,20.180992));
    pos = (double *)malloc(3*npos*sizeof(double));
    double f_const = sqrt(2.*DELTAT);// updation of random function in brownian pos update
    // Initialisations required for INTERACTION FUNCTION******** NOTE: Can take input to bd itself!!! 
    double krepul = 100, a=1, a_sq, phi=0.2, f;
    a_sq = a*a;
//    double L; 
    int boxdim;// boxdim is number of cells in L 
    double cutoff2; int numpairs_p;
//    L = pow(4.0/3.0*M_PI*a_sq*a*npos/phi, 1.0/3.0);// L = ~60 for N=10000 and L=~27 for N=1000
    cutoff2 = 2;// cutoff < L/boxdim
    boxdim = (int)(L/cutoff2*0.8);
    printf("L = %lf cutoff2 = %lf boxdim = %d\n", L, cutoff2, boxdim);
    struct box b[boxdim][boxdim][boxdim];
    struct box *bp;
    struct box *neigh_bp;

    // box indices
    int idx, idy, idz;
    int neigh_idx, neigh_idy, neigh_idz;
    // allocate implied linked list
    int *next = (int *) malloc(npos*sizeof(int));
    if (next == NULL)
    {
        printf("interactions: could not malloc array for %d particles\n", npos);
        return 1;
    }
    int p1, p2, j, i;
    double d2, dx, dy, dz, s;
    //*****************************************END initialisations***********************************

    double *forces = (double *)malloc(3*npos*sizeof(double)); //NOTE: Replace by __mm__malloc and 64 bit align
    for (int step=0; step<INTERVAL_LEN; step++)
    {
        // generate random values from standard normal distribution
        // note: this MKL function is sequential but vectorized
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, 3*npos, buf, 0., 1.);
        // Calculation of interaction per time step
        if (boxdim < 4 || cutoff2 > (L/boxdim)*(L/boxdim))
        {
            printf("interactions: bad input parameters\n");
            return 1;
        }

        // allocate memory for particles in each box
        for (idx=0; idx<boxdim; idx++){
            for (idy=0; idy<boxdim; idy++){
                for (idz=0; idz<boxdim; idz++){
                    b[idx][idy][idz].head = -1;// initialising that all boxes contain no particles
                }
            }
        }

        // traverse all particles and assign to boxes
        for (i=0; i<npos; i++)
        {

            if (pos_orig[3*i] >= 0){pos[3*i]= fmod(pos_orig[3*i], L);}// OR SINCE PARTICLES moving slowly.. change to -L
            else {// pos_orig[i] is negative
                pos[3*i] = L-fmod(-1*pos_orig[3*i], L);
            }
            if (pos_orig[3*i+1] >= 0){pos[3*i+1]= fmod(pos_orig[3*i+1], L);}// OR SINCE PARTICLES moving slowly.. change to -L
            else {// pos_orig[i] is negative
                pos[3*i+1] = L-fmod(-1*pos_orig[3*i+1], L);
            }
            if (pos_orig[3*i+2] >= 0){pos[3*i+2]= fmod(pos_orig[3*i+2], L);}// OR SINCE PARTICLES moving slowly.. change to -L
            else {// pos_orig[i] is negative
                pos[3*i+2] = L-fmod(-1*pos_orig[3*i+2], L);
            }
            if (pos[3*i]<0){printf("pos_orig = %lf pos defect = %lf and i = %d and L =%lf\n", pos_orig[3*i], pos[3*i], i, L);}


            // initialize entry of implied linked list
            next[i] = -1;
            forces[3*i+0] = 0; forces[3*i+1] = 0; forces[3*i+2] = 0; // re-initialising interaction forces at each time step
            // which box does the particle belong to?
            // assumes particles have positions within [0,L]^3
            idx = (int)(pos[3*i  ]/L*boxdim);
            idy = (int)(pos[3*i+1]/L*boxdim);
            idz = (int)(pos[3*i+2]/L*boxdim);

            // add to beginning of implied linked list
            bp = &b[idx][idy][idz];
            next[i] = bp->head; // next = previous (my notation)
            bp->head = i; // head = latest (my notation)
        }

        for (idx=0; idx<boxdim; idx++)
        {
            for (idy=0; idy<boxdim; idy++)
            {
                for (idz=0; idz<boxdim; idz++)
                {
                    bp = &b[idx][idy][idz];

                    // within box interactions
                    p1 = bp->head;
                    while (p1 != -1)
                    {
                        p2 = next[p1];
                        while (p2 != -1)
                        {

                            // do not need minimum image since we are in same box
                            dx = pos[3*p1+0] - pos[3*p2+0];
                            dy = pos[3*p1+1] - pos[3*p2+1];
                            dz = pos[3*p1+2] - pos[3*p2+2];
                            d2 = dx*dx+dy*dy+dz*dz;
                            if ( d2< cutoff2 && d2<4.0*a_sq)//updating the forces
                            {
                                s = sqrt(d2);
                                f = krepul*(2*a-s);
                                // NOTE : use pragma while parallelizing
                                forces[3*p1+0] += f*dx/s;
                                forces[3*p1+1] += f*dy/s;
                                forces[3*p1+2] += f*dz/s;
                                forces[3*p2+0] -= f*dx/s;
                                forces[3*p2+1] -= f*dy/s;
                                forces[3*p2+2] -= f*dz/s;
                            }

                            p2 = next[p2];
                        }
                        p1 = next[p1];
                    }

                    // interactions with other boxes
                    for (j=0; j<NUM_BOX_NEIGHBORS; j++)
                    {
                        neigh_idx = (idx + box_neighbors[j][0] + boxdim) % boxdim;
                        neigh_idy = (idy + box_neighbors[j][1] + boxdim) % boxdim;
                        neigh_idz = (idz + box_neighbors[j][2] + boxdim) % boxdim;

                        neigh_bp = &b[neigh_idx][neigh_idy][neigh_idz];

                        // when using boxes, the minimum image computation is 
                        // known beforehand, thus we can  compute position offsets 
                        // to compensate for wraparound when computing distances
                        double xoffset = 0.;
                        double yoffset = 0.;
                        double zoffset = 0.;
                        if (idx + box_neighbors[j][0] == -1)     xoffset = -L;
                        if (idy + box_neighbors[j][1] == -1)     yoffset = -L;
                        if (idz + box_neighbors[j][2] == -1)     zoffset = -L;
                        if (idx + box_neighbors[j][0] == boxdim) xoffset =  L;
                        if (idy + box_neighbors[j][1] == boxdim) yoffset =  L;
                        if (idz + box_neighbors[j][2] == boxdim) zoffset =  L;

                        // NOTE: modifying the function to update the forces
                        p1 = neigh_bp->head;
                        while (p1 != -1)
                        {
                            p2 = bp->head;
                            while (p2 != -1)
                            {
                                // compute distance vector
                                dx = pos[3*p1+0] - pos[3*p2+0] + xoffset;
                                dy = pos[3*p1+1] - pos[3*p2+1] + yoffset;
                                dz = pos[3*p1+2] - pos[3*p2+2] + zoffset;
                                d2 = dx*dx+dy*dy+dz*dz;
                                if ( d2< cutoff2 && d2<4.0*a_sq)
                                {
                                    s = sqrt(d2);
                                    f = krepul*(2*a-s);
                                    forces[3*p1+0] += f*dx/s;
                                    forces[3*p1+1] += f*dy/s;
                                    forces[3*p1+2] += f*dz/s;
                                    forces[3*p2+0] -= f*dx/s;
                                    forces[3*p2+1] -= f*dy/s;
                                    forces[3*p2+2] -= f*dz/s;
                                }

                                p2 = next[p2];
                            }
                            p1 = next[p1];
                        }
                    }
                }
            }
        }
        // update positions with Brownian displacements
        for (int i=0; i<3*npos; i++)
        {
//            pos_orig[i] += forces[i]*DELTAT+f_const*(-1+2*(double)rand() / (double)((unsigned)RAND_MAX + 1));//*buf[i];
            pos_orig[i] += forces[i]*DELTAT+f_const*buf[i];
        }
    }
//    free(next);
}
