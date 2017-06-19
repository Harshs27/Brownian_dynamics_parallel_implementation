#include <stdlib.h>
#include <stdio.h>
#include <unistd.h> // access
#include <math.h>
#include <assert.h>
#include "timer.h"
#include "bd.h"
#include <omp.h>

#define NTHREADS 24
#define M_PI 3.14159265358979323846
#define my_EPS 0.000000001

void get_indices(int index, int *i, int *j, int *k, int b){
    int ib, ib2;
    ib = index%(b); ib2 = index%(b*b);
    *k = ib;
    *i = (index-ib2)/(b*b);
    *j = (ib2-*k)/b;
    return;
}


struct box
{
    int head;
};

// it is possible to use smaller boxes and more complex neighbor patterns
#define NUM_BOX_NEIGHBORS 14
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
    { 0, 0,-1},
    { 0, 0, 0} // will calculate within the box interactions
};

int bd(int npos, double * restrict pos_orig, double * restrict buf, const int *types, double L, double * restrict pos, int* restrict next, double* restrict forces, double f_const)
{
    // Initialisations required for INTERACTION FUNCTION******** NOTE: Can take input to bd itself!!! 
    double krepul = 100, a=1, a_sq, phi=0.2, f;
    a_sq = a*a;
    int boxdim;// boxdim is number of cells in L 
    double cutoff2; int numpairs_p;
    cutoff2 = 4;// cutoff < L/boxdim
    boxdim =(int)(L/cutoff2)*a;//(int)(L/cutoff2*0.8);
    printf("L = %lf cutoff2 = %lf boxdim = %d\n", L, cutoff2, boxdim);
    struct box b[boxdim][boxdim][boxdim];
    struct box *bp;
    struct box *neigh_bp;

    // box indices
    int idx, idy, idz, index, box2, ib2;
    int neigh_idx, neigh_idy, neigh_idz;
    // allocate implied linked list
    int p1, p2, j, i;
    double d2, dx, dy, dz, s;
    box2 = boxdim*boxdim;
    //*****************************************END initialisations***********************************
    if (boxdim < 4 || cutoff2 > (L/boxdim)*(L/boxdim))
    {
        printf("interactions: bad input parameters\n");
//        return 1;
    }
    double t0, t_init_cells = 0, t_assign_to_cells=0, t_update_pos=0, t_force=0;    
    for (int step=0; step<INTERVAL_LEN; step++)
    {
        // Calculation of interaction per time step
        t0 = time_in_seconds();
        // allocate memory for particles in each box
//        #pragma omp parallel for schedule(static) private(idx, idy, idz, ib2) shared(b, boxdim, box2)
//        for (index=0; index<boxdim*box2; index++){
//            idz = index%(boxdim); 
//            ib2 = index%(box2);
//            idx = (index-ib2)/(box2);
//            idy = (ib2-idz)/boxdim;
//            b[idx][idy][idz].head=-1;    
//        }
        for (idx=0; idx<boxdim; idx++){
            for (idy=0; idy<boxdim; idy++){
                for (idz=0; idz<boxdim; idz++){
                    b[idx][idy][idz].head=-1;
                }
            }
        }

        t_init_cells += time_in_seconds()-t0;
        t0 = time_in_seconds();
        // traverse all particles and assign to boxes
        #pragma omp parallel for schedule(static) private(i, idx, idy, idz, bp) shared(b, next) num_threads(NTHREADS)
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
//            next[i] = bp->head; // next = previous (my notation)
//            #pragma omp critical
//            {
            next[i] = bp->head; // next = previous (my notation)
            bp->head = i; // head = latest (my notation)
//            }
        }
        t_assign_to_cells += time_in_seconds()-t0;
        t0 = time_in_seconds();
        #pragma omp parallel for schedule(static) private(j, neigh_idx, neigh_idy, neigh_idz, neigh_bp, p1, p2, dx, dy, dz, d2, s, f, idx, idy, idz, ib2, bp) shared(b, box_neighbors, boxdim, L, pos, forces, krepul, a, a_sq, next, box2) num_threads(NTHREADS)
        for (index=0; index<boxdim*box2; index++){
            idz = index%(boxdim); 
            ib2 = index%(box2);
            idx = (index-ib2)/(box2);
            idy = (ib2-idz)/boxdim;
            bp = &b[idx][idy][idz];

            // interactions within and other boxes
            #pragma omp parallel for schedule(static) private(j, neigh_idx, neigh_idy, neigh_idz, neigh_bp, p1, p2, dx, dy, dz, d2, s, f) shared(bp, b, box_neighbors, boxdim, L, pos, forces, krepul, a, a_sq, next, idx, idy, idz)// num_threads(NTHREADS)
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
                        d2 = dx*dx+dy*dy+dz*dz+my_EPS;
                        if ( d2<4.0*a_sq)
                        {
                            s = sqrt(d2);
                            f = krepul*(2*a-s);
                            #pragma omp atomic
                            forces[3*p1+0] += f*dx/s;
                            #pragma omp atomic
                            forces[3*p1+1] += f*dy/s;
                            #pragma omp atomic
                            forces[3*p1+2] += f*dz/s;
                            #pragma omp atomic
                            forces[3*p2+0] -= f*dx/s;
                            #pragma omp atomic
                            forces[3*p2+1] -= f*dy/s;
                            #pragma omp atomic
                            forces[3*p2+2] -= f*dz/s;
                        }

                        p2 = next[p2];
                    }
                    p1 = next[p1];
                }
            }
        }

        t_force += time_in_seconds() - t0;
        t0 = time_in_seconds();
        // generate random values from standard normal distribution
        // note: this MKL function is sequential but vectorized
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, 3*npos, buf, 0., 1.);
        // update positions with Brownian displacements
        #pragma omp parallel for schedule(static) shared(pos_orig) private(i) num_threads(NTHREADS)
        for (int i=0; i<3*npos; i++)
        {
            pos_orig[i] += forces[i]*DELTAT+f_const*buf[i];
        }
        t_update_pos += time_in_seconds() - t0;
    }
    printf("--------------------------------------------------------\n");
    printf("Time: %f for initiating the cell head  \n", t_init_cells);
    printf("Time: %f for assigning particles to cells \n", t_assign_to_cells);
    printf("Time: %f for force calculations \n", t_force);
    printf("Time: %f for pos update \n", t_update_pos);
    printf("--------------------------------------------------------\n");
    return 0;
}
