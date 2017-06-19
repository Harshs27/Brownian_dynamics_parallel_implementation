#include <stdlib.h>
#include <stdio.h>
#include <unistd.h> // access
#include <math.h>
#include <assert.h>
#include "timer.h"
#include "bd.h"
#include <omp.h>
#include <mkl.h>

#define NTHREADS 240 
#define M_PI 3.14159265358979323846
#define my_EPS 0.000000001

void print_matrix(double *a, int n){
    for(int i=0;i<5;i++){
        for(int j=0;j<n;j++){
            printf("%lf  ", a[i*n+j]);
        }
        printf("\n\n");
    }
    return;
}


void print_array(double *a, int n){
    for(int i=0;i<n;i++){
        printf("%lf  ", a[i]);
    }
    printf("\n");
    return;
}


//****************************** RPY_EWALD part *****************************************************

inline void scalar_rpy_ewald_real(double r, double xi, double a3, double *m11, double *m12)
{
    double a = 1.;

    double xi2 = xi*xi;
    double xi3 = xi2*xi;
    double xi5 = xi3*xi2;
    double xi7 = xi5*xi2;

    double r2 = r*r;
    double r4 = r2*r2;
    double ri = 1./r;
    double ri2 = ri*ri;
    double ri3 = ri*ri2;

    double erfc_xi_r = erfc(xi*r);
    double pi_exp = 1./sqrt(M_PI) * exp(-xi2*r2);
    *m11 = (0.75*a*ri + 0.5*a3*ri3)*erfc_xi_r + ( 4*xi7*a3*r4 + 3*xi3*a*r2 - 20*xi5*a3*r2 - 4.5*xi*a + 14*xi3*a3 +   xi*a3*ri2)*pi_exp;
    *m12 = (0.75*a*ri - 1.5*a3*ri3)*erfc_xi_r + (-4*xi7*a3*r4 - 3*xi3*a*r2 + 16*xi5*a3*r2 + 1.5*xi*a -  2*xi3*a3 - 3*xi*a3*ri2)*pi_exp;
}


inline void scalar_rpy_ewald_recip(double k, double xi, double *m2)
{
    double a = 1.;
    double a3 = 1.;

    double k2 = k*k;
    double xii2k2 = k2/(xi*xi);

    *m2 = (1. + 0.25*xii2k2 + 0.125*xii2k2*xii2k2) * 6.*M_PI/k2 * exp(-0.25*xii2k2);
}

// note: positions must be wrapped inside the box [0,L]
int rpy_ewald(int np, double * restrict a, const double * restrict pos, double L, const double * restrict rad, double xi, int nr, int nk)
{
//    printf("Inside function rpy_ewald\n");
    __declspec(align(64)) double rvec[8];
    __declspec(align(64)) double rvec0[8];
    __declspec(align(64)) double temp[8];

    double a3;

    double m11, m12, m2;
    double eye3_coef;
    double r2, r;

    int x, y, z;
    int i, j;

    double *ap0, *ap;

    int vsize = ((2*nk+1)*(2*nk+1)*(2*nk+1) - 1) / 2;
#define VSIZE ((2*6+1)*(2*6+1)*(2*6+1) - 1) / 2
//    int A_VSIZE = ceil(VSIZE/8.0)*8;
//    int K_VSIZE = ceil(3*VSIZE/8.0)*8;
//    printf("check vsize=%d\n", A_VSIZE);
    __declspec(align(64)) double k_array[VSIZE];//1104
    __declspec(align(64)) double m2_array[VSIZE];//1104
    __declspec(align(64)) double kvec_array[3*VSIZE];//3296
    int ind;

    __declspec(align(64)) double kvec[8];
    double k;
    double t;

    double vinv = 1./(L*L*L);

    double time0, time1;
    double time0_real, time1_real;
    double time0_recip, time1_recip;

    // INDICES for converting for loops 
    int _b, _index, ib, ib2;
    // *************************************************************************
    //     // compute and save coefficients for reciprocal-space sum
    //     // Due to symmetry, only need half of the grid points
    ind = 0;
    _b = (2*nk+1);
    for (_index =0 ;_index < (_b*_b*_b -1)/2; _index++){// Using indices x,y,z are recalculated
        z = _index%(_b)-nk;// adjusting the indices
        x = (_index-_index%(_b*_b))/(_b*_b)-nk;
        y = (_index%(_b*_b)-_index%(_b))/_b-nk;
        k_array[ind] = 2.*M_PI/L*sqrt((double)(x*x + y*y + z*z));
        scalar_rpy_ewald_recip(k_array[ind], xi, &m2_array[ind]);
        kvec_array[3*ind  ] = 2.*M_PI/L*x;
        kvec_array[3*ind+1] = 2.*M_PI/L*y;
        kvec_array[3*ind+2] = 2.*M_PI/L*z;
        ind++;
    }

    #pragma omp parallel for schedule(static) num_threads(NTHREADS) private(i, j, ap, ap0, _b, temp, eye3_coef, _index, rvec0, rvec, x, y, z, r, r2, m11, m12, a3 )
    for (int _index1 = np*(np-1)/2-1; _index1>=0; _index1--){
        i = np-1-(int)((1+sqrt(8*_index1+1))/2);
        j = np-1-_index1 + (int)((1+sqrt(8*_index1+1))/2)*((int)((1+sqrt(8*_index1+1))/2)-1)/2;

        temp[0] = 0.;
        temp[1] = 0.;  temp[3] = 0.;
        temp[2] = 0.;  temp[4] = 0.;  temp[5] = 0.;
        eye3_coef = 0.;


        rvec0[0] = pos[3*i] - pos[3*j];
        rvec0[1] = pos[3*i+1] - pos[3*j+1];
        rvec0[2] = pos[3*i+2] - pos[3*j+2];

        a3 = 0.5*(rad[i]*rad[i] + rad[j]*rad[j]); 

        _b = (2*nr+1);
//shared(eye3_coef, temp, rvec0, L, xi, a3, m11, m12, _b, xi3, xi5, xi7, xi)
////        #pragma omp parallel for schedule(static) private(rvec, x, y, z, r, r2, m11, m12) shared(eye3_coef, temp, rvec0, a3)
        for (_index =0 ;_index < _b*_b*_b; _index++){
            z =_index%(_b)-nr;// adjusting the indices
            x = (_index-_index%(_b*_b))/(_b*_b)-nr;
            y = (_index%(_b*_b)-_index%(_b))/_b-nr;
                                                                                    
            rvec[0] = rvec0[0] + x*L;
            rvec[1] = rvec0[1] + y*L;
            rvec[2] = rvec0[2] + z*L;

            // compute norm
            r2 = rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2];
            r  = sqrt(r2);


            rvec[0] /= r;
            rvec[1] /= r;
            rvec[2] /= r;
            scalar_rpy_ewald_real(r, xi, a3, &m11, &m12);

            eye3_coef += m11;
            temp[0] += m12 * rvec[0] * rvec[0];
            temp[1] += m12 * rvec[0] * rvec[1];
            temp[2] += m12 * rvec[0] * rvec[2];
            temp[3] += m12 * rvec[1] * rvec[1];
            temp[4] += m12 * rvec[1] * rvec[2];
            temp[5] += m12 * rvec[2] * rvec[2];
        }

        // add contribution to eye3 term
        temp[0] += eye3_coef;
        temp[3] += eye3_coef;
        temp[5] += eye3_coef;
        // sum into global matrix (only lower-triangular part)
        // // Use matlab to add transpose

         ap0  = &a[np*3*3*i + 3*j];
         ap   = ap0;
        *ap++ = temp[0];
        *ap++ = temp[1];
        *ap   = temp[2];
         ap   = ap0+np*3;
        *ap++ = temp[1];
        *ap++ = temp[3];
        *ap   = temp[4];
         ap   = ap0+np*3+np*3;
        *ap++ = temp[2];
        *ap++ = temp[4];
        *ap   = temp[5];

    }

    // reciprocal-space sum
    #pragma omp parallel for schedule(static) num_threads(NTHREADS) private(i, j, temp, ap, ap0, ind, rvec, kvec, k, m2, t, a3)
    for (_index = np*(np+1)/2-1; _index>=0; _index--){
        i = np-1-(int)((-1+sqrt(8*_index+1))/2);
        j = np-1-_index + (int)((-1+sqrt(8*_index+1))/2)*((int)((-1+sqrt(8*_index+1))/2)+1)/2;
        rvec[0] = pos[3*i+0] - pos[3*j];
        rvec[1] = pos[3*i+1] - pos[3*j+1];
        rvec[2] = pos[3*i+2] - pos[3*j+2];

        temp[0] = 0.;
        temp[1] = 0.;  temp[3] = 0.;
        temp[2] = 0.;  temp[4] = 0.;  temp[5] = 0.;

        a3 = 0.5*(rad[i]*rad[i] + rad[j]*rad[j]);

        for (ind=0; ind<vsize; ind++)
        {
            k = k_array[ind];
            m2 = m2_array[ind];
            kvec[0] = kvec_array[3*ind  ];
            kvec[1] = kvec_array[3*ind+1];
            kvec[2] = kvec_array[3*ind+2];


            t = 2.*vinv*m2*cos(kvec[0]*rvec[0] + kvec[1]*rvec[1] + kvec[2]*rvec[2])*(1.-a3*k*k/3.);


            kvec[0] /= k;
            kvec[1] /= k;
            kvec[2] /= k;

            temp[0] += t * (1. - kvec[0]*kvec[0]);
            temp[1] += t *     - kvec[0]*kvec[1];
            temp[2] += t *     - kvec[0]*kvec[2];
            temp[3] += t * (1. - kvec[1]*kvec[1]);
            temp[4] += t *     - kvec[1]*kvec[2];
            temp[5] += t * (1. - kvec[2]*kvec[2]);
        }

        // sum into matrix
        // // sum with existing values
         ap0   = &a[np*3*3*i + 3*j];
         ap    = ap0;
        *ap++ += temp[0];
        *ap++ += temp[1];
        *ap   += temp[2];
         ap    = ap0+np*3;
        *ap++ += temp[1];
        *ap++ += temp[3];// diagonal element
        *ap   += temp[4];
         ap    = ap0+np*3+np*3;
        *ap++ += temp[2];
        *ap++ += temp[4];
        *ap   += temp[5];// diagonal element
    }        

    // self-part
    for (i=0; i<np; i++)// adding some term to diagonal 
    {
        t = 1./rad[i] - (6. - 40./3.*xi*xi*rad[i]*rad[i])*xi/sqrt(M_PI);
        t *= 0.5;
        for (j=0; j<3; j++)
        {
            ind = 3*i+j;
            a[ind*np*3+ind] = a[ind*np*3+ind]*0.5+t;// taking care of (i==j) condition 
        }
    }
    return 0;
}


//**************************************************************************************************






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


/*
// CHECK RPY*************

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
    printf("a = %lf\n", a[3]);
    printf("gold = %lf\n", gold[3]);
    for (int i=0; i<npos; i++) {
        for (int j=0; j<npos; j++) {
            double diff = a[i*(npos*3) + j] - gold[i*(npos*3) +j];
            err += diff*diff;
//            if(err>0){printf("error at position: i=%d j=%d and err = %lf\n", i, j, err);}
//            printf("error at position: i=%d j=%d and err = %lf\n", i, j, err);
        }
    }
    return err;
}
// **********************
*/
int bd(int npos, double * restrict pos_orig, double * restrict buf, const int *types, double L, double * restrict pos, int* restrict next, double* restrict forces, double f_const, double * restrict au, double * restrict rad, double xi, int nr, int nk, double * restrict hd_vec)
{


/*
    //************************** CHECK RPY part ***************************************************
    printf("npos = %d, L= %lf\n", npos, L);

    char *gold_filename = "gold.dat";
    double *gold = (double *) _mm_malloc((3*npos) * (3*npos) * sizeof(double), 64);

    if (access(gold_filename, F_OK) == -1) {
        printf("[WARNING] Unable to access gold file \"%s\"; comparison will not proceed.\n", gold_filename);
    } else {
        gold_read(gold_filename, npos, gold);
    }
    rpy_ewald(npos, au, pos_orig, L, rad, xi, nr, nk);// DELETE after testing 
    double error = compare_gold(npos, au, gold);
    printf("Squared Error: %f\n", error);
    return 500;
    //*********************************************************************************************

*/

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
    double t0, t_init_cells = 0, t_assign_to_cells=0, t_update_pos=0, t_force=0, t_hd = 0, t_cho = 0;    
    for (int step=0; step<INTERVAL_LEN; step++)
    {
//        printf("step = %d\n", step);
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
//        #pragma omp parallel for schedule(static) private(i, idx, idy, idz, bp) shared(b, next) num_threads(NTHREADS)
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
//        printf("Calculating the Hydrodynamic Interations for the given particle positions\n");
        // au = upper triangular matrix with hydrodynamic interaction values
        // pos = wrapped up position inside the box_width = L;
        // rad = radius of particles; xi, nr, nk are constants.
        
        for (int p1=0; p1<3*npos*3*npos; p1++){
            au[p1] = 0;
        } 
        rpy_ewald(npos, au, pos, L, rad, xi, nr, nk);
        t_hd += time_in_seconds() - t0;
//        print_matrix(au, 3*npos);
//        printf("Getting the cholesky decomposition\n");
        t0 = time_in_seconds();
        LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', 3*npos, au, 3*npos);
        t_cho += time_in_seconds() - t0;
        // Get interations vector by multiplying l_cols by buf)
//        print_matrix(au, 3*npos);
//        print_matrix(au, 3*npos);
//        printf("Multiplying by random gaussian vector \n");
        t0 = time_in_seconds();
        // generate random values from standard normal distribution
        // note: this MKL function is sequential but vectorized
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, 3*npos, buf, 0., 1.);
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, 3*npos, 1, 3*npos, 1, au, 3*npos, buf, 3*npos, 0, hd_vec, 1);
//        print_array(buf, 3*npos);
//        printf("printing the correlation vector\n");
//        print_array(hd_vec, 3*npos);
        // update positions with Brownian displacements
        #pragma omp parallel for schedule(static) shared(pos_orig) private(i) num_threads(NTHREADS)
        for (int i=0; i<3*npos; i++)
        {
//            pos_orig[i] += forces[i]*DELTAT+f_const*buf[i];
            pos_orig[i] += forces[i]*DELTAT+f_const*hd_vec[i];
        }
        t_update_pos += time_in_seconds() - t0;
    }
    printf("--------------------------------------------------------\n");
    printf("Time: %f for initiating the cell head  \n", t_init_cells);
    printf("Time: %f for assigning particles to cells \n", t_assign_to_cells);
    printf("Time: %f for force calculations \n", t_force);
    printf("Time: %f for hydrodynamic \n", t_hd);
    printf("Time: %f for cholesky \n", t_cho);
    printf("Time: %f for pos update \n", t_update_pos);
    printf("--------------------------------------------------------\n");
    return 0;
}
