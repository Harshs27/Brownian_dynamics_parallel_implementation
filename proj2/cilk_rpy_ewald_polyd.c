#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "timer.h"
#include <sys/time.h>
#include <cilk/cilk.h>

#define M_PI 3.14159265358979323846
#define NTHREADS 240

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
int rpy_ewald(int np, double * restrict a, 
    const double * restrict pos, double L, const double * restrict rad, double xi, int nr, int nk)
{
    __declspec(align(64)) double rvec[8];
    __declspec(align(64)) double rvec0[8];
    __declspec(align(64)) double temp[8];
//    double temp_0, temp_1, temp_2, temp_3, temp_4, temp_5;

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
    // compute and save coefficients for reciprocal-space sum
    // Due to symmetry, only need half of the grid points
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
//    #pragma omp parallel for schedule(static) num_threads(NTHREADS) private(i, j, ap, ap0, _b, temp, eye3_coef, _index, rvec0, rvec, x, y, z, r, r2, m11, m12, a3 )
    cilk_for (int _index1 = np*(np-1)/2-1; _index1>=0; _index1--){
        int i, j, _b, _index, x, y, z;
        double *ap, *ap0, eye3_coef, r, r2, m11, m12, a3; 
        __declspec(align(64)) double rvec[8];
        __declspec(align(64)) double rvec0[8];
        __declspec(align(64)) double temp[8];
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
//        #pragma omp parallel for schedule(static) private(rvec, x, y, z, r, r2, m11, m12) shared(eye3_coef, temp, rvec0, a3)
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
        // Use matlab to add transpose

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
    // *************************************************************************
    // reciprocal-space sum
//    #pragma omp parallel for schedule(static) num_threads(NTHREADS) private(i, j, temp, ap, ap0, ind, rvec, kvec, k, m2, t, a3)
    cilk_for (_index = np*(np+1)/2-1; _index>=0; _index--){
        int i, j, ind; 
        double *ap, *ap0, k, m2, t, a3;
        __declspec(align(64)) double temp[8];
        __declspec(align(64)) double rvec[8];
        __declspec(align(64)) double kvec[8];
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
        // sum with existing values

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
    // *************************************************************************
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
