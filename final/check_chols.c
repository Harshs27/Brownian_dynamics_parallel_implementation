//compile : icc -std=c99 -mkl check_chols.c

#include <stdio.h>
#include <mkl.h>

void print_matrix(double *a, int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            printf("%lf  ", a[i*n+j]);
        }
        printf("\n");
    }
    return;
}


int main(){
    int n = 3;
    double a[9] = {25, 15, -5, 0, 18, 0, 0, 0, 11};
    double b[3] = {1,1,1};
    LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', n, a, n);
    print_matrix(a, n);// a = upper matrix
    double *c = (double *)malloc(n*sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, n, 1, n, 1, a, n, b, n, 0, c, 1);
    printf("\n");
    for(int i =0;i<n;i++){
        printf("%lf  ", c[i]);
    }
    printf("\n");
    return 1;
}

