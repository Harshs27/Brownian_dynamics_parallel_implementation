#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
//#include <mkl.h>

void print_array(int *x, int len){
    for(int i=0; i<len; i++){
        printf("%d\t", x[i]);
    }
    printf("\n");
    return;
}

void print_matrix(int * a, int row, int col){
    printf("\n");
    for(int i=0; i<row; i++){
        for(int j=0; j<col; j++){
            printf("%d\t", a[i*col+j]);
        }
        printf("\n");
    }
    return;
}

void print_matrix_d(double * a, int row, int col){
    printf("\n");
    for(int i=0; i<row; i++){
        for(int j=0; j<col; j++){
            printf("%lf\t", a[i*col+j]);
        }
        printf("\n");
    }
    return;
}

void laplacian2d(int * a, int m, int n){
    int *e = (int *)malloc(m*n*sizeof(int));
    int *b1 = (int *)malloc(m*n*sizeof(int));
    int *b2 = (int *)malloc(m*n*sizeof(int));

    for(int i=0; i<m*n; i++){
        e[i] = -1;
        b1[i] = -1;
        b2[i] = -1;
    }
    for(int i=m-1; i<n*m; i=i+m){
        b1[i] = 0;
        b2[i-m+1] = 0;
    }
//    print_array(b1, m*n);
//    print_array(b2, m*n);
    
    // NOTE: [-m, -1, 0, 1, m] has to be imitated here
    for(int x=0; x<m*n; x++){// all rows
        for(int y=0; y<m*n; y++){// all cols
            a[x*(m*n)+y] = 0; //NOTE: insert code for MKL sparse format! 
            if(x==y){// diagonal element -4*e
                a[x*(m*n)+y] = -4*e[y];
            }
            else if (y == x+1){// b2 term
                a[x*(m*n)+y] = b2[y];
            }
            else if (y == x+m){
                a[x*(m*n)+y] = -1;
            }
            else if (y == x-1){// b1 term
                a[x*(m*n)+y] = b1[y];
            }
            else if (y == x-m){
                a[x*(m*n)+y] = -1;
            }
        }
    }
    free(e);
    free(b1);
    free(b2);
    return;
}
double get_rand(int a, int b){//gen random num between (a, b)
    return a+(b-a)*(double)rand() / (double)((unsigned)RAND_MAX + 1);
}

void rand_init(double* A, double* B, int row, int col){ 
    for (int i=0; i<row; i++){
        for(int j=0;j<col; j++){
            A[i*col+j] = get_rand(0, 1)-0.5;
            B[i*col+j] = A[i*col+j];
        }
    }     
    return;
}

void rand_init_single(double* A, int row, int col){ 
    for (int i=0; i<row; i++){
        for(int j=0;j<col; j++){
            A[i*col+j] = get_rand(0, 1)-0.5;
        }
    }     
    return;
}

void zero_init(double* A, int row, int col){ 
    for (int i=0; i<row; i++){
        for(int j=0;j<col; j++){
            A[i*col+j] = 0;
        }
    }     
    return;
}

void copy_matrix(double* A, double* B, int row, int col){// copy A to B
    for (int i=0; i<row; i++){
        for(int j=0;j<col; j++){
            B[i*col+j] = A[i*col+j];
        }
    }     
    return;
}

void jacobi_schwarz(int m, int p, int q, int grow){
    int * aglobal          = (int *)malloc((p*m)*(p*m)*(q*m)*(q*m)*sizeof(int));
    int * alocal           = (int *)malloc((m+2*grow)*(m+2*grow)*(m+2*grow)*(m+2*grow)*sizeof(int));
    int * colset           = (int *)malloc(m*sizeof(int));
    int * colset_expanded  = (int *)malloc((m+2*grow)*sizeof(int));
    int * rowset           = (int *)malloc(m*sizeof(int));
    int * rowset_expanded  = (int *)malloc((m+2*grow)*sizeof(int));

    int flag_colset_exp, flag_rowset_exp, colset_exp_len, rowset_exp_len;
    int ind, starti, startj;

    laplacian2d(aglobal, p*m, q*m);
    // just calculate the total number of points in the grid (say t) then t by t is the dimension of A
//    print_matrix(aglobal, (p*m)*(q*m), (p*m)*(q*m));

    double * rhs           = (double*)malloc((p*m)*(q*m)*sizeof(double));
    double * rhslocal      = (double*)malloc((m+2*grow)*(m+2*grow)*sizeof(double));
    double * temp          = (double*)malloc((m+2*grow)*(m+2*grow)*sizeof(double));
    double * oldsol        = (double*)malloc((p*m)*(q*m)*sizeof(double));
    double * newsol        = (double*)malloc((p*m)*(q*m)*sizeof(double));
    rand_init_single(rhs, (p*m), (q*m));
//    zero_init(rhs, (p*m), (q*m)); // initialising the rhs to 0 NOTE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print_matrix_d(rhs, p*m, q*m);
    rand_init(newsol, oldsol, (p*m), (q*m));// and old and new sol to some random values
/*
    print_matrix_d(newsol, p*m, q*m);
    print_matrix_d(oldsol, p*m, q*m);
    print_matrix_d(oldsol, p*m, q*m);
*/  


    printf("matrices initialised\n");
//    assert(colset);
    // Skipping the exact solution calculation
    for (int sweep=0; sweep<1; sweep++){
        for(int i=0; i<p; i++){
            for(int j=0; j<q; j++){
/*
// DELETE : FOR TESTING only
for(int k=0; k<m+2*grow; k++){
    colset_expanded[k] = -1;
    rowset_expanded[k] = -1;
}
*/
                printf("i=%d and j=%d\n", i, j);
                for(int k=0; k<m; k++){// resetting the size of colset_expanded for each loop
                    colset[k] = i*m+1+k;
                    rowset[k] = j*m+1+k;
                    colset_expanded[k] = colset[k];
                    rowset_expanded[k] = rowset[k];
                }
                colset_exp_len = m;
                rowset_exp_len = m;
                // if the array expands to left the grow size will increase for the remaining loop
                flag_colset_exp = 0;
                flag_rowset_exp = 0;

                // growing on all possible sides by grow
                if(i>0){// grow on the left
                    for (int k=0; k<grow+m; k++){
                        if(k<grow){
                            colset_expanded[k] = -grow+i*m+1+k;
                        }
                        else{
                            colset_expanded[k] = colset[k-grow];
                        }
                    }
//                    print_array(colset_expanded, m+2*grow);
                    colset_exp_len+=grow;
                    flag_colset_exp = 1;
                }

                if(i < p-1){// grow on right
                    // Find the length of colset_expanded
                    if (flag_colset_exp==0){
                        for(int k=0; k<m+grow; k++){
                            if(k<m){
                                colset_expanded[k] = colset[k];
                            }
                            else{
                                colset_expanded[k] = (i+1)*m+1+k-m;
                            }
                        }
                        colset_exp_len+=grow;
//                        print_array(colset_expanded, m+2*grow);
                    }
                    else{// flag is 1
                        for(int k=m+grow; k<m+2*grow; k++){
                            colset_expanded[k] = (i+1)*m+1+k-m-grow;
                        }
                        colset_exp_len+=grow;
//                        print_array(colset_expanded, m+2*grow);
                    }
                }

                if(j>0){// grow on the left
                    for (int k=0; k<grow+m; k++){
                        if(k<grow){
                            rowset_expanded[k] = -grow+j*m+1+k;
                        }
                        else{
                            rowset_expanded[k] = rowset[k-grow];
                        }
                    }
                    flag_rowset_exp=1;
                    rowset_exp_len+=grow;
//                    print_array(rowset_expanded, m+2*grow);
                }

                if(j < q-1){// grow on right
                    if (flag_rowset_exp == 0){
                        for(int k=0; k<m+grow; k++){
                            if(k<m){
                                rowset_expanded[k] = rowset[k];
                            }
                            else{
                                rowset_expanded[k] = (j+1)*m+1+k-m;
                            }
                        }
                        rowset_exp_len+=grow;
//                        print_array(rowset_expanded, m+2*grow);
                    }
                    else{// flag == 1, already expanded in the opp direction
                        for(int k=m+grow; k<m+2*grow; k++){
                            rowset_expanded[k] = (j+1)*m+1+k-m-grow;
                        }
                        rowset_exp_len+=grow;
//                        print_array(rowset_expanded, m+2*grow);
                    }
                }
                // CHECKED OK!
//    print_matrix_d(rhs, p*m, q*m);

                // ADD the RHS local part
//                double * rhslocal      = (double*)malloc((rowset_exp_len)*(colset_exp_len)*sizeof(double));
                // getting the length of rowset and colset expanded
                for(int c=0; c<colset_exp_len; c++){
                    printf("c=%d\t", colset_expanded[c]);
                    for(int r=0; r<rowset_exp_len; r++){
                        printf("r = %d\t", rowset_expanded[r]);
                        rhslocal[c*rowset_exp_len+r] = rhs[(colset_expanded[c]-1)*(q*m)+(rowset_expanded[r]-1)];
//                        rhslocal[c*rowset_exp_len+r] = rhs[c*(3)+r];
                    }
                    printf("\n");
                }
                print_matrix_d(rhslocal, colset_exp_len, rowset_exp_len);
//                free(rhslocal);
                // CHECKED OK!

                // Modify the RHS with the current solution
                if(i>0){ // TOP boundary
                    printf("Top boundary\n");
                    print_matrix_d(oldsol, p*m, q*m);
                    ind = i*m-grow-1;// -1 to adjust index
                    for(int k=0; k<rowset_exp_len; k++){
                        rhslocal[k] = rhslocal[k] + oldsol[(ind)*(q*m)+(rowset_expanded[k]-1)];
                    }
                    print_matrix_d(rhslocal, colset_exp_len, rowset_exp_len);
                }
                
                if(j>0){// LEFT boundary
                    printf("Left boundary\n");
                    print_matrix_d(oldsol, p*m, q*m);
                    ind = j*m-grow-1;
                    for(int k=0; k<colset_exp_len; k++){
                        rhslocal[k*rowset_exp_len] = rhslocal[k*rowset_exp_len]+oldsol[(colset_expanded[k]-1)*(q*m) + ind];
                    }
                    print_matrix_d(rhslocal, colset_exp_len, rowset_exp_len);
                }

                if(i<p-1){// BOTTOM boundary
                    printf("Bottom boundary\n");
                    print_matrix_d(oldsol, p*m, q*m);
                    ind = (i+1)*m+grow;// adjusted the index
                    for(int k=0; k<rowset_exp_len; k++){
                        rhslocal[(colset_exp_len-1)*(rowset_exp_len)+k] = rhslocal[(colset_exp_len-1)*(rowset_exp_len)+k]+oldsol[ind*(q*m)+(rowset_expanded[k]-1)];
                    }
                    print_matrix_d(rhslocal, colset_exp_len, rowset_exp_len);
                }

                if(j<q-1){// RIGHT boundary
                    printf("Right boundary\n");
                    print_matrix_d(oldsol, p*m, q*m);
                    ind = (j+1)*m+grow;
                    for(int k=0; k<colset_exp_len; k++){
                        rhslocal[k*rowset_exp_len+(rowset_exp_len-1)] = rhslocal[k*rowset_exp_len+(rowset_exp_len-1)] + oldsol[(colset_expanded[k]-1)*(q*m)+ind];
                    }
                    print_matrix_d(rhslocal, colset_exp_len, rowset_exp_len);
                }
                
                // SOLVE the local problem : USE MKL???
                laplacian2d(alocal, colset_exp_len, rowset_exp_len);
                printf("colset len = %d and rowset len = %d\n", colset_exp_len, rowset_exp_len);
                print_matrix(alocal,  colset_exp_len*rowset_exp_len, colset_exp_len*rowset_exp_len);
                // NOTE: The alocal matrix should be represented in Sparse format!
                // SOLVE locally and store solution in x = alocal \ rhslocal(:);  *************************IMP_currently incomplete**************
                // SAY that we have a solution and reshaped to colset_len and rowset_len
                rand_init_single(temp, colset_exp_len, rowset_exp_len); // matrix

                starti = colset[0]-colset_expanded[0]; 
                startj = rowset[0]-rowset_expanded[0];
                printf("starti = %d and startj= %d\n ", starti, startj);
                // updating the new solution (only the m by m part)
                print_matrix_d(newsol, p*m, q*m);
                print_matrix_d(temp, colset_exp_len, rowset_exp_len);
                for(int c=0; c<m; c++){
                    printf("c=%d\t", colset[c]);
                    for(int r=0; r<m; r++){
                        printf("r = %d\t", rowset[r]);
                        newsol[(colset[c]-1)*(q*m)+(rowset[r]-1)] = temp[(starti+c)*rowset_exp_len+(startj+r)];
                    }
                    printf("\n");
                }
                print_matrix_d(newsol, p*m, q*m);
            }//j
        }//i
        // SOME PRINTF 
        print_matrix_d(newsol, p*m, q*m);

        print_matrix_d(oldsol, p*m, q*m);
        copy_matrix(newsol, oldsol, p*m, q*m);
        print_matrix_d(oldsol, p*m, q*m);
    }//sweep
//    print_array(colset, m);
//    print_array(colset_expanded, m+grow);
    free(aglobal);
    free(alocal);
    free(rhs);
    free(temp);
    free(oldsol);
    free(newsol);
    free(rhslocal);
    free(colset);
    free(colset_expanded);
    free(rowset);
    free(rowset_expanded);
    return;
}

int main(){
//    srand(time(NULL));
    int m, p, q, grow;
    m = 4;
    p = 3;
    q = 3;
    grow = 2;
    if (grow>m){
        printf("grow cannot be greater than m... exiting\n");
        exit(0);
    }
    // We have p by q processor grid with each subdomain as m by m
    jacobi_schwarz(m,p,q, grow);
    return 0;
}

