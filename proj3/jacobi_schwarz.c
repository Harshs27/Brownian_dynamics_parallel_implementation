#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mkl.h>
#include <mpi.h>
#include "mkl_dss.h"
#include "mkl_types.h"
#include <math.h>
//#include "timer.h"
#include <sys/time.h>


#define M_DOMAIN 100
#define P_MESH 10 
#define Q_MESH 6 
#define GROW_DOMAIN 1
#define SWEEP 500

MKL_INT solver(_MKL_DSS_HANDLE_t handle, MKL_INT opt, MKL_INT nRows,  MKL_INT nCols,  MKL_INT nNonZeros,  MKL_INT nRhs, _INTEGER_t *rowIndex, _INTEGER_t* columns, _DOUBLE_PRECISION_t* rhs, _DOUBLE_PRECISION_t* solValues){

    MKL_INT opt1;
    MKL_INT i, j;
    _INTEGER_t error;
    _CHARACTER_t statIn[] = "determinant", *uplo;
    _DOUBLE_PRECISION_t statOut[5], eps = 1e-6;
    for (i = 0; i < 1; i++){
        if (i == 0)
        {
            uplo = "non-transposed";
            opt1 = MKL_DSS_DEFAULTS;
        }
        else if (i == 1)
        {
            uplo = "transposed";
            opt1 = MKL_DSS_TRANSPOSE_SOLVE;
        }
        else        // Conjugate transposed == transposed for real matrices
        if (i == 2)
        {
          uplo = "conjugate transposed";
          opt1 = MKL_DSS_CONJUGATE_SOLVE;
        }
        // Nullify solution on entry (for sure)
        for (j = 0; j < nCols; j++){
        solValues[j] = 0.0;
        }
        // Apply trans or non-trans option, solve system
        opt |= opt1;
        error = dss_solve_real (handle, opt, rhs, nRhs, solValues);
        if (error != MKL_DSS_SUCCESS)
        goto printError;
        opt &= ~opt1;
    }
    return 0;
    printError:
    printf ("Solver returned error code %d\n", error);
    exit (1);
}



void set_solver(_MKL_DSS_HANDLE_t* handle, MKL_INT* opt, MKL_INT* type, MKL_INT* sym, MKL_INT nRows,  MKL_INT nCols,  MKL_INT nNonZeros,  MKL_INT nRhs,  MKL_INT *rowIndex,  MKL_INT * columns, double *values){
    /* Allocate storage for the solver handle and the right-hand side. */
    _INTEGER_t error;
    /* --------------------- */
    /* Initialize the solver */
    /* --------------------- */
    error = dss_create (*handle, *opt);
    if (error != MKL_DSS_SUCCESS)
    goto printError;

    /* ------------------------------------------- */
    /* Define the non-zero structure of the matrix */
    /* ------------------------------------------- */
    error = dss_define_structure (*handle, *sym, rowIndex, nRows, nCols, columns, nNonZeros);
    if (error != MKL_DSS_SUCCESS)
    goto printError;
    /* ------------------ */
    /* Reorder the matrix */
    /* ------------------ */
    error = dss_reorder (*handle, *opt, 0);
    if (error != MKL_DSS_SUCCESS)
    goto printError;
    /* ------------------ */
    /* Factor the matrix  */
    /* ------------------ */
    error = dss_factor_real (*handle, *type, values);
    if (error != MKL_DSS_SUCCESS)
    goto printError;
    return;
    printError:
    printf ("Solver returned error code %d\n", error);
    exit (1);
}


void convert_column_major(double * A, double *B, int row, int col){
    int k = 0;
    for (int i=0; i<col; i++){
        for(int j=0;j<row; j++){
            B[k++] = A[j*col+i];
        }
    }
    return;
}

void print_array(int *x, int len){
    for(int i=0; i<len; i++){
        printf("%d\t", x[i]);
    }
    printf("\n");
    return;
}

void print_array_d(double *x, double len){
    for(int i=0; i<len; i++){
        printf("%lf\t", x[i]);
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


int csr_laplacian2d(int nRows, int nCols, int *rowIndex, int *columns, double* values, int *e, int *b1, int *b2){// Returns number of nonzeros
    int ri = 1, ci = 0, vi = 0, num_elements_row;
    int m, n;    
    m = nRows, n = nCols;
//    printf("m=%d and n=%d\n", m, n);
    for(int i=0; i<m*n; i++){
        e[i] = -1;
        b1[i] = -1;
        b2[i] = -1;
    }
    for(int i=m-1; i<n*m; i=i+m){
//        printf("i=%d\n", i);
        b1[i] = 0;
        b2[i-m+1] = 0;
    }
    // NOTE: [-m, -1, 0, 1, m] has to be imitated here
    rowIndex[0] = 1;
//    printf("something\n");
    for(int x=0; x<m*n; x++){// all rows
        num_elements_row = 0;
        for(int y=0; y<m*n; y++){// all cols
            if(x==y){// diagonal element -4*e
                printf("");
                values[vi++] = -4*e[y];
                columns[ci++] = y+1;// adjusting 1 for the solver function
                num_elements_row++;
            }
            else if (y == x+1){// b2 term
                values[vi++] = b2[y];
                columns[ci++] = y+1;
                num_elements_row++;
            }
            else if (y == x+m){
                values[vi++] = -1;
                columns[ci++] = y+1;
                num_elements_row++;
            }
            else if (y == x-1){// b1 term
                values[vi++] = b1[y];
                columns[ci++] = y+1;
                num_elements_row++;
            }
            else if (y == x-m){
                values[vi++] = -1;
                columns[ci++] = y+1;
                num_elements_row++;
            }
            if (num_elements_row==5){
                break;
            }
        }
        rowIndex[ri] = rowIndex[ri-1]+num_elements_row;
        ri++;
    }
    int NNONZEROS = rowIndex[nCols*nRows]-1;
//    print_array_d(values, NNONZEROS);
//    print_array(columns, NNONZEROS);
//    print_array(rowIndex, m*n+1);
    return NNONZEROS;
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

int main(int argc, char * argv[]){
//    srand(time(NULL));

    double mytime;
    int m, p, q, grow;
    m = M_DOMAIN;
    p = P_MESH;
    q = Q_MESH;
    grow = GROW_DOMAIN;
    if (grow>m){
        printf("grow cannot be greater than m... exiting\n");
        exit(0);
    }
    // We have p by q processor grid with each subdomain as m by m
    // MPI program start 
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, size;
    MPI_Comm_size(comm, &size);
    // NOTE: Fix this issue!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if(p*q!=size){
        printf("the number of processes do not match the processor mesh (p*q = %d and n=%d)!.. fix that.. exiting!\n", p*q, size);
        exit(0);
    }
    MPI_Comm_rank(comm, &rank);
 
    int * colset           = (int *)malloc(m*sizeof(int));
    int * colset_expanded  = (int *)malloc((m+2*grow)*sizeof(int));
    int * rowset           = (int *)malloc(m*sizeof(int));
    int * rowset_expanded  = (int *)malloc((m+2*grow)*sizeof(int));

    double * rhslocal      = (double*)malloc((m+2*grow)*(m+2*grow)*sizeof(double));
    double * rhslocal_c      = (double*)malloc((m+2*grow)*(m+2*grow)*sizeof(double));// column major 
    double * oldsol        = (double*)malloc((m*m)*sizeof(double)); //storing only the local solution
    double * newsol        = (double*)malloc((m*m)*sizeof(double));

    double * receive_rhs_bottom     = (double *)malloc((m+2*grow)*sizeof(double));
    double * receive_rhs_bottom_L   = (double *)malloc((grow)*sizeof(double));
    double * receive_rhs_bottom_C   = (double *)malloc((m)*sizeof(double));
    double * receive_rhs_bottom_R   = (double *)malloc((grow)*sizeof(double));

    double * receive_rhs_top        = (double *)malloc((m+2*grow)*sizeof(double));
    double * receive_rhs_top_L      = (double *)malloc((grow)*sizeof(double));
    double * receive_rhs_top_C      = (double *)malloc((m)*sizeof(double));
    double * receive_rhs_top_R      = (double *)malloc((grow)*sizeof(double));

    double * receive_rhs_left     = (double *)malloc((m+2*grow)*sizeof(double));
    double * receive_rhs_left_U     = (double *)malloc((grow)*sizeof(double));
    double * receive_rhs_left_C     = (double *)malloc((m)*sizeof(double));
    double * receive_rhs_left_B     = (double *)malloc((grow)*sizeof(double));

    double * receive_rhs_right    = (double *)malloc((m+2*grow)*sizeof(double));
    double * receive_rhs_right_U    = (double *)malloc((grow)*sizeof(double));
    double * receive_rhs_right_C    = (double *)malloc((m)*sizeof(double));
    double * receive_rhs_right_B    = (double *)malloc((grow)*sizeof(double));

    double * send_rhs_bottom_L   = (double *)malloc((grow)*sizeof(double));
    double * send_rhs_bottom_C   = (double *)malloc((m)*sizeof(double));
    double * send_rhs_bottom_R   = (double *)malloc((grow)*sizeof(double));

    double * send_rhs_top_L      = (double *)malloc((grow)*sizeof(double));
    double * send_rhs_top_C      = (double *)malloc((m)*sizeof(double));
    double * send_rhs_top_R      = (double *)malloc((grow)*sizeof(double));

    double * send_rhs_left_U     = (double *)malloc((grow)*sizeof(double));
    double * send_rhs_left_C     = (double *)malloc((m)*sizeof(double));
    double * send_rhs_left_B     = (double *)malloc((grow)*sizeof(double));

    double * send_rhs_right_U    = (double *)malloc((grow)*sizeof(double));
    double * send_rhs_right_C    = (double *)malloc((m)*sizeof(double));
    double * send_rhs_right_B    = (double *)malloc((grow)*sizeof(double));
    rand_init(newsol, oldsol, m, m);// and old and new sol to some random values

    int nRows = (m+2*grow), nCols = (m+2*grow), MEM_SIZE = (nCols*nRows)*5;
    int *columns      = (int *)malloc(MEM_SIZE*sizeof(int));
    int *rowIndex     = (int *)malloc((nCols*nRows+1)*sizeof(int));
    double *values    = (double *)malloc(MEM_SIZE*sizeof(double));
    assert(columns);
    assert(values);
    assert(rowIndex);
    int NNONZEROS, nRhs = 1;

    int *e     = (int *)malloc(nRows*nCols*sizeof(int));
    int *b1    = (int *)malloc(nRows*nCols*sizeof(int));
    int *b2    = (int *)malloc(nRows*nCols*sizeof(int));
    
    double * temp  = (double*)malloc((m+2*grow)*(m+2*grow)*sizeof(double));
    int starti, startj;

    int p_idx_top_L, p_idx_top_C, p_idx_top_R;
    int p_idx_bottom_L, p_idx_bottom_C, p_idx_bottom_R;
    int p_idx_right_U, p_idx_right_C, p_idx_right_B;
    int p_idx_left_U, p_idx_left_C, p_idx_left_B;

    MPI_Request req_top[10], req_bottom[10], req_left[10], req_right[10];
    MPI_Status stat_top[21], stat_bottom[21], stat_left[21], stat_right[21];

    int p_idx_top, r_idx_top, s_idx_top, p_idx_bottom, r_idx_bottom, s_idx_bottom, p_idx_right, r_idx_right, s_idx_right, p_idx_left, r_idx_left, s_idx_left;
   
    int i, j;
    int flag_colset_exp, flag_rowset_exp, colset_exp_len, rowset_exp_len;
    

    _MKL_DSS_HANDLE_t handle;
    MKL_INT opt = MKL_DSS_DEFAULTS;
    MKL_INT sym = MKL_DSS_NON_SYMMETRIC;
    MKL_INT type = MKL_DSS_INDEFINITE;

//        MPI_Barrier(comm);
    // Get the i and j for each processor (i.e. the position in the grid)
    j = rank%q;
    i = (rank-rank%q)/q;
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
        }
        else{// flag is 1
            for(int k=m+grow; k<m+2*grow; k++){
                colset_expanded[k] = (i+1)*m+1+k-m-grow;
            }
            colset_exp_len+=grow;
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
        }
        else{// flag == 1, already expanded in the opp direction
            for(int k=m+grow; k<m+2*grow; k++){
                rowset_expanded[k] = (j+1)*m+1+k-m-grow;
            }
            rowset_exp_len+=grow;
        }
    }

    mytime = MPI_Wtime();
    int sweep;
    for (sweep=0; sweep<SWEEP; sweep++){// ITERATIONS START HERE
        // RHS LOCAL PART
        for(int c=0; c<colset_exp_len; c++){
            for(int r=0; r<rowset_exp_len; r++){
                rhslocal[c*rowset_exp_len+r] = 0;
            }
        }

        // Modify the rhslocal with the boundary conditions from the neighbours
        // ******************** New function for grown top and bottom boundary value updation ******************************
        
       

        if(i>0){ // TOP boundary (communicate with the top processor)
            p_idx_top_L = (i-1)*q+j-1;
            p_idx_top_C = (i-1)*q+j;
            p_idx_top_R = (i-1)*q+j+1;
            s_idx_top = m-grow-1;// -1 to adjust index the index to be sent
            
            // cater to the request from the top_C processor
            MPI_Isend(&s_idx_top, 1, MPI_INT, p_idx_top_C, 0, comm, &req_top[0]);// NOTE: the tag is important!
            MPI_Recv(&r_idx_top, 1, MPI_INT, p_idx_top_C, 0, comm, &stat_top[0]);
            for(int k=0; k<m; k++){
                send_rhs_top_C[k] = oldsol[r_idx_top*m+k];
            }
            MPI_Isend(send_rhs_top_C, m, MPI_DOUBLE, p_idx_top_C, 0, comm, &req_top[1]);

            // Send the index of row to the processor above
            // Receive the row from the idx processor
            MPI_Recv(receive_rhs_top_C, m, MPI_DOUBLE, p_idx_top_C, 0, comm, &stat_top[3]);
            MPI_Wait(&req_top[1], &stat_top[1]);// wait send self
            MPI_Wait(&req_top[0], &stat_top[2]);// wait send self
            if(rowset_exp_len == m){
                // update the receive_rhs_top 
                for(int k=0; k<m; k++){
                    receive_rhs_top[k] = receive_rhs_top_C[k];
                }
            }
            else if(rowset_exp_len == m+2*grow){
                // cater to the request from top_L processor
                MPI_Isend(&s_idx_top, 1, MPI_INT, p_idx_top_L, 0, comm, &req_top[2]);// keep tag as 0
                MPI_Recv(&r_idx_top, 1, MPI_INT, p_idx_top_L, 0, comm, &stat_top[5]);
                for(int k=0;k<grow; k++){
                    send_rhs_top_L[k] = oldsol[r_idx_top*m+k]; 
                }
                MPI_Isend(send_rhs_top_L, grow, MPI_DOUBLE, p_idx_top_L, 0, comm, &req_top[3]);
                MPI_Recv(receive_rhs_top_L, grow, MPI_DOUBLE, p_idx_top_L, 0, comm, &stat_top[6]);
                MPI_Wait(&req_top[3], &stat_top[7]);
                MPI_Wait(&req_top[2], &stat_top[8]);

                // cater to the request from the top_R processor
                MPI_Isend(&s_idx_top, 1, MPI_INT, p_idx_top_R, 0, comm, &req_top[4]);
                MPI_Recv(&r_idx_top, 1, MPI_INT, p_idx_top_R, 0, comm, &stat_top[9]);
                for(int k=m-grow; k<m; k++){// the upper right corner part
                    send_rhs_top_R[k-(m-grow)] = oldsol[r_idx_top*m+k];
                }
                MPI_Isend(send_rhs_top_R, grow, MPI_DOUBLE, p_idx_top_R, 0, comm, &req_top[5]);
                MPI_Recv(receive_rhs_top_R, grow, MPI_DOUBLE, p_idx_top_R, 0, comm, &stat_top[10]);
                MPI_Wait(&req_top[5], &stat_top[11]);
                MPI_Wait(&req_top[4], &stat_top[12]);
                // update the receive_rhs_top
                for(int k=0; k<m+2*grow; k++){
                    if(k<grow){
                        receive_rhs_top[k] = receive_rhs_top_L[k];
                    }
                    else if(k>=grow && k<m+grow){
                        receive_rhs_top[k] = receive_rhs_top_C[k-grow];
                    }
                    else{
                        receive_rhs_top[k] = receive_rhs_top_R[k-(m+grow)];
                    }
                }
            
            }
            else if(rowset_exp_len==m+grow){ // so either at left corner or right corner
                if(j==0){// processor at left boundary
                    // cater to the request of TOP_R processor
                    MPI_Isend(&s_idx_top, 1, MPI_INT, p_idx_top_R, 0, comm, &req_top[6]);
                    MPI_Recv(&r_idx_top, 1, MPI_INT, p_idx_top_R, 0, comm, &stat_top[13]);
                    for(int k=m-grow; k<m; k++){// the upper right corner part
                        send_rhs_top_R[k-(m-grow)] = oldsol[r_idx_top*m+k];
                    }
                    MPI_Isend(send_rhs_top_R, grow, MPI_DOUBLE, p_idx_top_R, 0, comm, &req_top[7]);
                    MPI_Recv(receive_rhs_top_R, grow, MPI_DOUBLE, p_idx_top_R, 0, comm, &stat_top[14]);
                    MPI_Wait(&req_top[7], &stat_top[15]);
                    MPI_Wait(&req_top[6], &stat_top[16]);
                    // update the receive_rhs_top
                    for(int k=0; k<m+grow; k++){
                        if(k<m){
                            receive_rhs_top[k] = receive_rhs_top_C[k];
                        }
                        else{
                            receive_rhs_top[k] = receive_rhs_top_R[k-m];
                        }
                    }
                    
                }
                else if(j==q-1){// processor at right boundary
                    // cater to the request of TOP_L processor
                    MPI_Isend(&s_idx_top, 1, MPI_INT, p_idx_top_L, 0, comm, &req_top[8]);// keep tag as 0
                    MPI_Recv(&r_idx_top, 1, MPI_INT, p_idx_top_L, 0, comm, &stat_top[17]);
                    for(int k=0;k<grow; k++){
                        send_rhs_top_L[k] = oldsol[r_idx_top*m+k];
                    }
                    MPI_Isend(send_rhs_top_L, grow, MPI_DOUBLE, p_idx_top_L, 0, comm, &req_top[9]);
                    MPI_Recv(receive_rhs_top_L, grow, MPI_DOUBLE, p_idx_top_L, 0, comm, &stat_top[18]);
                    MPI_Wait(&req_top[9], &stat_top[19]);
                    MPI_Wait(&req_top[8], &stat_top[20]);
                    // update the receive_rhs_top
                    for(int k=0; k<m+grow; k++){
                        if(k<grow){
                            receive_rhs_top[k] = receive_rhs_top_L[k];
                        }
                        else{
                            receive_rhs_top[k] = receive_rhs_top_C[k-grow];
                        }
                    }
                }
            } 
            // update the top boundary of the rhslocal 
            for(int k=0; k<rowset_exp_len; k++){
                rhslocal[k] = rhslocal[k] + receive_rhs_top[k];//oldsol[(ind)*(q*m)+(rowset_expanded[k]-1)];
            }

        }// TOP BOUNDARY

        if(i<p-1){// BOTTOM boundary
            p_idx_bottom_L = (i+1)*q+j-1; // bottom left processor
            p_idx_bottom_C = (i+1)*q+j;
            p_idx_bottom_R = (i+1)*q+j+1;
            s_idx_bottom = grow;        
    
            // cater to the request from the bottom_C processor
            MPI_Isend(&s_idx_bottom, 1, MPI_INT, p_idx_bottom_C, 0, comm, &req_bottom[0]);// NOTE: the tag is important!
            MPI_Recv(&r_idx_bottom, 1, MPI_INT, p_idx_bottom_C, 0, comm, &stat_bottom[0]);
            for(int k=0; k<m; k++){
                send_rhs_bottom_C[k] = oldsol[r_idx_bottom*m+k];
            }
            MPI_Isend(send_rhs_bottom_C, m, MPI_DOUBLE, p_idx_bottom_C, 0, comm, &req_bottom[1]);

            // Send the index of row to the processor below
            // Receive the row from the idx processor
            MPI_Recv(receive_rhs_bottom_C, m, MPI_DOUBLE, p_idx_bottom_C, 0, comm, &stat_bottom[3]);
            MPI_Wait(&req_bottom[1], &stat_bottom[1]);// wait send self
            MPI_Wait(&req_bottom[0], &stat_bottom[2]);// wait send self
            if(rowset_exp_len == m){
                // update the receive_rhs_bottom 
                for(int k=0; k<m; k++){
                    receive_rhs_bottom[k] = receive_rhs_bottom_C[k];
                }
            }
            else if(rowset_exp_len == m+2*grow){
                // cater to the request from bottom_L processor
                MPI_Isend(&s_idx_bottom, 1, MPI_INT, p_idx_bottom_L, 0, comm, &req_bottom[2]);// keep tag as 0
                MPI_Recv(&r_idx_bottom, 1, MPI_INT, p_idx_bottom_L, 0, comm, &stat_bottom[5]);
                for(int k=0;k<grow; k++){
                    send_rhs_bottom_L[k] = oldsol[r_idx_bottom*m+k]; 
                }
                MPI_Isend(send_rhs_bottom_L, grow, MPI_DOUBLE, p_idx_bottom_L, 0, comm, &req_bottom[3]);
                MPI_Recv(receive_rhs_bottom_L, grow, MPI_DOUBLE, p_idx_bottom_L, 0, comm, &stat_bottom[6]);
                MPI_Wait(&req_bottom[3], &stat_bottom[7]);
                MPI_Wait(&req_bottom[2], &stat_bottom[8]);

                // cater to the request from the bottom_R processor
                MPI_Isend(&s_idx_bottom, 1, MPI_INT, p_idx_bottom_R, 0, comm, &req_bottom[4]);
                MPI_Recv(&r_idx_bottom, 1, MPI_INT, p_idx_bottom_R, 0, comm, &stat_bottom[9]);
                for(int k=m-grow; k<m; k++){// the lower right corner part
                    send_rhs_bottom_R[k-(m-grow)] = oldsol[r_idx_bottom*m+k];
                }
                MPI_Isend(send_rhs_bottom_R, grow, MPI_DOUBLE, p_idx_bottom_R, 0, comm, &req_bottom[5]);
                MPI_Recv(receive_rhs_bottom_R, grow, MPI_DOUBLE, p_idx_bottom_R, 0, comm, &stat_bottom[10]);
                MPI_Wait(&req_bottom[5], &stat_bottom[11]);
                MPI_Wait(&req_bottom[4], &stat_bottom[12]);
                // update the receive_rhs_bottom
                for(int k=0; k<m+2*grow; k++){
                    if(k<grow){
                        receive_rhs_bottom[k] = receive_rhs_bottom_L[k];
                    }
                    else if(k>=grow && k<m+grow){
                        receive_rhs_bottom[k] = receive_rhs_bottom_C[k-grow];
                    }
                    else{
                        receive_rhs_bottom[k] = receive_rhs_bottom_R[k-(m+grow)];
                    }
                }
            
            }
            else if(rowset_exp_len==m+grow){ // so either at left corner or right corner
                if(j==0){// processor at left boundary
                    // cater to the request of BOTTOM_R processor
                    MPI_Isend(&s_idx_bottom, 1, MPI_INT, p_idx_bottom_R, 0, comm, &req_bottom[6]);
                    MPI_Recv(&r_idx_bottom, 1, MPI_INT, p_idx_bottom_R, 0, comm, &stat_bottom[13]);
                    for(int k=m-grow; k<m; k++){// the lower right corner part
                        send_rhs_bottom_R[k-(m-grow)] = oldsol[r_idx_bottom*m+k];
                    }
                    MPI_Isend(send_rhs_bottom_R, grow, MPI_DOUBLE, p_idx_bottom_R, 0, comm, &req_bottom[7]);
                    MPI_Recv(receive_rhs_bottom_R, grow, MPI_DOUBLE, p_idx_bottom_R, 0, comm, &stat_bottom[14]);
                    MPI_Wait(&req_bottom[7], &stat_bottom[15]);
                    MPI_Wait(&req_bottom[6], &stat_bottom[16]);
                    // update the receive_rhs_bottom
                    for(int k=0; k<m+grow; k++){
                        if(k<m){
                            receive_rhs_bottom[k] = receive_rhs_bottom_C[k];
                        }
                        else{
                            receive_rhs_bottom[k] = receive_rhs_bottom_R[k-m];
                        }
                    }
                    
                }
                else if(j==q-1){// processor at right boundary
                    // cater to the request of BOTTOM_L processor
                    MPI_Isend(&s_idx_bottom, 1, MPI_INT, p_idx_bottom_L, 0, comm, &req_bottom[8]);// keep tag as 0
                    MPI_Recv(&r_idx_bottom, 1, MPI_INT, p_idx_bottom_L, 0, comm, &stat_bottom[17]);
                    for(int k=0;k<grow; k++){
                        send_rhs_bottom_L[k] = oldsol[r_idx_bottom*m+k]; 
                    }
                    MPI_Isend(send_rhs_bottom_L, grow, MPI_DOUBLE, p_idx_bottom_L, 0, comm, &req_bottom[9]);
                    MPI_Recv(receive_rhs_bottom_L, grow, MPI_DOUBLE, p_idx_bottom_L, 0, comm, &stat_bottom[18]);
                    MPI_Wait(&req_bottom[9], &stat_bottom[19]);
                    MPI_Wait(&req_bottom[8], &stat_bottom[20]);
                    // update the receive_rhs_bottom
                    for(int k=0; k<m+grow; k++){
                        if(k<grow){
                            receive_rhs_bottom[k] = receive_rhs_bottom_L[k];
                        }
                        else{
                            receive_rhs_bottom[k] = receive_rhs_bottom_C[k-grow];
                        }
                    }
                }
            } 
            // update the bottom boundary of the rhslocal 
            for(int k=0; k<rowset_exp_len; k++){
                rhslocal[(colset_exp_len-1)*(rowset_exp_len)+k] = rhslocal[(colset_exp_len-1)*(rowset_exp_len)+k] + receive_rhs_bottom[k];//oldsol[ind*(q*m)+(rowset_expanded[k]-1)];
            }
            
        }// updating BOTTOM boundary
        // *********************************************************************************************************

        // ***********************************LEFT and RIGHT boundary***********************************************
        // Add the remaining left and right communications
        if(j>0){// LEFT boundary
            p_idx_left_U = (i-1)*q+j-1;
            p_idx_left_C = (i)*q+j-1;
            p_idx_left_B = (i+1)*q+j-1;
            s_idx_left = m-grow-1;// -1 to adjust index the index to be sent
            
            // cater to the request from the left_C processor
            MPI_Isend(&s_idx_left, 1, MPI_INT, p_idx_left_C, 0, comm, &req_left[0]);// NOTE: the tag is important!
            MPI_Recv(&r_idx_left, 1, MPI_INT, p_idx_left_C, 0, comm, &stat_left[0]);
            for(int k=0; k<m; k++){
                send_rhs_left_C[k] = oldsol[k*m+r_idx_left];
            }
            MPI_Isend(send_rhs_left_C, m, MPI_DOUBLE, p_idx_left_C, 0, comm, &req_left[1]);

            // Send the index of row to the processor on left 
            // Receive the row from the idx processor
            MPI_Recv(receive_rhs_left_C, m, MPI_DOUBLE, p_idx_left_C, 0, comm, &stat_left[3]);
            MPI_Wait(&req_left[1], &stat_left[1]);// wait send self
            MPI_Wait(&req_left[0], &stat_left[2]);// wait send self
            if(colset_exp_len == m){
                // update the receive_rhs_left
                for(int k=0; k<m; k++){
                    receive_rhs_left[k] = receive_rhs_left_C[k];
                }
            }
            else if(colset_exp_len == m+2*grow){
                // cater to the request from left_U processor
                MPI_Isend(&s_idx_left, 1, MPI_INT, p_idx_left_U, 0, comm, &req_left[2]);// keep tag as 0
                MPI_Recv(&r_idx_left, 1, MPI_INT, p_idx_left_U, 0, comm, &stat_left[5]);
                for(int k=0;k<grow; k++){
                    send_rhs_left_U[k] = oldsol[k*m+r_idx_left]; 
                }
                MPI_Isend(send_rhs_left_U, grow, MPI_DOUBLE, p_idx_left_U, 0, comm, &req_left[3]);
                MPI_Recv(receive_rhs_left_U, grow, MPI_DOUBLE, p_idx_left_U, 0, comm, &stat_left[6]);
                MPI_Wait(&req_left[3], &stat_left[7]);
                MPI_Wait(&req_left[2], &stat_left[8]);

                // cater to the request from the left_B processor
                MPI_Isend(&s_idx_left, 1, MPI_INT, p_idx_left_B, 0, comm, &req_left[4]);
                MPI_Recv(&r_idx_left, 1, MPI_INT, p_idx_left_B, 0, comm, &stat_left[9]);
                for(int k=m-grow; k<m; k++){// the upper right corner part
                    send_rhs_left_B[k-(m-grow)] = oldsol[k*m+r_idx_left];
                }
                MPI_Isend(send_rhs_left_B, grow, MPI_DOUBLE, p_idx_left_B, 0, comm, &req_left[5]);
                MPI_Recv(receive_rhs_left_B, grow, MPI_DOUBLE, p_idx_left_B, 0, comm, &stat_left[10]);
                MPI_Wait(&req_left[5], &stat_left[11]);
                MPI_Wait(&req_left[4], &stat_left[12]);
                // update the receive_rhs_left
                for(int k=0; k<m+2*grow; k++){
                    if(k<grow){
                        receive_rhs_left[k] = receive_rhs_left_U[k];
                    }
                    else if(k>=grow && k<m+grow){
                        receive_rhs_left[k] = receive_rhs_left_C[k-grow];
                    }
                    else{
                        receive_rhs_left[k] = receive_rhs_left_B[k-(m+grow)];
                    }
                }
            
            }
            else if(colset_exp_len==m+grow){ // so either at right boundary
                if(i==0){// processor at upper boundary
                    // cater to the request of left_B processor
                    MPI_Isend(&s_idx_left, 1, MPI_INT, p_idx_left_B, 0, comm, &req_left[6]);
                    MPI_Recv(&r_idx_left, 1, MPI_INT, p_idx_left_B, 0, comm, &stat_left[13]);
                    for(int k=m-grow; k<m; k++){// the upper right corner part
                        send_rhs_left_B[k-(m-grow)] = oldsol[m*k+r_idx_left];
                    }
                    MPI_Isend(send_rhs_left_B, grow, MPI_DOUBLE, p_idx_left_B, 0, comm, &req_left[7]);
                    MPI_Recv(receive_rhs_left_B, grow, MPI_DOUBLE, p_idx_left_B, 0, comm, &stat_left[14]);
                    MPI_Wait(&req_left[7], &stat_left[15]);
                    MPI_Wait(&req_left[6], &stat_left[16]);
                    // update the receive_rhs_left
                    for(int k=0; k<m+grow; k++){
                        if(k<m){
                            receive_rhs_left[k] = receive_rhs_left_C[k];
                        }
                        else{
                            receive_rhs_left[k] = receive_rhs_left_B[k-m];
                        }
                    }
                    
                }
                else if(i==p-1){// processor at bottom boundary
                    // cater to the request of left_U processor
                    MPI_Isend(&s_idx_left, 1, MPI_INT, p_idx_left_U, 0, comm, &req_left[8]);// keep tag as 0
                    MPI_Recv(&r_idx_left, 1, MPI_INT, p_idx_left_U, 0, comm, &stat_left[17]);
                    for(int k=0;k<grow; k++){
                        send_rhs_left_U[k] = oldsol[m*k+r_idx_left]; 
                    }
                    MPI_Isend(send_rhs_left_U, grow, MPI_DOUBLE, p_idx_left_U, 0, comm, &req_left[9]);
                    MPI_Recv(receive_rhs_left_U, grow, MPI_DOUBLE, p_idx_left_U, 0, comm, &stat_left[18]);
                    MPI_Wait(&req_left[9], &stat_left[19]);
                    MPI_Wait(&req_left[8], &stat_left[20]);
                    // update the receive_rhs_left
                    for(int k=0; k<m+grow; k++){
                        if(k<grow){
                            receive_rhs_left[k] = receive_rhs_left_U[k];
                        }
                        else{
                            receive_rhs_left[k] = receive_rhs_left_C[k-grow];
                        }
                    }
                }
            } 
            // update the left boundary of the rhslocal 
            for(int k=0; k<colset_exp_len; k++){
                rhslocal[k*rowset_exp_len] = rhslocal[k*rowset_exp_len]+receive_rhs_left[k];//oldsol[(colset_expanded[k]-1)*(q*m) + ind];
            }
        }


        if(j<q-1){// RIGHT boundary
            p_idx_right_U = (i-1)*q+j+1;
            p_idx_right_C = (i)*q+j+1;
            p_idx_right_B = (i+1)*q+j+1;
            s_idx_right = grow;// -1 to adjust index the index to be sent
            
            // cater to the request from the right_C processor
            MPI_Isend(&s_idx_right, 1, MPI_INT, p_idx_right_C, 0, comm, &req_right[0]);// NOTE: the tag is important!
            MPI_Recv(&r_idx_right, 1, MPI_INT, p_idx_right_C, 0, comm, &stat_right[0]);
            for(int k=0; k<m; k++){
                send_rhs_right_C[k] = oldsol[k*m+r_idx_right];
            }
            MPI_Isend(send_rhs_right_C, m, MPI_DOUBLE, p_idx_right_C, 0, comm, &req_right[1]);

            // Send the index of row to the processor on right 
            // Receive the row from the idx processor
            MPI_Recv(receive_rhs_right_C, m, MPI_DOUBLE, p_idx_right_C, 0, comm, &stat_right[3]);
            MPI_Wait(&req_right[1], &stat_right[1]);// wait send self
            MPI_Wait(&req_right[0], &stat_right[2]);// wait send self
            if(colset_exp_len == m){
                // update the receive_rhs_right
                for(int k=0; k<m; k++){
                    receive_rhs_right[k] = receive_rhs_right_C[k];
                }
            }
            else if(colset_exp_len == m+2*grow){
                // cater to the request from right_U processor
                MPI_Isend(&s_idx_right, 1, MPI_INT, p_idx_right_U, 0, comm, &req_right[2]);// keep tag as 0
                MPI_Recv(&r_idx_right, 1, MPI_INT, p_idx_right_U, 0, comm, &stat_right[5]);
                for(int k=0;k<grow; k++){
                    send_rhs_right_U[k] = oldsol[k*m+r_idx_right]; 
                }
                MPI_Isend(send_rhs_right_U, grow, MPI_DOUBLE, p_idx_right_U, 0, comm, &req_right[3]);
                MPI_Recv(receive_rhs_right_U, grow, MPI_DOUBLE, p_idx_right_U, 0, comm, &stat_right[6]);
                MPI_Wait(&req_right[3], &stat_right[7]);
                MPI_Wait(&req_right[2], &stat_right[8]);

                // cater to the request from the right_B processor
                MPI_Isend(&s_idx_right, 1, MPI_INT, p_idx_right_B, 0, comm, &req_right[4]);
                MPI_Recv(&r_idx_right, 1, MPI_INT, p_idx_right_B, 0, comm, &stat_right[9]);
                for(int k=m-grow; k<m; k++){// the upper right corner part
                    send_rhs_right_B[k-(m-grow)] = oldsol[k*m+r_idx_right];
                }
                MPI_Isend(send_rhs_right_B, grow, MPI_DOUBLE, p_idx_right_B, 0, comm, &req_right[5]);
                MPI_Recv(receive_rhs_right_B, grow, MPI_DOUBLE, p_idx_right_B, 0, comm, &stat_right[10]);
                MPI_Wait(&req_right[5], &stat_right[11]);
                MPI_Wait(&req_right[4], &stat_right[12]);
                // update the receive_rhs_right
                for(int k=0; k<m+2*grow; k++){
                    if(k<grow){
                        receive_rhs_right[k] = receive_rhs_right_U[k];
                    }
                    else if(k>=grow && k<m+grow){
                        receive_rhs_right[k] = receive_rhs_right_C[k-grow];
                    }
                    else{
                        receive_rhs_right[k] = receive_rhs_right_B[k-(m+grow)];
                    }
                }
            
            }
            else if(colset_exp_len==m+grow){ // so either at right boundary
                if(i==0){// processor at upper boundary
                    // cater to the request of right_B processor
                    MPI_Isend(&s_idx_right, 1, MPI_INT, p_idx_right_B, 0, comm, &req_right[6]);
                    MPI_Recv(&r_idx_right, 1, MPI_INT, p_idx_right_B, 0, comm, &stat_right[13]);
                    for(int k=m-grow; k<m; k++){// the upper right corner part
                        send_rhs_right_B[k-(m-grow)] = oldsol[m*k+r_idx_right];
                    }
                    MPI_Isend(send_rhs_right_B, grow, MPI_DOUBLE, p_idx_right_B, 0, comm, &req_right[7]);
                    MPI_Recv(receive_rhs_right_B, grow, MPI_DOUBLE, p_idx_right_B, 0, comm, &stat_right[14]);
                    MPI_Wait(&req_right[7], &stat_right[15]);
                    MPI_Wait(&req_right[6], &stat_right[16]);
                    // update the receive_rhs_right
                    for(int k=0; k<m+grow; k++){
                        if(k<m){
                            receive_rhs_right[k] = receive_rhs_right_C[k];
                        }
                        else{
                            receive_rhs_right[k] = receive_rhs_right_B[k-m];
                        }
                    }
                    
                }
                else if(i==p-1){// processor at bottom boundary
                    // cater to the request of right_U processor
                    MPI_Isend(&s_idx_right, 1, MPI_INT, p_idx_right_U, 0, comm, &req_right[8]);// keep tag as 0
                    MPI_Recv(&r_idx_right, 1, MPI_INT, p_idx_right_U, 0, comm, &stat_right[17]);
                    for(int k=0;k<grow; k++){
                        send_rhs_right_U[k] = oldsol[m*k+r_idx_right]; 
                    }
                    MPI_Isend(send_rhs_right_U, grow, MPI_DOUBLE, p_idx_right_U, 0, comm, &req_right[9]);
                    MPI_Recv(receive_rhs_right_U, grow, MPI_DOUBLE, p_idx_right_U, 0, comm, &stat_right[18]);
                    MPI_Wait(&req_right[9], &stat_right[19]);
                    MPI_Wait(&req_right[8], &stat_right[20]);
                    // update the receive_rhs_right
                    for(int k=0; k<m+grow; k++){
                        if(k<grow){
                            receive_rhs_right[k] = receive_rhs_right_U[k];
                        }
                        else{
                            receive_rhs_right[k] = receive_rhs_right_C[k-grow];
                        }
                    }
                }
            } 

            for(int k=0; k<colset_exp_len; k++){
                rhslocal[k*rowset_exp_len+(rowset_exp_len-1)] = rhslocal[k*rowset_exp_len+(rowset_exp_len-1)] + receive_rhs_right[k];//oldsol[(colset_expanded[k]-1)*(q*m)+ind];
            }
        }

        nRows = colset_exp_len ;
        nCols = rowset_exp_len;
        if(sweep ==0 ){
            NNONZEROS = csr_laplacian2d(nRows, nCols, rowIndex, columns, values, e, b1, b2);
        }
        // NOTE: important to convert to column major format!
        convert_column_major(rhslocal, rhslocal_c, colset_exp_len, rowset_exp_len);
        if(sweep==0){
            set_solver(&handle, &opt, &type, &sym, nRows*nCols, nRows*nCols, NNONZEROS, nRhs, rowIndex, columns, values);
        }
        solver(handle, opt, nRows*nCols, nCols*nRows, NNONZEROS, nRhs, rowIndex, columns, rhslocal_c, temp);
        starti = colset[0]-colset_expanded[0]; 
        startj = rowset[0]-rowset_expanded[0];
        // updating the new solution (only the m by m part)
        for(int c=0; c<m; c++){
            for(int r=0; r<m; r++){
                newsol[c*m+r] = temp[(starti+c)*rowset_exp_len+(startj+r)];
            }
        }
        // Calculating the error for each processor's newsol
        double error = 0;
        for(int c=0; c<m; c++){
            for(int r=0; r<m; r++){
                error += newsol[c*m+r] * newsol[c*m+r];
            }
        }
        double total_error; 
        // Collect the error calculations and find total error, then broadcast to all
        MPI_Allreduce(&error, &total_error, 1, MPI_DOUBLE, MPI_SUM, comm);
        if(rank == 0){
            printf("sweep = %d error norm = %lf\n", sweep, sqrt(total_error));
        }
/*
        if(sqrt(total_error) < 0.001){
            printf("sweep = %d total error = %lf which is less than 10^-3, breaking out of loop...\n", sweep, total_error);
            break;
        }
*/
        copy_matrix(newsol, oldsol, m, m);        
    } // sweep
    if(rank==0){
        mytime = MPI_Wtime() - mytime;
        printf("Total time = %lf and time per iteration = %lf\n", mytime, mytime/(double)(sweep+1));
    }
    /* -------------------------- */
    /* Deallocate solver storage  */
    /* -------------------------- */
    _INTEGER_t error;
    error = dss_delete (handle, opt);

    MPI_Finalize();

    free(oldsol);
    free(newsol);
    free(rhslocal);
    free(rhslocal_c);
    free(colset);
    free(colset_expanded);
    free(rowset);
    free(rowset_expanded);
    free(receive_rhs_bottom);
    free(receive_rhs_bottom_L);
    free(receive_rhs_bottom_C);
    free(receive_rhs_bottom_R);
    free(receive_rhs_top);
    free(receive_rhs_top_L);
    free(receive_rhs_top_C);
    free(receive_rhs_top_R);
    free(receive_rhs_left);
    free(receive_rhs_left_U);
    free(receive_rhs_left_C);
    free(receive_rhs_left_B);
    free(receive_rhs_right);
    free(receive_rhs_right_U);
    free(receive_rhs_right_C);
    free(receive_rhs_right_B);
    free(send_rhs_bottom_L);
    free(send_rhs_bottom_C);
    free(send_rhs_bottom_R);
    free(send_rhs_top_L);
    free(send_rhs_top_C);
    free(send_rhs_top_R);
    free(send_rhs_left_U);
    free(send_rhs_left_C);
    free(send_rhs_left_B);
    free(send_rhs_right_U);
    free(send_rhs_right_C);
    free(send_rhs_right_B);
    free(e);
    free(b1);
    free(b2);
    return 0;
}

