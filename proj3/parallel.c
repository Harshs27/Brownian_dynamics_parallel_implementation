#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mkl.h>
#include <mpi.h>
#include "mkl_dss.h"
#include "mkl_types.h"
#include <math.h>

#define M_DOMAIN 100
#define P_MESH 10 
#define Q_MESH 6 
#define GROW_DOMAIN 10 
#define SWEEP 500

MKL_INT solver(_MKL_DSS_HANDLE_t handle, MKL_INT opt, MKL_INT sym, MKL_INT type, MKL_INT nRows,  MKL_INT nCols,  MKL_INT nNonZeros,  MKL_INT nRhs,  _INTEGER_t *rowIndex,  _INTEGER_t* columns, double* values,  _DOUBLE_PRECISION_t* rhs, _DOUBLE_PRECISION_t* solValues){

        MKL_INT opt1;
  MKL_INT i, j;
  _INTEGER_t error;
  _CHARACTER_t statIn[] = "determinant", *uplo;
  _DOUBLE_PRECISION_t statOut[5], eps = 1e-6;
/* ------------------------ */
/* Get the solution vector for Ax=b and ATx=b and check correctness */
/* ------------------------ */
  for (i = 0; i < 1; i++)
    {
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
      else
// Conjugate transposed == transposed for real matrices
      if (i == 2)
        {
          uplo = "conjugate transposed";
          opt1 = MKL_DSS_CONJUGATE_SOLVE;
        }

//      printf ("\nSolving %s system...\n", uplo);

// Compute rhs respectively to uplo to have solution solValue
//      mkl_dcsrgemv (uplo, &nRows, values, rowIndex, columns, solValues, rhs);

// Nullify solution on entry (for sure)
      for (j = 0; j < nCols; j++){
        solValues[j] = 0.0;
//        printf("print %lf\t", rhs[j]);
    }
//    printf("\n");

// Apply trans or non-trans option, solve system
      opt |= opt1;
      error = dss_solve_real (handle, opt, rhs, nRhs, solValues);
      if (error != MKL_DSS_SUCCESS)
        goto printError;
      opt &= ~opt1;

// Check solution vector: should be {0,1,2,3,4}
/*
      for (j = 0; j < nCols; j++)
        {
          if ((solValues[j] > j + eps) || (solValues[j] < j - eps))
            {
              printf ("Incorrect solution\n");
              error = 1000 + i;
              goto printError;
            }
        }
*/
/*
      printf ("Print solution array: ");
      for (j = 0; j < nCols; j++)
        printf (" %g", solValues[j]);

      printf ("\n");
*/
    }

/* -------------------------- */
/* Deallocate solver storage  */
/* -------------------------- */
  error = dss_delete (handle, opt);
  if (error != MKL_DSS_SUCCESS)
    goto printError;
/* ---------------------- */
/* Print solution vector  */
/* ---------------------- */
//  printf ("\nExample successfully PASSED!\n");
//  exit (0);
    return 0;
printError:
  printf ("Solver returned error code %d\n", error);
  exit (1);
}



void set_solver(_MKL_DSS_HANDLE_t* handle, MKL_INT* opt, MKL_INT* type, MKL_INT* sym, MKL_INT nRows,  MKL_INT nCols,  MKL_INT nNonZeros,  MKL_INT nRhs,  MKL_INT *rowIndex,  MKL_INT * columns, double* values,  double *rhs, double* solValues){
        /* Allocate storage for the solver handle and the right-hand side. */
        _INTEGER_t error;
        /* --------------------- */
        /* Initialize the solver */
        /* --------------------- */
//        printf("entered the solver\n");
        error = dss_create (*handle, *opt);
        if (error != MKL_DSS_SUCCESS)
        goto printError;

        /* ------------------------------------------- */
        /* Define the non-zero structure of the matrix */
        /* ------------------------------------------- */
//        printf("pass 0\n");
//        printf("sym = %d nRows=%d and nCols = %d and nNonZeros = %d\n", *sym, nRows, nCols, nNonZeros);
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
//        printf("pass 1\n");

//        solver(handle, opt, type, sym, nRows, nCols, nNonZeros, nRhs, rowIndex, columns, values, rhs, solValues);
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
/*
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
    int num_elements_row;
    int nCols = n;
    int nRows = m;
    int MEM_SIZE = (nCols*nRows)*5;
    int *columns = (int *)malloc(MEM_SIZE*sizeof(int));
    double *values = (double *)malloc(MEM_SIZE*sizeof(double));
    int *rowIndex = (int *)malloc((nRows*nCols+1)*sizeof(int));
    int ri = 1, ci = 0, vi = 0;
    rowIndex[0] = 1;
    for(int x=0; x<m*n; x++){// all rows
        num_elements_row = 0;
        for(int y=0; y<m*n; y++){// all cols
            a[x*(m*n)+y] = 0; //NOTE: insert code for MKL sparse format! 
            if(x==y){// diagonal element -4*e
                a[x*(m*n)+y] = -4*e[y];
                printf("%d  ", a[x*(m*n)+y]);
                values[vi++] = -4*e[y];
                columns[ci++] = y;
                num_elements_row++;
            }
            else if (y == x+1){// b2 term
                a[x*(m*n)+y] = b2[y];
                printf("%d  ", a[x*(m*n)+y]);
                values[vi++] = b2[y];
                columns[ci++] = y;
                num_elements_row++;
            }
            else if (y == x+m){
                a[x*(m*n)+y] = -1;
                printf("%d  ", a[x*(m*n)+y]);
                values[vi++] = -1;
                columns[ci++] = y;
                num_elements_row++;
            }
            else if (y == x-1){// b1 term
                a[x*(m*n)+y] = b1[y];
                printf("%d  ", a[x*(m*n)+y]);
                values[vi++] = b1[y];
                columns[ci++] = y;
                num_elements_row++;
            }
            else if (y == x-m){
                a[x*(m*n)+y] = -1;
                printf("%d  ", a[x*(m*n)+y]);
                values[vi++] = -1;
                columns[ci++] = y;
                num_elements_row++;
            }
        }
        rowIndex[ri] = rowIndex[ri-1]+num_elements_row;
        ri++;
//        printf(" \t \t row num = %d and num_elements = %d and %d and ri = %d\n", x, num_elements_row, rowIndex[ri-1], ri);
        printf("\nrow num = %d and num_elements = %d and %d and ri = %d\n", x, num_elements_row, rowIndex[ri-1], ri);
//    printf("\n");
//    print_array(rowIndex, nCols*nRows+1);
    }
    int NNONZEROS = rowIndex[nCols*nRows]-1;
    printf("non zeros = %d\n", NNONZEROS);
    printf("\ncolumns_array\n");
    print_array(columns, NNONZEROS);
    printf("\nvalues_array\n");
    print_array_d(values, NNONZEROS);
    printf("\n row_array1\n");
    print_array(rowIndex, nCols*nRows+1);
    free(e);
    free(b1);
    free(b2);
    return;
}
*/
//MKL_INT csr_laplacian2d(MKL_INT nRows, MKL_INT nCols, MKL_INT *rowIndex, MKL_INT *columns, double* values, int *e, int *b1, int *b2){// Returns number of nonzeros
int csr_laplacian2d(int nRows, int nCols, int *rowIndex, int *columns, double* values, int *e, int *b1, int *b2){// Returns number of nonzeros
/*    int MEM_SIZE = (nCols*nRows)*5;
    int *columns = (int *)malloc(MEM_SIZE*sizeof(int));
    double *values = (double *)malloc(MEM_SIZE*sizeof(double));
    int *rowIndex = (int *)malloc((nCols*nRows+1)*sizeof(int));
*/
    int ri = 1, ci = 0, vi = 0, num_elements_row;
    int m, n;    
/*    if(nRows > nCols){
        m = nRows;
        n = nCols; 
    }
    else{
        m = nCols; n = nRows;
    }
*/
    m = nRows, n = nCols;

    for(int i=0; i<m*n; i++){
        e[i] = -1;
        b1[i] = -1;
        b2[i] = -1;
    }
    for(int i=m-1; i<n*m; i=i+m){
        b1[i] = 0;
        b2[i-m+1] = 0;
    }
    // NOTE: [-m, -1, 0, 1, m] has to be imitated here
    rowIndex[0] = 1;
    for(int x=0; x<m*n; x++){// all rows
        num_elements_row = 0;
        for(int y=0; y<m*n; y++){// all cols
            if(x==y){// diagonal element -4*e
//                printf("%d  ", -4*e[y]);
                values[vi++] = -4*e[y];
                columns[ci++] = y+1;// adjusting 1 for the solver function
                num_elements_row++;
            }
            else if (y == x+1){// b2 term
//                printf("%d  ", b2[y]);
                values[vi++] = b2[y];
                columns[ci++] = y+1;
                num_elements_row++;
            }
            else if (y == x+m){
//                printf("%d  ", -1);
                values[vi++] = -1;
                columns[ci++] = y+1;
                num_elements_row++;
            }
            else if (y == x-1){// b1 term
//                printf("%d  ", b1[y]);
                values[vi++] = b1[y];
                columns[ci++] = y+1;
                num_elements_row++;
            }
            else if (y == x-m){
//                printf("%d  ", -1);
                values[vi++] = -1;
                columns[ci++] = y+1;
                num_elements_row++;
            }
        }
        rowIndex[ri] = rowIndex[ri-1]+num_elements_row;
        ri++;
    }
    int NNONZEROS = rowIndex[nCols*nRows]-1;
/*
    printf("non zeros = %d\n", NNONZEROS);
    printf("\ncolumns_array\n");
    print_array(columns, NNONZEROS);
    printf("\nvalues_array\n");
    print_array_d(values, NNONZEROS);
    printf("\n row_array1\n");
    print_array(rowIndex, nCols*nRows+1);
*/
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
//    double * send_rhs_left     = (double *)malloc((m+2*grow)*sizeof(double));
    double * send_rhs_left_B     = (double *)malloc((grow)*sizeof(double));

    double * send_rhs_right_U    = (double *)malloc((grow)*sizeof(double));
    double * send_rhs_right_C    = (double *)malloc((m)*sizeof(double));
//    double * send_rhs_right    = (double *)malloc((m+2*grow)*sizeof(double));
    double * send_rhs_right_B    = (double *)malloc((grow)*sizeof(double));
//    rhs_init = ;// INITIALISATION of RHS (NOTE: Ask how to distribute the initial rhs global matrix among processors)
    rand_init(newsol, oldsol, m, m);// and old and new sol to some random values
/*    
    for (int i1=0; i1<m; i1++){
        for(int j1=0;j1<m; j1++){
            oldsol[i1*m+j1] = rank+10;
            newsol[i1*m+j1] = oldsol[i1*m+j1];
        }
    }     
*/
        int nRows = (m+2*grow) ;
        int nCols = (m+2*grow) ;
        int MEM_SIZE = (nCols*nRows)*5;
//        printf("mkl int = %d and int = %d\n",sizeof(MKL_INT), sizeof(int));
        int *columns = (int *)malloc(MEM_SIZE*sizeof(int));
        assert(columns);
        double *values = (double *)malloc(MEM_SIZE*sizeof(double));
        assert(values);
        int *rowIndex = (int *)malloc((nCols*nRows+1)*sizeof(int));
        assert(rowIndex);
        int NNONZEROS, nRhs = 1;//NRHS;
    int *e = (int *)malloc(nRows*nCols*sizeof(int));
    int *b1 = (int *)malloc(nRows*nCols*sizeof(int));
    int *b2 = (int *)malloc(nRows*nCols*sizeof(int));
        double * temp          = (double*)malloc((m+2*grow)*(m+2*grow)*sizeof(double));
        int starti, startj;


    MPI_Request req_top[10], req_bottom[10], req_left[10], req_right[10];
    MPI_Status stat_top[21], stat_bottom[21], stat_left[21], stat_right[21];

    int p_idx_top, r_idx_top, s_idx_top, p_idx_bottom, r_idx_bottom, s_idx_bottom, p_idx_right, r_idx_right, s_idx_right, p_idx_left, r_idx_left, s_idx_left;
   
    int i, j;
    int flag_colset_exp, flag_rowset_exp, colset_exp_len, rowset_exp_len;
    
    for (int sweep=0; sweep<SWEEP; sweep++){
        MPI_Barrier(comm);
        // Get the i and j for each processor (i.e. the position in the grid)
        j = rank%q;
        i = (rank-rank%q)/q;
//        printf("rank = %d, i=%d, j=%d\n", rank, i, j);
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

        // RHS LOCAL PART
        // NOTE: As the RHS local part 
//        printf("i=%d j=%d colset_len=%d rowset_len =%d\n", i, j, colset_exp_len, rowset_exp_len);
        for(int c=0; c<colset_exp_len; c++){
//            printf("c=%d\t", colset_expanded[c]);
            for(int r=0; r<rowset_exp_len; r++){
//                printf("r = %d\t", rowset_expanded[r]);
//                rhslocal[c*rowset_exp_len+r] = rhs[(colset_expanded[c]-1)*(q*m)+(rowset_expanded[r]-1)];
//                  rhslocal[c*rowset_exp_len+r] = rank+1;//0; NOTE: CHANGE this .. currently for debugging !!!!!!!!!!!!!!!!!
                  rhslocal[c*rowset_exp_len+r] = 0;// NOTE: CHANGE this .. currently for debugging !!!!!!!!!!!!!!!!!
            }
//            printf("\n");
        }
//        printf("initialised rhs local \n");
//        print_matrix_d(rhslocal, colset_exp_len, rowset_exp_len);

        // Modify the rhslocal with the boundary conditions from the neighbours
        // ******************** New function for grown top and bottom boundary value updation ******************************
        int p_idx_top_L, p_idx_top_C, p_idx_top_R;
        int p_idx_bottom_L, p_idx_bottom_C, p_idx_bottom_R;
        int p_idx_right_U, p_idx_right_C, p_idx_right_B;
        int p_idx_left_U, p_idx_left_C, p_idx_left_B;
        
       

        if(i>0){ // TOP boundary (communicate with the top processor)
            p_idx_top_L = (i-1)*q+j-1;
            p_idx_top_C = (i-1)*q+j;
            p_idx_top_R = (i-1)*q+j+1;
            s_idx_top = m-grow-1;// -1 to adjust index the index to be sent
            
            // cater to the request from the top_C processor
            MPI_Isend(&s_idx_top, 1, MPI_INT, p_idx_top_C, 0, comm, &req_top[0]);// NOTE: the tag is important!
            MPI_Recv(&r_idx_top, 1, MPI_INT, p_idx_top_C, 0, comm, &stat_top[0]);
//          printf("Top boundary i=%d, j=%d and receiving the idx %d\n", i, j, r_idx_top);
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
//                        printf("i==%d and j=%d , k=%d and send_rhs_top_L = %lf\n", i,j,k,send_rhs_top_L[k]);
                    }
//                    printf("\np_idx_top_L = %d\n", p_idx_top_L);
                    MPI_Isend(send_rhs_top_L, grow, MPI_DOUBLE, p_idx_top_L, 0, comm, &req_top[9]);
                    MPI_Recv(receive_rhs_top_L, grow, MPI_DOUBLE, p_idx_top_L, 0, comm, &stat_top[18]);
                    MPI_Wait(&req_top[9], &stat_top[19]);
                    MPI_Wait(&req_top[8], &stat_top[20]);
//                    printf("i=%d j=%d receive_rhs_top_L values = %lf, %lf\n",i,j, receive_rhs_top_L[0], receive_rhs_top_L[1]);
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
//                        printf("i=%d j=%d send_rhs_bottom_R[%d] = %lf\n", i, j,k, send_rhs_bottom_R[k-(m-grow)]);
                    }
                    MPI_Isend(send_rhs_bottom_R, grow, MPI_DOUBLE, p_idx_bottom_R, 0, comm, &req_bottom[7]);
                    MPI_Recv(receive_rhs_bottom_R, grow, MPI_DOUBLE, p_idx_bottom_R, 0, comm, &stat_bottom[14]);
//                    printf("\np_idx_bottom_R = %d and received_rhs_bottom_R_val=%lf and grow = %d\n", p_idx_bottom_R, receive_rhs_bottom_R[0], grow);
                    MPI_Wait(&req_bottom[7], &stat_bottom[15]);
                    MPI_Wait(&req_bottom[6], &stat_bottom[16]);
                    // update the receive_rhs_bottom
                    for(int k=0; k<m+grow; k++){
                        if(k<m){
                            receive_rhs_bottom[k] = receive_rhs_bottom_C[k];
                        }
                        else{
//                            printf("i-%d and j=%d , k=%d and receive_rhs_bottom_R = %lf\n", i,j,k,receive_rhs_bottom_R[k-m]);
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
//            printf("Right boundary i=%d, j=%d\n", i, j);
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
//            print_matrix_d(rhslocal, colset_exp_len, rowset_exp_len);
        }
        // *********************************************************************************************************
/*
        if(rank==0){
            printf("rank = %d\n", rank);
            print_matrix_d(oldsol, m, m);
            print_matrix_d(rhslocal, colset_exp_len, rowset_exp_len);
            
        }
*/
        // ADD the local calculation part here!
// ****************************************EDIT THIS PART*****************************************************                
        // SOLVE the local problem : USE MKL???
//        int * alocal           = (int *)malloc((m+2*grow)*(m+2*grow)*(m+2*grow)*(m+2*grow)*sizeof(int));
/*
        MKL_INT nRows = colset_exp_len * rowset_exp_len;
        MKL_INT nCols = colset_exp_len * rowset_exp_len;
        MKL_INT MEM_SIZE = (nCols*nRows)*5;
        printf("mkl int = %d and int = %d\n",sizeof(MKL_INT), sizeof(int));
        MKL_INT *columns = (MKL_INT *)malloc(MEM_SIZE*sizeof(MKL_INT));
        assert(columns);
        double *values = (double *)malloc(MEM_SIZE*sizeof(double));
        assert(values);
        MKL_INT *rowIndex = (MKL_INT *)malloc((nCols*nRows+1)*sizeof(MKL_INT));
        assert(rowIndex);
        MKL_INT NNONZEROS, nRhs = 1;//NRHS;
*/

        nRows = colset_exp_len ;
        nCols = rowset_exp_len;
//        printf("nRows = %d and nCols=%d\n", nRows, nCols);
//        MEM_SIZE = (nCols*nRows)*5;
//        printf("mkl int = %d and int = %d\n",sizeof(MKL_INT), sizeof(int));
//        int *columns = (int *)malloc(MEM_SIZE*sizeof(int));
//        assert(columns);
//        double *values = (double *)malloc(MEM_SIZE*sizeof(double));
//        assert(values);
//        int *rowIndex = (int *)malloc((nCols*nRows+1)*sizeof(int));
//        assert(rowIndex);
//        int NNONZEROS, nRhs = 1;//NRHS;

// CHECK laplacian for CSR
/*        laplacian2d(alocal, colset_exp_len, rowset_exp_len);
        printf("colset len = %d and rowset len = %d\n", colset_exp_len, rowset_exp_len);
        print_matrix(alocal, colset_exp_len*rowset_exp_len, colset_exp_len*rowset_exp_len);

*/
        // RESET THE rowIndex, columns and values ???
        NNONZEROS = csr_laplacian2d(nRows, nCols, rowIndex, columns, values, e, b1, b2);
//        NNONZEROS = csr_laplacian2d(nCols, nRows, rowIndex, columns, values, e, b1, b2);
//        printf("non = %d\n", NNONZEROS);
        

//        printf("non = %d\n", NNONZEROS);
//        printf("non zeros = %d\n", NNONZEROS);
//        printf("\ncolumns_array\n");
//        print_array(columns, NNONZEROS);
//        printf("\nvalues_array\n");
//        print_array_d(values, NNONZEROS);
//        printf("\n row_array1\n");
///        print_array(rowIndex, nCols+1);

//       return 10; 

//        static _DOUBLE_PRECISION_t solValues[NROWS] = { 0, 1, 2, 3, 4 };

        _MKL_DSS_HANDLE_t handle;
        MKL_INT opt = MKL_DSS_DEFAULTS;
        MKL_INT sym = MKL_DSS_NON_SYMMETRIC;
//        printf("sym = %d\n", sym);
        MKL_INT type = MKL_DSS_INDEFINITE;
        // NOTE: The alocal matrix should be represented in Sparse format!
        // SOLVE locally and store solution in x = alocal \ rhslocal(:);  *************************IMP_currently incomplete**************
        // SAY that we have a solution and reshaped to colset_len and rowset_len
//        printf("rhslocal before\n");
//            print_matrix_d(rhslocal, colset_exp_len, rowset_exp_len);
        //NOTE: rhs local in column major format
        convert_column_major(rhslocal, rhslocal_c, colset_exp_len, rowset_exp_len);
        set_solver(&handle, &opt, &type, &sym, nRows*nCols, nRows*nCols, NNONZEROS, nRhs, rowIndex, columns, values, rhslocal_c, temp);
//        printf("Calling the solver_function\n");
        solver(handle, opt, type, sym, nRows*nCols, nCols*nRows, NNONZEROS, nRhs, rowIndex, columns, values, rhslocal_c, temp);
        

        // temp stores the solution values
//        printf("rhslocal\n");
//            print_matrix_d(rhslocal, colset_exp_len, rowset_exp_len);
//            print_matrix_d(temp, colset_exp_len, rowset_exp_len);
        

//        rand_init_single(temp, colset_exp_len, rowset_exp_len); // matrix

        starti = colset[0]-colset_expanded[0]; 
        startj = rowset[0]-rowset_expanded[0];
//        printf("starti = %d and startj= %d\n ", starti, startj);
        // updating the new solution (only the m by m part)
//        print_matrix_d(newsol, m, m);
//        print_matrix_d(temp, colset_exp_len, rowset_exp_len);
        for(int c=0; c<m; c++){
 //           printf("c=%d\t", colset[c]);
            for(int r=0; r<m; r++){
 //               printf("r = %d\t", rowset[r]);
//                newsol[(colset[c]-1)*(q*m)+(rowset[r]-1)] = temp[(starti+c)*rowset_exp_len+(startj+r)];
                newsol[c*m+r] = temp[(starti+c)*rowset_exp_len+(startj+r)];
            }
//            printf("\n");
        }
//        print_matrix_d(newsol, m, m);
        // Calculating the error for each processor's newsol
        double error = 0;
        for(int c=0; c<m; c++){
            for(int r=0; r<m; r++){
                error += newsol[c*m+r] * newsol[c*m+r];
            }
        }
        double total_error; 
//        printf("check seg and error =%lf \n", error);
        // Collect the error calculations and find total error, then broadcast to all
        MPI_Allreduce(&error, &total_error, 1, MPI_DOUBLE, MPI_SUM, comm);
        
//        printf("check seg 1 and total_error = %lf \n", total_error);
        if(rank == 0){
            printf("total error = %lf\n", total_error);
        }
//        print_matrix_d(newsol, m, m);

//        print_matrix_d(oldsol, m, m);
        if(sqrt(total_error) < 0.001){
            printf("sweep = %d total error = %lf which is less than 10^-3, breaking out of loop...\n", sweep, total_error);
            break;
        }

        copy_matrix(newsol, oldsol, m, m);
 //       print_matrix_d(oldsol, m, m);
// ****************************************END: EDIT THIS PART*****************************************************                


    } // sweep
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

