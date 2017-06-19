NUM_PROCESS = 60 

all: jacobi_schwarz jacobi_schwarz_mic

serial:
	gcc -std=c99 serial.c -o serial
	valgrind --leak-check=yes ./serial

host:
	mpiicc -std=c99 parallel.c -mkl -o jacobi_schwarz

runhost:
	I_MPI_DEBUG=4 mpirun -n $(NUM_PROCESS) ./jacobi_schwarz

mic:
	mpiicc -mmic -std=c99 -O3 parallel.c -mkl -o jacobi_schwarz_mic
	scp jacobi_schwarz_mic mic2:

runmic:
	I_MPI_MIC=1 mpirun -host mic2 -n $(NUM_PROCESS) ~/jacobi_schwarz_mic

jacobi_schwarz: jacobi_schwarz.c
	mpiicc -std=c99 jacobi_schwarz.c -mkl -o jacobi_schwarz

jacobi_schwarz_mic: jacobi_schwarz.c
	mpiicc -mmic -std=c99 -O3 jacobi_schwarz.c -mkl -o jacobi_schwarz_mic
	scp jacobi_schwarz_mic mic2:

clean:
	rm -rf jacobi_schwarz jacobi_schwarz_mic jacobi_schwarz.o
