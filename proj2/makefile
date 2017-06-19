NOVEC_FLAGS=-qno-openmp-simd -no-vec -no-simd -O3
VEC_FLAGS=-qopenmp -O3 

all: check

check: hd run

check_novec: hd_novec run

check_mic: hd_mic runmic

check_mic_novec: hd_mic_novec runmic

check_only_openmp: only_openmp_mic runmic

check_only_vec: only_vec_mic runmic

clean:
	rm -f *.o hd hd_mic output.xyz

hd: rpy_ewald_polyd.c harness.c timer.c
	icc -std=c99 -mkl $(VEC_FLAGS) harness.c timer.c rpy_ewald_polyd.c -o hd -qopt-report=5 -qopt-report-phase=vec

hd_novec: rpy_ewald_polyd.c harness.c timer.c
	icc -std=c99 -mkl $(NOVEC_FLAGS) harness.c timer.c rpy_ewald_polyd.c -o hd -qopt-report=5 -qopt-report-phase=vec

hd_mic: rpy_ewald_polyd.c harness.c
	icc -std=c99 -mkl $(VEC_FLAGS) harness.c timer.c  rpy_ewald_polyd.c -o hd_mic -mmic -qopt-report=5 -qopt-report-phase=vec

hd_mic_novec: rpy_ewald_polyd.c harness.c
	icc -std=c99 -mkl $(NOVEC_FLAGS) harness.c timer.c  rpy_ewald_polyd.c -o hd_mic -mmic -qopt-report=5 -qopt-report-phase=vec

hd_cilk: cilk_rpy_ewald_polyd.c harness.c
	icc -std=c99 -mkl harness.c timer.c  cilk_rpy_ewald_polyd.c -o hd_mic -mmic -qopt-report=5 -qopt-report-phase=vec

hd_cilk_novec: cilk_rpy_ewald_polyd.c harness.c
	icc -std=c99 -mkl -no-vec -no-simd -O3 harness.c timer.c  cilk_rpy_ewald_polyd.c -o hd_mic -mmic -qopt-report=5 -qopt-report-phase=vec

run:
	rm -f output.dat
	./hd lac1_novl2.xyz output.dat gold.dat

runcilk:
	ssh mic2 rm -f output.dat
	scp hd_mic lac1_novl2.xyz gold.dat mic2:
	ssh mic2 CILK_NWORKERS=240 ./hd_mic lac1_novl2.xyz output.dat gold.dat

runmic:
	ssh mic2 rm -f output.dat
	scp hd_mic lac1_novl2.xyz gold.dat mic2:
	ssh mic2 KMP_AFFINITY=scatter time ./hd_mic lac1_novl2.xyz output.dat gold.dat

only_openmp_mic: rpy_ewald_polyd.c harness.c timer.c
	icc -std=c99 -mkl -qopenmp -O3 -no-vec -no-simd harness.c timer.c rpy_ewald_polyd.c -o hd_mic -mmic -qopt-report=5 -qopt-report-phase=vec

only_vec_mic: rpy_ewald_polyd.c harness.c timer.c
	icc -std=c99 -mkl -qno-openmp-simd -O3 harness.c timer.c rpy_ewald_polyd.c -o hd_mic -mmic -qopt-report=5 -qopt-report-phase=vec

.PHONY: all check check_novec check_mic check_mic_novec clean hd hd_novec hd_mic hd_mic_novec run runmic
