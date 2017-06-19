bd_omp: bd_omp.c
	icc -std=c99 -mkl -restrict -qopenmp -O3 harness.c bd_omp.c timer.c -o bd_mic -mmic -qopt-report=5 -qopt-report-phase=vec #-no-vec

bd_cilk: bd_cilk.c
	icc -std=c99 -mkl -restrict -O3 harness.c bd_cilk.c timer.c -o bd_mic -mmic -qopt-report=5 -qopt-report-phase=vec #-no-vec

bd_serial: bd_serial.c
	icc -std=c99 -mkl -restrict -O3 harness.c bd_cilk.c timer.c -o bd_mic -mmic -qopt-report=5 -qopt-report-phase=vec #-no-vec

bd_mic: bd_all.c
	icc -std=c99 -mkl -restrict -qopenmp -O3 harness.c bd_all.c timer.c -o bd_mic -mmic -qopt-report=5 -qopt-report-phase=vec #-no-vec

bd_host: bd_all.c
	icc -std=c99 -mkl -restrict -qopenmp -O3 harness.c bd_all.c timer.c -o bd_host -qopt-report=5 -qopt-report-phase=vec

runmic:
	ssh mic2 rm -f output.xyz
	scp lac1_novl2.xyz mic2:
	scp bd_mic mic2:
#	ssh mic2 KMP_AFFINITY=verbose,granularity=fine,balanced ./bd_mic lac1_novl2.xyz output.xyz 10
	ssh mic2 ./bd_mic lac1_novl2.xyz output.xyz 1
	scp Ahshrivastava3@joker-mic2:/home/Ahshrivastava3/output.xyz .

runcilk:
	ssh mic2 rm -f output.xyz
	scp lac1_novl2.xyz mic2:
	scp bd_mic mic2:
#	ssh mic2 KMP_AFFINITY=verbose,granularity=fine,balanced ./bd_mic lac1_novl2.xyz output.xyz 10
	ssh mic2 ./bd_mic lac1_novl2.xyz output.xyz 1
	scp Ahshrivastava3@joker-mic2:/home/Ahshrivastava3/output.xyz .

runhost:
	rm -f output.xyz
	./bd_host lac1_novl2.xyz output.xyz 10

create:
	icc -std=c99 -mkl create_input.c -lm

rungcc_my_desktop:
	gcc -std=c99 -g -fopenmp harness.c bd.c timer.c -lm
	./a.out lac1_novl2.xyz output.xyz 10
	scp Ahshrivastava3@joker.cc.gatech.edu:/home/Ahshrivastava3/project1/cse6230/proj1/output.xyz .

clean:
	rm -f *.o bd bd_mic output.xyz bd_host *.optrpt a.out
