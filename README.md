GPU tensor transpose
(c) Antti-Pekka Hynninen, 2016

How to compile:
----------------
make

Profiling on Titan:
--------------------
aprun -n1 -b nvprof -o prof334.time ./gpu3d

aprun -n1 -b nvprof --analysis-metrics -o prof334.met ./gpu3d

cp prof* /ccs/home/hynninen/gpu_transpose/

