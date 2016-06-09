aprun -n1 -b nvprof -o prof334.time ./gpu3d
aprun -n1 -b nvprof --analysis-metrics -o prof334.met ./gpu3d
cp prof* /ccs/home/hynninen/gpu_transpose/
