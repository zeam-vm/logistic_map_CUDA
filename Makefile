all: a.out

a.out: logistic_map.cu
	nvcc logistic_map.cu
	