#include <stdio.h>
#include<stdlib.h>
#include <math.h>
#include <sys/time.h>

#define LOOP 10
#define P 6700417
#define MU 22
#define NUM_VALUES 0x2000000

#define INLINE

static int logisticsmap_calc(int x, int p, int mu) {
    return mu * x * (x + 1) % p;
}

static int logisticsmap_loopCalc(int num, int x, int p, int mu) {
    for(int i = 0; i < num; i++) {
        x = logisticsmap_calc(x, p, mu);
    }
    return x;
}

#ifdef INLINE
__global__
void logisticsmap(int *x, int *p, int *mu, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    x[i] = mu[i] * x[i] * (x[i] + 1) % p[i];
    x[i] = mu[i] * x[i] * (x[i] + 1) % p[i];
    x[i] = mu[i] * x[i] * (x[i] + 1) % p[i];
    x[i] = mu[i] * x[i] * (x[i] + 1) % p[i];
    x[i] = mu[i] * x[i] * (x[i] + 1) % p[i];
    x[i] = mu[i] * x[i] * (x[i] + 1) % p[i];
    x[i] = mu[i] * x[i] * (x[i] + 1) % p[i];
    x[i] = mu[i] * x[i] * (x[i] + 1) % p[i];
    x[i] = mu[i] * x[i] * (x[i] + 1) % p[i];
    x[i] = mu[i] * x[i] * (x[i] + 1) % p[i];
}
#else
__global__
void logisticsmap(int *x, int *p, int *mu, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for(int j = 0; j < LOOP; j++) {
    	x[i] = mu[i] * x[i] * (x[i] + 1) % p[i];
    }
}
#endif

static void EC(cudaError_t err, const char *message) {
	if(err != cudaSuccess) {
		fprintf(stderr, "error in %s\n", message);
		exit(EXIT_FAILURE);
	}
}

int main(void){
    int *x, *p, *mu, *d_x, *d_p, *d_mu;
    const size_t n_byte = NUM_VALUES * sizeof(float);

    x = (int *)malloc(n_byte);
    p = (int *)malloc(n_byte);
    mu = (int *)malloc(n_byte);

    for(int i = 0; i < NUM_VALUES; i++) {
    	x[i] = i;
    	p[i] = P;
    	mu[i] = MU;
    }

    printf("start cudaMalloc\n");
    EC(cudaMalloc((void**)&d_x, NUM_VALUES), "cudaMalloc");
    EC(cudaMalloc((void**)&d_p, NUM_VALUES), "cudaMalloc");
    EC(cudaMalloc((void**)&d_mu, NUM_VALUES), "cudaMalloc");
    printf("finish cudaMalloc\n");

    printf("%p\n", d_x);

    printf("start cudaMemcpy\n");
    EC(cudaMemcpy(d_x, x, n_byte, cudaMemcpyHostToDevice), "cudaMemcpy: HostToDevice");
    EC(cudaMemcpy(d_p, p, n_byte, cudaMemcpyHostToDevice), "cudaMemcpy: HostToDevice");
    EC(cudaMemcpy(d_mu, mu, n_byte, cudaMemcpyHostToDevice), "cudaMemcpy: HostToDevice");
    printf("finish cudaMemcpy\n");

    printf("start kernel function\n");
    logisticsmap<<<(NUM_VALUES+255)/256, 256>>>(d_x, d_p, d_mu, n_byte);
    printf("finish kernel function\n");
    EC(cudaMemcpy(x, d_x, n_byte, cudaMemcpyDeviceToHost), "cudaMemcpy: DeviceToHost");

    EC(cudaFree(d_x), "cudaFree");
    EC(cudaFree(d_p), "cudaFree");
    EC(cudaFree(d_mu), "cudaFree");

    for(int i = 0; i < NUM_VALUES; i++) {
    	int expected = logisticsmap_loopCalc(10, i, p[i], mu[i]);
    	if(expected != x[i]) {
    		printf("invalid value of %d, expeted %d but %d\n", i, expected, x[i]);
    		return EXIT_FAILURE;
    	}
    }
    return EXIT_SUCCESS;
}
