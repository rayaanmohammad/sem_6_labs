#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

__global__ void linearAlgebraKernel(float *x, float *y, float alpha, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        y[idx] = alpha * x[idx] + y[idx];
    }
}

int main() {
    int N;
    float alpha;

    printf("Enter the length of the vectors (N): ");
    scanf("%d", &N);
    printf("Enter the scalar alpha: ");
    scanf("%f", &alpha);

    size_t size = N * sizeof(float);

    float *X = (float *)malloc(size);
    float *Y = (float *)malloc(size);

    printf("Enter elements for vector x: ");
    for (int i = 0; i < N; i++) {
        scanf("%f", &X[i]);
    }

    printf("Enter elements for vector y: ");
    for (int i = 0; i < N; i++) {
        scanf("%f", &Y[i]);
    }

    float *d_x, *d_y;
    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);

    cudaMemcpy(d_x, X, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, Y, size, cudaMemcpyHostToDevice);

    linearAlgebraKernel<<<1, N>>>(d_x, d_y, alpha, N);

    cudaMemcpy(Y, d_y, size, cudaMemcpyDeviceToHost);

    printf("Resultant vector y: ");
    for (int i = 0; i < N; i++) {
        printf("%f ", Y[i]);
    }
    printf("\n");

    cudaFree(d_x);      
    cudaFree(d_y);      

    free(X);            
    free(Y);
    return 0;
}
