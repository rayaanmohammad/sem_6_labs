#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

__global__ void computeSine(float *angles, float *sineValues, int N) {
    int idx = threadIdx.x;
    if (idx < N) {
        sineValues[idx] = sin(angles[idx]);
    }
}

int main() {
    int N;
    printf("Enter the number of angles: ");
    scanf("%d", &N);

    size_t size = N * sizeof(float);

    float *h_angles = (float *)malloc(size);
    float *h_sineValues = (float *)malloc(size);

    printf("Enter the angles (in radians): ");
    for (int i = 0; i < N; i++) {
        scanf("%f", &h_angles[i]);
    }

    float *d_angles, *d_sineValues;
    cudaMalloc((void **)&d_angles, size);
    cudaMalloc((void **)&d_sineValues, size);

    cudaMemcpy(d_angles, h_angles, size, cudaMemcpyHostToDevice);

    computeSine<<<1, N>>>(d_angles, d_sineValues, N);

    cudaMemcpy(h_sineValues, d_sineValues, size, cudaMemcpyDeviceToHost);

    printf("Sine values of the angles: ");
    for (int i = 0; i < N; i++) {
        printf("%f ", h_sineValues[i]);
    }
    printf("\n");

    cudaFree(d_angles);      
    cudaFree(d_sineValues);  

    free(h_angles);          
    free(h_sineValues);
    return 0;
}
