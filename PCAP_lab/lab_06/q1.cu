#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

__global__ void convulution_one_dimension(float *N, float *M, float *P, int width, int mask_width) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    float pvalue = 0;
    int i = idx - (mask_width/2);
    for(int j = 0 ;j<mask_width;j++){
        if(i+j>=0 && i+j<width){
            pvalue += N[i+j] * M[j];
        }
    }
    P[idx] = pvalue;
}

int main() {
    int width, mask_width;
    
    printf("Enter the length of the width of array N: ");
    scanf("%d",&width);

    size_t size_N = width * sizeof(float);
    float *N = (float *)malloc(size_N);
    float *P = (float *)malloc(size_N);

    printf("Enter elements for array N: ");
    for (int i = 0; i < width; i++) {
        scanf("%f", &N[i]);
    }

    printf("Enter the length of the width of mask array M: ");
    scanf("%d",&mask_width);

    size_t size_M = mask_width * sizeof(float);
    float *M = (float *)malloc(size_M);

    printf("Enter elements for mask M: ");
    for (int i = 0; i < mask_width; i++) {
        scanf("%f", &M[i]);
    }

    float *d_N, *d_M, *d_P;
    cudaMalloc((void **)&d_N, size_N);
    cudaMalloc((void **)&d_M, size_M);
    cudaMalloc((void **)&d_P, size_N);

    cudaMemcpy(d_N, N, size_N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, M, size_M, cudaMemcpyHostToDevice);

    convulution_one_dimension<<<1, width>>>(d_N, d_M, d_P, width, mask_width);

    cudaMemcpy(P, d_P, size_N, cudaMemcpyDeviceToHost);

    printf("Result of convolution operation: ");
    for (int i = 0; i < width; i++) {
        printf("%f ",P[i]);
    }
    printf("\n");
    
    cudaFree(d_N);
    cudaFree(d_M);
    cudaFree(d_P);

    free(N);
    free(M);
    free(P);
    return 0;
}
