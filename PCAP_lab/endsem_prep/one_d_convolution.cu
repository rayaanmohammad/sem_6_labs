#include <stdio.h>
#include <cuda_runtime.h>

__global__ void convolution_1d(int *N, int *M, int *P, int width, int mask_width) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < width) {
        int half_mask = mask_width / 2;
        int sum = 0;
        int start_index = idx - half_mask;

        for (int j = 0; j < mask_width; j++) {
            int input_index = start_index + j;
            if (input_index >= 0 && input_index < width) {
                sum += N[input_index] * M[j];
            }
        }
        
        P[idx] = sum;  
    }
}

int main() {
    int width, mask_width;
    printf("Enter the width of input array: ");
    scanf("%d", &width);
    printf("Enter the width of the mask array (odd number): ");
    scanf("%d", &mask_width);

    int *N = (int *)malloc(width * sizeof(int));
    int *M = (int *)malloc(mask_width * sizeof(int));
    int *P = (int *)malloc(width * sizeof(int));

    printf("Enter the input array (size %d):\n", width);
    for (int i = 0; i < width; i++) {
        scanf("%d", &N[i]);
    }

    printf("Enter the mask array (size %d):\n", mask_width);
    for (int i = 0; i < mask_width; i++) {
        scanf("%d", &M[i]);
    }

    int *d_N, *d_M, *d_P;
    cudaMalloc((void**)&d_N, width * sizeof(int));
    cudaMalloc((void**)&d_M, mask_width * sizeof(int));
    cudaMalloc((void**)&d_P, width * sizeof(int));

    cudaMemcpy(d_N, N, width * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, M, mask_width * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (width + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d<<<blocks, threadsPerBlock>>>(d_N, d_M, d_P, width, mask_width);
    
    cudaDeviceSynchronize();

    cudaMemcpy(P, d_P, width * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nResulting Convoluted Array:\n");
    for (int i = 0; i < width; i++) {
        printf("%d ", P[i]);
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