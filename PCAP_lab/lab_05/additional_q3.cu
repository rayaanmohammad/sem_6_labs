#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

__global__ void oddEvenTranspositionSortKernel(int *d_arr, int N, int phase) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N - 1) {
        if (phase % 2 == 0) {
            if (idx % 2 == 0 && d_arr[idx] > d_arr[idx + 1]) {
                int temp = d_arr[idx];
                d_arr[idx] = d_arr[idx + 1];
                d_arr[idx + 1] = temp;
            }
        } else {
            if (idx % 2 != 0 && d_arr[idx] > d_arr[idx + 1]) {
                int temp = d_arr[idx];
                d_arr[idx] = d_arr[idx + 1];
                d_arr[idx + 1] = temp;
            }
        }
    }
}

int main() {
    int N;
    printf("Enter the number of elements in the array: ");
    scanf("%d", &N);

    size_t size = N * sizeof(int);
    int *h_arr = (int *)malloc(size);

    printf("Enter elements for the array: \n");
    for (int i = 0; i < N; i++) {
        scanf("%d", &h_arr[i]);
    }

    int *d_arr;
    cudaMalloc((void **)&d_arr, size);

    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

    int numThreadsPerBlock = N;
    int numBlocks = 1;

    for (int phase = 0; phase < N; phase++) {
        oddEvenTranspositionSortKernel<<<numBlocks, numThreadsPerBlock>>>(d_arr, N, phase);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);

    printf("Sorted array: \n");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    free(h_arr);
    cudaFree(d_arr);
    return 0;
}
