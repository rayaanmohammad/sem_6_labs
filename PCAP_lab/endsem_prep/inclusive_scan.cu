#include <stdio.h>

__global__ void inclusiveScanKernel(int *d_in, int *d_out, int n) {
    extern __shared__ int temp[]; // shared memory

    int tid = threadIdx.x;

    // Load input into shared memory
    if (tid < n) temp[tid] = d_in[tid];
    __syncthreads();

    // Inclusive scan: Hillis-Steele algorithm
    for (int offset = 1; offset < n; offset *= 2) {
        int val = 0;
        if (tid >= offset) {
            val = temp[tid - offset];
        }
        __syncthreads();
        if (tid < n) {
            temp[tid] += val;
        }
        __syncthreads();
    }

    // Write result to global memory
    if (tid < n) d_out[tid] = temp[tid];
}

int main() {
    const int N = 8;
    int h_in[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    int h_out[N];

    int *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));

    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with 1 block of N threads and N shared memory size
    inclusiveScanKernel<<<1, N, N * sizeof(int)>>>(d_in, d_out, N);
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Inclusive scan result:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
