#include <stdio.h>
#include <cuda_runtime.h>

__global__ void oddEvenSort(int *a, int n, int phase) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int i = idx * 2 + phase;

    if (i + 1 < n) {
        if (a[i] > a[i + 1]) {
            int temp = a[i];
            a[i] = a[i + 1];
            a[i + 1] = temp;
        }
    }
}

int main() {
    int n;
    printf("Enter number of elements: ");
    scanf("%d", &n);

    int *a = (int *)malloc(n * sizeof(int));
    int *result = (int *)malloc(n * sizeof(int));

    printf("Enter array elements:\n");
    for (int i = 0; i < n; i++) {
        scanf("%d", &a[i]);
    }

    int *d_a;
    cudaMalloc((void**)&d_a, n * sizeof(int));
    cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numThreads = (n + 1) / 2;
    int blocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;

    for (int phase = 0; phase < n; phase++) {
        oddEvenSort<<<blocks, threadsPerBlock>>>(d_a, n, phase % 2);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(result, d_a, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nSorted array:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", result[i]);
    }
    printf("\n");

    cudaFree(d_a);
    free(a);
    free(result);

    return 0;
}
