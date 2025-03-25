#include <stdio.h>
#include <cuda_runtime.h>

__global__ void addRowWise(int* A, int* B, int* C, int n, int m) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < n) {
        for (int j = 0; j < m; j++) {
            C[row * m + j] = A[row * m + j] + B[row * m + j];
        }
    }
}

__global__ void addColumnWise(int* A, int* B, int* C, int n, int m) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (col < m) {
        for (int i = 0; i < n; i++) {
            C[i * m + col] = A[i * m + col] + B[i * m + col];
        }
    }
}

__global__ void addElementWise(int* A, int* B, int* C, int n, int m) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row < n && col < m) {
        C[row * m + col] = A[row * m + col] + B[row * m + col];
    }
}

int main() {
    int n, m;
    printf("Enter number of rows (N) and columns (M): ");
    scanf("%d %d", &n, &m);

    int size = n * m * sizeof(int);
    int *A, *B, *C;
    A = (int*)malloc(size);
    B = (int*)malloc(size);
    C = (int*)malloc(size);

    printf("Enter elements for matrix A (%d x %d):\n", n, m);
    for (int i = 0; i < n * m; i++) scanf("%d", &A[i]);

    printf("Enter elements for matrix B (%d x %d):\n", n, m);
    for (int i = 0; i < n * m; i++) scanf("%d", &B[i]);

    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    printf("\nRow-wise Addition:\n");
    addRowWise<<<(n + 255) / 256, 256>>>(d_A, d_B, d_C, n, m);
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%d ", C[i * m + j]);
        }
        printf("\n");
    }

    printf("\nColumn-wise Addition:\n");
    addColumnWise<<<(m + 255) / 256, 256>>>(d_A, d_B, d_C, n, m);
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%d ", C[i * m + j]);
        }
        printf("\n");
    }

    printf("\nElement-wise Addition:\n");
    dim3 grid(n, 1);
    dim3 block(m, 1);
    addElementWise<<<grid, block>>>(d_A, d_B, d_C, n, m);
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%d ", C[i * m + j]);
        }
        printf("\n");
    }

    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
