#include <stdio.h>
#include <cuda_runtime.h>

__global__ void multiplyRowWise(int* A, int* B, int* C, int N, int M, int P) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < N) {
        for (int j = 0; j < P; j++) {
            int sum = 0;
            for (int k = 0; k < M; k++) {
                sum += A[row * M + k] * B[k * P + j];
            }
            C[row * P + j] = sum;
        }
    }
}

__global__ void multiplyColumnWise(int* A, int* B, int* C, int N, int M, int P) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (col < P) {
        for (int i = 0; i < N; i++) {
            int sum = 0;
            for (int k = 0; k < M; k++) {
                sum += A[i * M + k] * B[k * P + col];
            }
            C[i * P + col] = sum;
        }
    }
}

__global__ void multiplyElementWise(int* A, int* B, int* C, int N, int M, int P) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row < N && col < P) {
        int sum = 0;
        for (int k = 0; k < M; k++) {
            sum += A[row * M + k] * B[k * P + col];
        }
        C[row * P + col] = sum;
    }
}

int main() {
    int N, M, P;
    printf("Enter dimensions (N M P) for matrices A (NxM) and B (MxP): ");
    scanf("%d %d %d", &N, &M, &P);

    int sizeA = N * M * sizeof(int);
    int sizeB = M * P * sizeof(int);
    int sizeC = N * P * sizeof(int);

    int *A, *B, *C;
    A = (int*)malloc(sizeA);
    B = (int*)malloc(sizeB);
    C = (int*)malloc(sizeC);

    printf("Enter elements for matrix A (%d x %d):\n", N, M);
    for (int i = 0; i < N * M; i++) scanf("%d", &A[i]);

    printf("Enter elements for matrix B (%d x %d):\n", M, P);
    for (int i = 0; i < M * P; i++) scanf("%d", &B[i]);

    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    printf("\nRow-wise Multiplication:\n");
    multiplyRowWise<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C, N, M, P);
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < P; j++) {
            printf("%d ", C[i * P + j]);
        }
        printf("\n");
    }

    printf("\nColumn-wise Multiplication:\n");
    multiplyColumnWise<<<(P + 255) / 256, 256>>>(d_A, d_B, d_C, N, M, P);
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < P; j++) {
            printf("%d ", C[i * P + j]);
        }
        printf("\n");
    }

    printf("\nElement-wise Multiplication:\n");
    dim3 grid(N, 1);
    dim3 block(P, 1);
    multiplyElementWise<<<grid, block>>>(d_A, d_B, d_C, N, M, P);
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < P; j++) {
            printf("%d ", C[i * P + j]);
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