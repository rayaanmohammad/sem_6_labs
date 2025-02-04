#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

__global__ void sortRowsKernel(int *matrix, int N, int M) {
    int rowIdx = threadIdx.x;

    if(rowIdx<M){
        int i, j, minIndex;
        int temp;

        for (i = 0; i < N - 1; i++) {
            minIndex = i;
            for (j = i + 1; j < N; j++) {
                if (matrix[rowIdx * N + j] < matrix[rowIdx * N + minIndex]) {
                    minIndex = j;
                }
            }
            temp = matrix[rowIdx * N + minIndex];
            matrix[rowIdx * N + minIndex] = matrix[rowIdx * N + i];
            matrix[rowIdx * N + i] = temp;
        }
    }
}

int main() {
    int N, M;

    printf("Enter the number of rows (M): ");
    scanf("%d", &M);
    printf("Enter the number of columns (N): ");
    scanf("%d", &N);

    size_t size = M * N * sizeof(int);

    int *h_matrix = (int *)malloc(size);

    printf("Enter elements for the matrix: \n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            scanf("%d", &h_matrix[i * N + j]);
        }
    }

    int *d_matrix;
    cudaMalloc((void **)&d_matrix, size);

    cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice);

    int numThreads = M;
    sortRowsKernel<<<1, numThreads>>>(d_matrix, N, M);

    cudaMemcpy(h_matrix, d_matrix, size, cudaMemcpyDeviceToHost);

    printf("Sorted matrix: \n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_matrix[i * N + j]);
        }
        printf("\n");
    }

    free(h_matrix);
    cudaFree(d_matrix);

    return 0;
}
