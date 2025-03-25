#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void transformMatrix(int *input, int *output, int rows, int cols) {
    int row = blockIdx.x;
    if (row < rows) {
        int power = row + 1; 
        for (int col = 0; col < cols; col++) {
            int base = input[row * cols + col];
            int result = 1;
            for (int p = 0; p < power; p++) { 
                result *= base;
            }
            output[row * cols + col] = result;
        }
    }
}

int main() {
    int rows, cols;
    printf("Enter number of rows and columns: ");
    scanf("%d %d", &rows, &cols);

    int *h_input = (int*)malloc(rows * cols * sizeof(int));
    int *h_output = (int*)malloc(rows * cols * sizeof(int));

    printf("Enter matrix elements:\n");
    for (int i = 0; i < rows * cols; i++) {
        scanf("%d", &h_input[i]);
    }

    int *d_input, *d_output;
    cudaMalloc((void**)&d_input, rows * cols * sizeof(int));
    cudaMalloc((void**)&d_output, rows * cols * sizeof(int));

    cudaMemcpy(d_input, h_input, rows * cols * sizeof(int), cudaMemcpyHostToDevice);

    transformMatrix<<<rows, 1>>>(d_input, d_output, rows, cols);

    cudaMemcpy(h_output, d_output, rows * cols * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nOriginal Matrix:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", h_input[i * cols + j]);
        }
        printf("\n");
    }

    printf("\nTransformed Matrix:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", h_output[i * cols + j]);
        }
        printf("\n");
    }

    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
