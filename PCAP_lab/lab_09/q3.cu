#include <stdio.h>
#include <cuda_runtime.h>

__device__ int toBinary(int num, char binary[]) {
    int index = 0;

    if (num == 0) {
        binary[0] = '0';
        binary[1] = '\0';
        return 1;
    }

    while (num > 0) {
        binary[index++] = (num % 2) + '0';
        num /= 2;
    }
    binary[index] = '\0';

    for (int i = 0; i < index / 2; i++) {
        char temp = binary[i];
        binary[i] = binary[index - i - 1];
        binary[index - i - 1] = temp;
    }

    return index;
}

__device__ void onesComplement(char binary[], int length) {
    for (int i = 0; i < length; i++) {
        binary[i] = (binary[i] == '0') ? '1' : '0';
    }
}

__device__ int binaryStringToInt(const char binary[], int length) {
    int result = 0;
    for (int i = 0; i < length; i++) {
        result = result * 10 + (binary[i] - '0');
    }
    return result;
}

__global__ void modifyMatrix(int *mat, int *output, int rows, int cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    int idx = row * cols + col;

    if (row > 0 && row < rows - 1 && col > 0 && col < cols - 1) {
        char binary[32];
        int length = toBinary(mat[idx], binary);
        onesComplement(binary, length);
        output[idx] = binaryStringToInt(binary, length);
    } else {
        output[idx] = mat[idx]; 
    }
}

int main() {
    int n, m;
    printf("Enter number of rows and columns: ");
    scanf("%d %d", &n, &m);

    int size = n * m * sizeof(int);
    int *A = (int*)malloc(size);
    int *result = (int*)malloc(size);

    printf("Enter elements for matrix (%d x %d):\n", n, m);
    for (int i = 0; i < n * m; i++) {
        scanf("%d", &A[i]);
    }

    int *d_A, *d_result;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_result, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    dim3 grid(n, 1);
    dim3 block(m, 1);
    modifyMatrix<<<grid, block>>>(d_A, d_result, n, m);

    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);

    printf("\nModified Matrix (Non-border elements replaced with one's complement stored as integer):\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%d ", result[i * m + j]);
        }
        printf("\n");
    }

    free(A);
    free(result);
    cudaFree(d_A);
    cudaFree(d_result);

    return 0;
}
