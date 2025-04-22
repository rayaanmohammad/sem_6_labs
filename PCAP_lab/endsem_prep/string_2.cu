#include <stdio.h>
#include <cuda_runtime.h>
#include <string.h>
#include <stdlib.h>

__device__ int findStart(int n, int idx) {
    int answer = n * (n + 1) / 2;
    int after = n - idx;
    answer = answer - (after * (after + 1) / 2);
    return answer;
}

__global__ void change_string(char* d_s, char* d_result, int* d_len) {
    int idx = threadIdx.x;
    int len = *d_len;
    int len_2 = len - idx;
    int start_index = findStart(len, idx);

    if (idx < len) {
        for (int i = 0; i < len_2; i++) {
            d_result[start_index + i] = d_s[i];
        }
    }
}

int main() {
    char s[100];
    int len, result_len;

    printf("Enter the string: ");
    scanf("%s", s);

    len = strlen(s);
    result_len = len * (len + 1) / 2;

    char* result = (char*)malloc(result_len * sizeof(char));

    char *d_s, *d_result;
    int *d_len;

    cudaMalloc((void**)&d_s, len * sizeof(char));
    cudaMalloc((void**)&d_result, result_len * sizeof(char));
    cudaMalloc((void**)&d_len, sizeof(int));

    cudaMemcpy(d_s, s, len * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_len, &len, sizeof(int), cudaMemcpyHostToDevice);

    change_string<<<1, len>>>(d_s, d_result, d_len);
    cudaDeviceSynchronize();

    cudaMemcpy(result, d_result, result_len * sizeof(char), cudaMemcpyDeviceToHost);

    for (int i = 0; i < result_len; i++) {
        printf("%c", result[i]);
    }
    printf("\n");

    cudaFree(d_s);
    cudaFree(d_result);
    cudaFree(d_len);
    free(result);

    return 0;
}
