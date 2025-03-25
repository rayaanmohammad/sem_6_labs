#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void copyAndCount(char* input, char* output, int inputLength) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < inputLength) {
        int copyLength = inputLength - idx;
        int outputIdx = (inputLength - (copyLength+1) + 1) * ((copyLength+1) + inputLength) / 2;
        for (int i = 0; i < copyLength; ++i) {
            output[outputIdx + i] = input[i]; 
        }
    }
}

int main() {
    char word[100];
    printf("Enter the word: ");
    scanf("%s", word);
    int inputLength = strlen(word);

    int outputSize = (inputLength * (inputLength + 1)) / 2; 

    char* d_input;
    char* d_output;
    char* h_output = (char*)malloc(outputSize * sizeof(char));

    cudaMalloc((void**)&d_input, inputLength * sizeof(char));
    cudaMalloc((void**)&d_output, outputSize * sizeof(char)); 
    
    cudaMemcpy(d_input, word, inputLength * sizeof(char), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (inputLength + blockSize - 1) / blockSize;
    copyAndCount<<<numBlocks, blockSize>>>(d_input, d_output, inputLength);

    cudaMemcpy(h_output, d_output, outputSize * sizeof(char), cudaMemcpyDeviceToHost);

    printf("Output String: ");
    for (int i = 0; i < outputSize; i++) {
        printf("%c", h_output[i]);
    }
    printf("\n");

    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
