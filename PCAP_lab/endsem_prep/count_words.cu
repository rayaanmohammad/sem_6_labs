#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_LEN 1024

__device__ bool matchAt(const char* sentence, const char* word, int pos, int wordLen, int sentLen) {
    if (pos + wordLen > sentLen) return false;
    for (int i = 0; i < wordLen; ++i) {
        if (sentence[pos + i] != word[i]) return false;
    }
    if ((pos + wordLen == sentLen || sentence[pos + wordLen] == ' ') &&
        (pos == 0 || sentence[pos - 1] == ' ')) {
        return true;
    }
    return false;
}

__global__ void countWordOccurrences(const char* sentence, const char* word, int* count, int wordLen, int sentLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < sentLen && matchAt(sentence, word, idx, wordLen, sentLen)) {
        atomicAdd(count, 1);
    }
}

int main() {
    char host_sentence[MAX_LEN];
    char host_word[100];

    printf("Enter a sentence:\n");
    fgets(host_sentence, MAX_LEN, stdin);
    host_sentence[strcspn(host_sentence, "\n")] = '\0';

    printf("Enter the word to count:\n");
    fgets(host_word, sizeof(host_word), stdin);
    host_word[strcspn(host_word, "\n")] = '\0'; 
    int wordLen = strlen(host_word);
    int sentLen = strlen(host_sentence);

    char *dev_sentence, *dev_word;
    int *dev_count;
    int host_count = 0;

    cudaMalloc((void**)&dev_sentence, sentLen * sizeof(char));
    cudaMalloc((void**)&dev_word, wordLen * sizeof(char));
    cudaMalloc((void**)&dev_count, sizeof(int));

    cudaMemcpy(dev_sentence, host_sentence, sentLen * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_word, host_word, wordLen * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_count, &host_count, sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (sentLen + threadsPerBlock - 1) / threadsPerBlock;

    countWordOccurrences<<<blocks, threadsPerBlock>>>(dev_sentence, dev_word, dev_count, wordLen, sentLen);
    cudaDeviceSynchronize();

    cudaMemcpy(&host_count, dev_count, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Word \"%s\" occurred %d times in the sentence.\n", host_word, host_count);

    cudaFree(dev_sentence);
    cudaFree(dev_word);
    cudaFree(dev_count);

    return 0;
}
