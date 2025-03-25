#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void countWordOccurrences(const char* sentence, const char* word, int* wordCount, int sentenceLength, int wordLength) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < sentenceLength - wordLength + 1) {
        bool match = true;
        for (int i = 0; i < wordLength; ++i) {
            if (sentence[idx + i] != word[i]) {
                match = false;
                break;
            }
        }
        if (match) {
            atomicAdd(wordCount, 1);
        }
    }
}

int main() {
    char sentence[100];
    char word[100];
    
    printf("Enter the sentence: ");
    fgets(sentence, sizeof(sentence), stdin); 
    sentence[strcspn(sentence, "\n")] = '\0';

    printf("Enter the word: ");
    scanf("%s", word);
    
    int sentenceLength = strlen(sentence);
    int wordLength = strlen(word);
    
    int* d_wordCount;
    int h_wordCount = 0;

    cudaMalloc((void**)&d_wordCount, sizeof(int));
    cudaMemcpy(d_wordCount, &h_wordCount, sizeof(int), cudaMemcpyHostToDevice);

    char* d_sentence;
    char* d_word;
    cudaMalloc((void**)&d_sentence, sentenceLength + 1);
    cudaMalloc((void**)&d_word, wordLength + 1);

    cudaMemcpy(d_sentence, sentence, sentenceLength + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_word, word, wordLength + 1, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (sentenceLength - wordLength + 1 + blockSize - 1) / blockSize;

    countWordOccurrences<<<numBlocks, blockSize>>>(d_sentence, d_word, d_wordCount, sentenceLength, wordLength);

    cudaMemcpy(&h_wordCount, d_wordCount, sizeof(int), cudaMemcpyDeviceToHost);

    printf("The word '%s' appears %d times in the sentence.\n", word, h_wordCount);

    cudaFree(d_sentence);
    cudaFree(d_word);
    cudaFree(d_wordCount);

    return 0;
}
