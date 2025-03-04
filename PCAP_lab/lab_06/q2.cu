#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

__global__ void compute_ranks(float *arr, int *ranks, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        int rank = 0;
        for (int j = 0; j < n; j++) {
            if (arr[j] < arr[idx]) {
                rank++;
            }
        }
        ranks[idx] = rank;
    }
}

void parallel_rank_based_selection_sort(float *arr, int n) {
    float *d_arr;
    int *d_ranks;
    size_t size = n * sizeof(float);
    
    cudaMalloc((void **)&d_arr, size);
    cudaMalloc((void **)&d_ranks, n * sizeof(int));

    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);

    compute_ranks<<<(n + 255) / 256, 256>>>(d_arr, d_ranks, n);

    int *ranks = (int *)malloc(n * sizeof(int));
    cudaMemcpy(ranks, d_ranks, n * sizeof(int), cudaMemcpyDeviceToHost);

    float *temp_arr = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        temp_arr[ranks[i]] = arr[i];
    }

    for (int i = 0; i < n; i++) {
        arr[i] = temp_arr[i];
    }

    free(temp_arr);
    cudaFree(d_arr);
    cudaFree(d_ranks);
}

int main() {
    int n;

    cout << "Enter the number of elements: ";
    cin >> n;

    float *arr = (float *)malloc(n * sizeof(float));

    cout << "Enter the elements of the array: ";
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }

    parallel_rank_based_selection_sort(arr, n);

    cout << "Sorted array: ";
    for (int i = 0; i < n; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;

    free(arr);
    return 0;
}
