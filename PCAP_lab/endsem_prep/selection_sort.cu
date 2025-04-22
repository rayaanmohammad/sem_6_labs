#include<stdio.h>
#include<cuda_runtime.h>

__global__ void sortRow(int *a, int n, int m){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i=0; i<m-1; i++){
        int min = i;
        for(int j=i; j<m; j++){
            if(a[row*m + j]<a[row*m + min])
                min = j;
        }
        int temp = a[row*m + i];
        a[row*m + i] = a[row*m + min];
        a[row*m + min] = temp;
    }
}

int main(){
    int n, m;
    printf("Enter n: ");
    scanf("%d",&n);
    printf("Enter m: ");
    scanf("%d",&m);
    int size = n*m*sizeof(int);
    int *a = (int*)malloc(size);
    int *result = (int*)malloc(size);

    printf("Enter the matrix:\n");
    for(int i = 0; i < n*m; i++){
        scanf("%d",&a[i]);
    }

    int *d_a, *d_result;
    cudaMalloc((void**)&d_a,size);
    cudaMalloc((void**)&d_result,size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    sortRow<<<1,n>>> (d_a, n, m);

    cudaMemcpy(result, d_a, size, cudaMemcpyDeviceToHost);

    printf("\nResultant matrix:\n");
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            printf("%d ",result[i*m+j]);
        }
        printf("\n");
    }

    cudaFree(d_a);
    free(a);
    free(result);

    return 0;
}