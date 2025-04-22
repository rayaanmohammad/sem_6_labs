#include<stdio.h>
#include<cuda_runtime.h>

__global__ void printhello(){
    int idx = threadIdx.x;
    printf("Hello from thread no: %d\n",idx);
    return;
}

int main(){
    int n = 0;
    printf("Enter the number of threads you want:");
    scanf("%d",&n);
    printhello<<<1,n>>>();
    cudaDeviceSynchronize();
    return 0;
}