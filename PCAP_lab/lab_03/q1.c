#include "mpi.h"
#include<stdio.h>

int factorial(int n){
    int result=1;
    for(int i=1;i<=n;i++){
        result *= i;
    }
    return result;
}

int main(int argc, char *argv[]){
    int rank, size, N, A[10], B[10], c, i, sum=0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    if(rank==0){
        N=size;
        fprintf(stdout, "Enter %d values: \n",N);
        fflush(stdout);
        for(i=0;i<N;i++)
            scanf("%d",&A[i]);
    }

    MPI_Scatter(A,1,MPI_INT,B,1,MPI_INT,0,MPI_COMM_WORLD);
    fprintf(stdout,"Rank %d recieved: %d\n",rank,B[0]);
    fflush(stdout);

    c = factorial(B[0]);
    MPI_Gather(&c,1,MPI_INT,B,1,MPI_INT,0,MPI_COMM_WORLD);

    if(rank==0){
        fprintf(stdout, "The result gathered in the root:\n");
        fflush(stdout);
        for(int i=0;i<N;i++){
            fprintf(stdout,"%d ",B[i]);
            fflush(stdout);
            sum += B[i];
        }
        fprintf(stdout,"\nThe final sum: %d\n",sum);
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}