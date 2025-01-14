#include "mpi.h"
#include <stdio.h>

int main(int argc, char *argv[])
{
    int rank, size, x;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if(size<2){
        fprintf(stdout,"Requires at least two processes, try again\n");
        fflush(stdout);
        MPI_Finalize();
        return 0;
    }
    
    if(rank == 0) {
        fprintf(stdout,"Enter a number: ");
        fflush(stdout);
        scanf("%d", &x);
        x++;
        MPI_Ssend(&x, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
        MPI_Recv(&x, 1, MPI_INT, size-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        fprintf(stdout,"Finally rank %d receives the number: %d\n", rank, x);
        fflush(stdout);
    }
    
    if(rank != 0) {
        MPI_Recv(&x, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        fprintf(stdout,"Rank %d received the number: %d\n", rank, x);
        fflush(stdout);
        x++;
        
        if(rank == size-1){
            MPI_Ssend(&x, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
        else{
            MPI_Ssend(&x, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
        }
    }
    
    MPI_Finalize();
    return 0;
}
