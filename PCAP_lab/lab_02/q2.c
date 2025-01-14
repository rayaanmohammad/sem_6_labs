#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <ctype.h>

int main(int argc, char *argv[])
{
    int rank, size, i, received_value;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if(size<2){
        printf("Requires at least two processes, try again\n");
        MPI_Finalize();
        return 0;
    }
    
    if(rank == 0) {
        for(i=1; i<size; i++){
            MPI_Send(&i, 1, MPI_INT, i, i, MPI_COMM_WORLD);
        }
    }
    
    if(rank != 0) {
        MPI_Recv(&received_value, 1, MPI_INT, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        fprintf(stdout, "Rank %d received the integer %d from process 0\n", rank, received_value);
        fflush(stdout);
    }
    
    MPI_Finalize();
    return 0;
}
