#include "mpi.h"
#include <stdio.h>
#include<stdlib.h>

int main(int argc, char *argv[])
{
    int rank, size, *array, received_value, result;
    char *buffer;
    int buffer_size;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    buffer_size = MPI_BSEND_OVERHEAD + sizeof(int);
    buffer = malloc(buffer_size);
    MPI_Buffer_attach(buffer, buffer_size);

    if(rank == 0) {
        array = malloc(size * sizeof(int));
        printf("Enter %d elements:\n", size);
        for(int i = 0; i < size; i++) {
            scanf("%d", &array[i]);
        }
        for(int i = 1; i < size; i++) {
            MPI_Bsend(&array[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    }
    
    if(rank != 0) {
        MPI_Recv(&received_value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if(rank % 2 == 0) {
            result = received_value * received_value;
            printf("Rank %d: Square of %d is %d\n", rank, received_value, result);
        }
        else {
            result = received_value * received_value * received_value;
            printf("Rank %d: Cube of %d is %d\n", rank, received_value, result);
        }
    }
    
    MPI_Finalize();
    return 0;
}
