#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size, M, N, i;
    int *A = NULL;
    double local_sum = 0.0, local_average = 0.0;
    double total_average_final = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        fprintf(stdout, "Enter the value of M: ");
        fflush(stdout);
        scanf("%d", &M);
    }
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);

    N = M * size; 

    if (rank == 0) {
        A = (int *)malloc(N * sizeof(int));
        fprintf(stdout, "Enter %d elements for all processes (total %d elements): \n", N, N);
        fflush(stdout);
        for (i = 0; i < N; i++) {
            scanf("%d", &A[i]);
        }
    }

    int *sub_array = (int *)malloc(M * sizeof(int));

    MPI_Scatter(A, M, MPI_INT, sub_array, M, MPI_INT, 0, MPI_COMM_WORLD);

    for (i = 0; i < M; i++) {
        local_sum += sub_array[i];
    }

    local_average = local_sum / M;
    fprintf(stdout, "Process %d average: %.6f\n", rank, local_average);
    fflush(stdout);

    double *local_averages = NULL;
    if (rank == 0) {
        local_averages = (double *)malloc(size * sizeof(double));
    }

    MPI_Gather(&local_average, 1, MPI_DOUBLE, local_averages, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double total_sum_of_averages = 0.0;
        for (i = 0; i < size; i++) {
            total_sum_of_averages += local_averages[i];
        }
        total_average_final = total_sum_of_averages / size;

        fprintf(stdout, "The average of averages is: %.2f\n", total_average_final);
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}
