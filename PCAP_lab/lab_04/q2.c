#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


int search_row(int row[], int target) {
    int count = 0;
    for (int j = 0; j < 3; j++) {
        if (row[j] == target) {
            count++;
        }
    }
    return count;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int matrix[3][3];
    int target, local_count = 0, total_count = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 3) {
        if (rank == 0) {
            fprintf(stderr, "Error: This program requires exactly 3 processes.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("Enter the 3x3 matrix elements (row-wise):\n");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                scanf("%d", &matrix[i][j]);
            }
        }
        printf("Enter the element to search for: ");
        scanf("%d", &target);
    }

    MPI_Bcast(&target, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(matrix, 3 * 3, MPI_INT, 0, MPI_COMM_WORLD);

    local_count = search_row(matrix[rank], target);

    MPI_Scan(&local_count, &total_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 2) {
        printf("The element %d appeared %d times in the matrix.\n", target, total_count);
    }

    MPI_Finalize();

    return 0;
}
