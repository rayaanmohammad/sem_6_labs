#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

long long factorial(int n) {
    long long result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}

void handle_error(int error_code) {
    char error_string[256];
    int length_of_error, error_class;
    MPI_Error_class(error_code, &error_class);
    MPI_Error_string(error_class, error_string, &length_of_error);
    fprintf(stderr, "Error occurred: %d - %s\n", error_class,error_string);
    MPI_Abort(MPI_COMM_WORLD, error_code);
}

int main(int argc, char *argv[]) {
    int rank, size, N;
    long long local_fact, global_sum;
    int error_code;

    MPI_Init(&argc, &argv);
    MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 1) {
        fprintf(stderr, "Error: The number of processes must be at least 1.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("Enter the value of N: ");
        fflush(stdout);
        if (scanf("%d", &N) != 1 || N <= 0) {
            fprintf(stderr, "Error: Invalid input for N.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (N > size) {
            fprintf(stderr, "Error: N (%d) is greater than the number of processes (%d).\n", N, size);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }   

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank >= N) {
        local_fact = 0;
    } else {
        local_fact = factorial(rank + 1);
    }

    MPI_Comm invalid_comm;
    /*This is line throws an error*/
    /*error_code = MPI_Scan(&local_fact, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, invalid_comm);*/
    error_code = MPI_Scan(&local_fact, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (error_code != MPI_SUCCESS) {
        handle_error(error_code);
    }

    if (rank == N - 1) {
        printf("The sum of factorials from 1! to %d! is: %lld\n", N, global_sum);
    }

    MPI_Finalize();
    return 0;
}
