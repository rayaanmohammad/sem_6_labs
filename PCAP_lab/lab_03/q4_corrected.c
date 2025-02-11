#include<stdio.h>
#include<mpi.h>
#include<string.h>
#include<stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size, chars_per_string;
    char A[100], B[100];
    char *local_string_1;
    char *local_string_2;
    char *local_string_final;
    char *final_string;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(rank == 0) {
        fprintf(stdout, "Enter string 1: ");
        fflush(stdout);
        scanf("%s", A);
        fprintf(stdout, "Enter string 2: ");
        fflush(stdout);
        scanf("%s", B);

        if(strlen(A) % size != 0 || strlen(B) % size != 0 || strlen(A) != strlen(B)) {
            fprintf(stdout, "Strings don't follow the rules.\n");
            fflush(stdout);
            MPI_Finalize();
            return 0;
        }

        chars_per_string = strlen(A) / size;
    }
    MPI_Bcast(&chars_per_string, 1, MPI_INT, 0, MPI_COMM_WORLD);

    local_string_1 = (char*)malloc(chars_per_string * sizeof(char));
    local_string_2 = (char*)malloc(chars_per_string * sizeof(char));
    local_string_final = (char*)malloc(chars_per_string * 2 * sizeof(char));

    MPI_Scatter(A, chars_per_string, MPI_CHAR, local_string_1, chars_per_string, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, chars_per_string, MPI_CHAR, local_string_2, chars_per_string, MPI_CHAR, 0, MPI_COMM_WORLD);

    for(int i = 0; i < chars_per_string; i++) {
        local_string_final[2 * i] = local_string_1[i];
        local_string_final[2 * i + 1] = local_string_2[i];
    }

    if(rank == 0) {
        final_string = (char*)malloc((chars_per_string * 2 * size + 1) * sizeof(char));
    }

    MPI_Gather(local_string_final, chars_per_string * 2, MPI_CHAR, final_string, chars_per_string * 2, MPI_CHAR, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        final_string[chars_per_string * 2 * size] = '\0';
        fprintf(stdout, "The final string: %s\n", final_string);
        fflush(stdout);
        free(final_string);
    }
    free(local_string_1);
    free(local_string_2);
    free(local_string_final);

    MPI_Finalize();

    return 0;
}
