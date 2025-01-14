#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <ctype.h>

int main(int argc, char *argv[])
{
    int rank,size;
    char s[100];
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
        fprintf(stdout,"Enter a string: ");
        fflush(stdout);
        scanf("%s", s);
        MPI_Ssend(s, strlen(s)+1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(s, 100, MPI_CHAR, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        fprintf(stdout,"Modified string: %s\n", s);
        fflush(stdout);
    }
    if(rank == 1) {
        MPI_Recv(s, 100, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for(int i = 0; s[i]; i++) {
            if(islower(s[i]))
                s[i] = toupper(s[i]);
            else if(isupper(s[i]))
                s[i] = tolower(s[i]);
        }
        MPI_Ssend(s, strlen(s)+1, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}

