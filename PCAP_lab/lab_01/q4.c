#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <ctype.h>

int main(int argc, char *argv[])
{
int rank,size;
char s[] = "HELLO"; 
MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
if(size<strlen(s)){
    printf("Require number of threads more than the characters in the string, try again\n");
    MPI_Finalize();
    return 0;
}
if(rank<strlen(s)){
    char c = s[rank];
    s[rank] = tolower(s[rank]);
    printf("Process %d changed %c to %c : %s\n",rank,c,s[rank],s);
}
MPI_Finalize();
return 0;
}