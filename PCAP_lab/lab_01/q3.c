#include "mpi.h"
#include <stdio.h>

int main(int argc, char *argv[])
{
int rank,size;
int a = 6;
int b = 3;
MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
if(size<4){
    printf("Require atleast 4 threads, try again\n");
    MPI_Finalize();
    return 0;
}
switch(rank){
    case 0:
        printf("Process %d: Operation %d + %d: %d\n",rank,a,b,a+b);
        break;
    case 1:
        printf("Process %d: Operation %d - %d: %d\n",rank,a,b,a-b);
        break;
    case 2:
        printf("Process %d: Operation %d * %d: %d\n",rank,a,b,a*b);
        break;
    case 3:
        if(b==0){
            printf("Process %d: Operation %d / %d: Division by zero, redefine the divisor\n",rank,a,b);
        }
        else{
            printf("Process %d: Operation %d / %d: %d\n",rank,a,b,a/b);
        }
        break;
}
MPI_Finalize();
return 0;
}