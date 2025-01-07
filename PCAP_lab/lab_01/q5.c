#include "mpi.h"
#include <stdio.h>

unsigned long long fibonacci(int n) {
    if (n <= 1)
        return n;
    
    unsigned long long a = 0, b = 1, fib;
    for (int i = 2; i <= n; i++) {
        fib = a + b;
        a = b;
        b = fib;
    }
    return b;
}
unsigned long long factorial(int n) {
    if (n == 0 || n == 1)
        return 1;
    
    unsigned long long result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}
int main(int argc, char *argv[])
{
int rank,size;
MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
if(rank%2==0){
    printf("For process %d: %llu \n",rank,factorial(rank));
}
else{
    printf("For process %d: %llu \n", rank,fibonacci(rank));
}
MPI_Finalize();
return 0;
}