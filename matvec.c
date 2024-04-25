#include <stdio.h>
#include <sys/time.h>
#include <mpi.h>

#define DEBUG 1

#define N 1024

int main(int argc, char *argv[] ) {

    int i, j, numprocs, rank, error_code;
    float matrix[N][N];
    float vector[N];
    float result[N];
    struct timeval  tv1, tv2, tv3, tv4;
    int microseconds, t_comm, tcomp;

    

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank==0){
        for(i=0;i<N;i++) {
            vector[i] = i;
            for(j=0;j<N;j++) {
                matrix[i][j] = i+j;
            }
        }
    }
    

    gettimeofday(&tv1, NULL);

    for(i=0;i<N;i++) {
        result[i]=0;
        for(j=0;j<N;j++) {
            result[i] += matrix[i][j]*vector[j];
        }
    }

    gettimeofday(&tv2, NULL);
    
    microseconds = (tv2.tv_usec - tv1.tv_usec)+ 1000000 * (tv2.tv_sec - tv1.tv_sec);


    if(rank==0){
        if (DEBUG){
            for(i=0;i<N;i++) {
                printf(" %f \t ",result[i]);
            }
        } else {
            printf ("Time (seconds) = %lf\n", (double) microseconds/1E6);
        }    
    }

    return 0;
}