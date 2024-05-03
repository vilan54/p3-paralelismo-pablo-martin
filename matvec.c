#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define N 1024
#define DEBUG 1

int main(int argc, char *argv[]) {
    int i, j;
    int rank, size;
    int block_size;
    int padded_N;
    struct timeval tv1, tv2;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    if(N % size == 0){
        block_size = N/size;
        padded_N = N;
    } else {
        block_size = ceil(N / (float) size); // Calculate the block size with padding
        padded_N = N + (size - (N % size));
    }

    float matrix[padded_N][N];
    float result[padded_N];
    float vector[padded_N];
    float local_matrix[block_size][N];
    float local_result[block_size];


    // Inicializar la matriz y el vector en el proceso 0
    if (rank == 0) {
        for (i = 0; i < padded_N; i++) {
            for (j = 0; j < N; j++) {
                if(i<N){
                    matrix[i][j] = i + j;
                } else {
                    matrix[i][j] = 0;
                }
            }
            vector[i] = i;
        }
    }
    MPI_Scatter(matrix, block_size * N, MPI_FLOAT, local_matrix, block_size * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    gettimeofday(&tv1, NULL);

    // Broadcast vector a todos los procesos
    MPI_Bcast(vector, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // Realizar el cálculo del producto matriz-vector localmente en cada proceso
    for(i=0;i<block_size;i++) {
        local_result[i]=0;
        for(j=0;j<N;j++) {
            local_result[i] += local_matrix[i][j]*vector[j];
        }
    }

    MPI_Gather(local_result, block_size, MPI_FLOAT, result, block_size , MPI_FLOAT, 0, MPI_COMM_WORLD);


    gettimeofday(&tv2, NULL);

    if (rank == 0) {
        int microseconds = (tv2.tv_usec - tv1.tv_usec) + 1000000 * (tv2.tv_sec - tv1.tv_sec);
        
        if (DEBUG) {
            for (i = 0; i < N; i++) {
                printf(" %f \t ", result[i]);
            }
        } else {
            printf("Time (seconds) = %lf\n", (double) microseconds / 1E6);
        }
    }

    MPI_Finalize();
    return 0;
}