#include <stdio.h>
#include <sys/time.h>
#include <mpi.h>

#define DEBUG 1
#define N 1024

int main(int argc, char *argv[]) {
    int i, j;
    int rank, size;
    int block_size, padded_block_size;
    int remainder;
    float matrix[N][N];
    float vector[N];
    float local_vector[N];
    float result[N];
    struct timeval tv1, tv2;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    block_size = N / size;
    remainder = N % size;

    float local_matrix[block_size][N];
    float local_result[block_size];


    // Initialize Matrix and Vector
    if(rank == 0){
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                matrix[i][j] = i + j;
            }
            vector[i] = i;
        }
    }

    // Broadcast vector to all processes
    MPI_Bcast(vector, N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Scatter matrix to all processes
    MPI_Scatter(matrix, block_size * N, MPI_FLOAT, local_matrix, block_size * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    for (i = 0; i < block_size; i++) {
        local_result[i] = 0;
        for (j = 0; j < N; j++) {
            local_result[i] += local_matrix[i][j] * vector[j];
        }
    }

    // Gather local results
    MPI_Gather(local_result, block_size, MPI_FLOAT, result, block_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int microseconds = (tv2.tv_usec - tv1.tv_usec) + 1000000 * (tv2.tv_sec - tv1.tv_sec);
        
        if (DEBUG){
            for(i=0;i<N;i++) {
                printf(" %f \t ",result[i]);
            }
        } else {
            printf ("Time (seconds) = %lf\n", (double) microseconds/1E6);
        } 
    } 

    MPI_Finalize();
    return 0;
}
