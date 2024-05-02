#include <stdio.h>
#include <stdlib.h>
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
    float local_matrix[N][N];
    float local_vector[N];
    float local_result[N];
    float result[N];
    struct timeval tv1, tv2, tv3, tv4;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    block_size = N / size;
    remainder = N % size;
    padded_block_size = (N + size - 1) / size;

    // Adjust block size for processes handling extra data
    if (rank < remainder) {
        block_size = padded_block_size;
    }

    // Initialize Matrix only in process 0
    if (rank == 0) {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                matrix[i][j] = i + j;
            }
            vector[i] = i;
        }
    }

    // Broadcast matrix to all processes
    MPI_Bcast(matrix, N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Distribute vector to all processes
    MPI_Scatter(vector, N, MPI_FLOAT, vector, N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    gettimeofday(&tv1, NULL); // Time start for computation

    // Compute local result
    for (i = 0; i < block_size; i++) {
        local_result[i] = 0;
        for (j = 0; j < N; j++) {
            local_result[i] += matrix[i][j] * vector[j];
        }
    }

    gettimeofday(&tv2, NULL); // Time end for computation

    // Gather local results
    MPI_Gather(local_result, block_size, MPI_FLOAT, result, block_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    gettimeofday(&tv3, NULL); // Time end for communication

    // Print computation time for each process
    printf("Process %d computation time: %lf seconds\n", rank, (double)(tv2.tv_usec - tv1.tv_usec + (tv2.tv_sec - tv1.tv_sec) * 1e6) / 1e6);

    // Print communication time for each process
    printf("Process %d communication time: %lf seconds\n", rank, (double)(tv3.tv_usec - tv2.tv_usec + (tv3.tv_sec - tv2.tv_sec) * 1e6) / 1e6);

    if (rank == 0) {
        gettimeofday(&tv4, NULL); // Time end for total execution

        int microseconds = (tv4.tv_usec - tv1.tv_usec) + 1000000 * (tv4.tv_sec - tv1.tv_sec);

        if (DEBUG) {
            for (i = 0; i < N; i++) {
                printf(" %f \t ", result[i]);
            }
        } else {
            printf("Total execution time: %lf seconds\n", (double)microseconds / 1E6);
        }
    }

    MPI_Finalize();
    return 0;
}
