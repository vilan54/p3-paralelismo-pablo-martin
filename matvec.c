#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define N 1024
#define DEBUG 0

int main(int argc, char *argv[]) {
    int i, j;
    int rank, size;
    int block_size;
    int padded_N;
    struct timeval tv1, tv2, tv3, tv4;

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
    float t_communication_local, t_computation_local, t_total=0;
    float t_communication[size], t_computation[size];


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

    gettimeofday(&tv1, NULL);
    MPI_Scatter(matrix, block_size * N, MPI_FLOAT, local_matrix, block_size * N, MPI_FLOAT, 0, MPI_COMM_WORLD);    

    // Broadcast vector a todos los procesos
    MPI_Bcast(vector, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    gettimeofday(&tv2, NULL);

    t_communication_local = (tv2.tv_usec - tv1.tv_usec) + 1000000 * (tv2.tv_sec - tv1.tv_sec);

    gettimeofday(&tv3, NULL);

    if(rank == size - 1){
        block_size = N - block_size * (size - 1);
    }

    // Realizar el cálculo del producto matriz-vector localmente en cada proceso
    for(i=0;i<block_size;i++) {
        local_result[i]=0;
        for(j=0;j<N;j++) {
            local_result[i] += local_matrix[i][j]*vector[j];
        }
    }

    MPI_Gather(local_result, block_size, MPI_FLOAT, result, block_size , MPI_FLOAT, 0, MPI_COMM_WORLD);

    gettimeofday(&tv4, NULL);

    t_computation_local = (tv4.tv_usec - tv3.tv_usec) + 1000000 * (tv4.tv_sec - tv3.tv_sec);

    MPI_Gather(&t_communication_local, 1, MPI_FLOAT, t_communication, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(&t_computation_local, 1, MPI_FLOAT, t_computation, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {        
        if (DEBUG) {
            for (i = 0; i < N; i++) {
                printf(" %f \t ", result[i]);
            }
        } else {
            printf("\n%2s\t%9s\t%9s\t%9s\n", "Proc", "Comm time (s)", "Comp time (s)", "Total (s)");
            for(int i=0; i<size; i++){
                printf("%2d\t%2.6lf\t%2.6lf\t%2.6lf\n", i, (double)t_communication[i]/1E6 , (double)t_computation[i]/1E6, (double)(t_communication[i]+t_computation[i])/1E6);
                t_total += t_communication[i]+t_computation[i];
            }
            printf("\n\nTotal time: %2.6lf (s)\n", (double)t_total/1E6);
        }
    }

    MPI_Finalize();
    return 0;
}