#include "gpu_utils.h"
#include "cuda_c_mhamm_coal.h"

int main() {
    // Size of matrices
    const long A1 = 64;
    const long A2 = 64;
    const long B1 = 64;
    const long B2 = 64;
    const long C1 = 64;
    const long C2 = 64;

    // Testing matrices
    float M1[A1 * A2];
    float *M2, *M3;
    M2 = (float*)std::malloc(B1 * B2 * sizeof(float));
    M3 = (float*)std::malloc(C1 * C2 * sizeof(float));

    for(int i = 0; i < A1 * A2; ++i) {
        M1[i] = (float)1;
    }
    for(int i = 0; i < B1 * B2; ++i) {
        M2[i] = (float)1;
    }

    // Choose device
    cudaSetDevice(1);

    // Device allocation
    float* d_M1;
    HANDLE_ERROR( cudaMalloc(&d_M1, A1 * A2 * sizeof(float)) );
    float* d_M2;
    HANDLE_ERROR( cudaMalloc(&d_M2, B1 * B2 * sizeof(float)) );
    float* d_M3;
    HANDLE_ERROR( cudaMalloc(&d_M3, C1 * C2 * sizeof(float)) );

    // Copy to device
    HANDLE_ERROR( cudaMemcpy(d_M1, M1, A1 * A2 * sizeof(float), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(d_M2, M2, B1 * B2 * sizeof(float), cudaMemcpyHostToDevice) );

    /* Usually the below would be something like this:
     * int N_THREADS_PER_BLOCK = 256;
     * int N_BLOCKS_PER_GRID = (N + threadsPerBlock - 1) / threadsPerBlock;
     */
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B2 / dimBlock.x, A1 / dimBlock.y);
    std::cout << dimGrid.x << ',' << dimGrid.y << ' ' << dimBlock.x << ',' << dimBlock.y << std::endl;

    // Keep track of training time
    uint64_t start, end;
    QueryPerformanceCounter(&start);

    // Kernel launch
    // *A, *B, *C, Arows, Acols, Brows, Bcols, Crows, Ccols
    MatHammOuterCoal<<<dimGrid.x, dimBlock.x>>>(d_M1, d_M2, d_M3, A2, B2);
    cudaError_t err = cudaGetLastError();
    std::cout << err << std::endl << std::endl;

    QueryPerformanceCounter(&end);
    std::cout << "Hamming distance calculation took: " << double(end - start) / 1000 << " usec" << std::endl;

//    VectorizedOperation<<<N_BLOCKS_PER_GRID, N_THREADS_PER_BLOCK>>>(d_v1, d_v2, d_v3, N, Neq());
//    cudaError_t err = cudaGetLastError();
//    std::cout << err << std::endl << std::endl;

    // Copy back to host
    HANDLE_ERROR( cudaMemcpy(M3, d_M3, C1 * C2 * sizeof(float), cudaMemcpyDeviceToHost) );

    // Free device memory
    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_M3);

    // Output to test results
    //write_matrix(M1, A1, A2);
    //write_matrix(M2, B1, B2);
    //write_matrix(M3, C1, C2);

    // Free host memory here if dynamically allocated
    std::free(M2);
    std::free(M3);

    return 0;
}