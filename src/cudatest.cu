#include "gpu_utils.h"
#include "cuda_c_mmul_shared.h"
#include "bio_utils.h"

#include <string>
#include <cuda.h>

int main() {

    /////////////////////
    // Encoding Matrix //
    /////////////////////
    // Parameters
    const int k = 64;
    const int NUM_READS = 100;

    // Choose device
    cudaSetDevice(1);

    // Training matrix size
    const long T_cols = 64;

    // Example data
    std::string r1 = "TNGGCAGCCCGCCCACGTACAGATGTTGGCGGTGAGCGCTGCGCCTTTACCGGCCCGGCCGGGCATGCTGCGGGTGTGGTGGACGGCGGTCCGGCCGCGC";
    std::string r2 = "TNGGCAGCCCGCCCACGTACAGATGTTGGCGGTGAGCGCTGCGCCTTTACCGGCCCGGCCGGGCATGCTGCGGGTGTGGTGGACGGCGGTCCGGCCGCGC";
    std::string nucs = "ACGT";
    replaceAmbigs( r1 );
    replaceAmbigs( r2 );

    // Generate example unsigned char encodings
    int kmer_count = r1.length() - k + 1;
    unsigned char *F1, *F2, *d_F1, *d_F2;

    // Pinned memory for streaming; will have to account for dynamic size at some point
    HANDLE_ERROR( cudaHostAlloc( (void**)&F1, k * kmer_count * sizeof(unsigned char), cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&F2, k * kmer_count * sizeof(unsigned char), cudaHostAllocDefault ) );

    // Device memory
    HANDLE_ERROR( cudaMalloc( (void**)&d_F1, k * kmer_count * sizeof(unsigned char) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&d_F2, k * kmer_count * sizeof(unsigned char) ) );

    ////////////////////
    // Other Matrices //
    ////////////////////
    // Allocate memory on host
    unsigned char *T, *R1, *R2, *d_T, *d_R1, *d_R2;
    T = (unsigned char*)std::malloc(k * T_cols * sizeof(unsigned char));
    HANDLE_ERROR( cudaHostAlloc( (void**)&R1, kmer_count * T_cols * sizeof(unsigned char), cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&R2, kmer_count * T_cols * sizeof(unsigned char), cudaHostAllocDefault ) );

    // Device alloc
    HANDLE_ERROR( cudaMalloc( &d_T, k * T_cols * sizeof(unsigned char) ) );
    HANDLE_ERROR( cudaMalloc( &d_R1, kmer_count * T_cols * sizeof(unsigned char) ) );
    HANDLE_ERROR( cudaMalloc( &d_R2, kmer_count * T_cols * sizeof(unsigned char) ) );

    // Fill training matrix randomly
    int idx;
    for(unsigned long i = 0; i < k * T_cols; ++i) {
        idx = std::rand() % 4;
        T[i] = (unsigned char)(nucs[idx]-'0');
        //T[i] = 1;
    }

    // Copy over the training matrix, where it will remain
    HANDLE_ERROR( cudaMemcpy(d_T, T, k * T_cols * sizeof(unsigned char), cudaMemcpyHostToDevice) );

    // We don't need T on host anymore
    //std::free(T);

    /* Usually the below would be something like this:
     * int N_THREADS_PER_BLOCK = 256; (or 16x16, 32x8 for 2D)
     * int N_BLOCKS_PER_GRID = (N + threadsPerBlock - 1) / threadsPerBlock; for 1D
     */
    std::cout << "Input dimensions:" << std::endl;
    std::cout << "Number of total elements in feature arrays: " << 2 * k * kmer_count;
    std::cout << ", with size " << (double)(2 * k * kmer_count * sizeof(unsigned char)) / 1000000 << " MB" << std::endl;
    std::cout << "Number of elements in training array: " << k * T_cols;
    std::cout << ", with size " << (double)(k * T_cols * sizeof(unsigned char)) / 1000000 << " MB" << std::endl;
    std::cout << "Number of elements in result array: " << kmer_count * T_cols;
    std::cout << ", with size " << (double)(kmer_count * T_cols * sizeof(unsigned char)) / 1000000 << " MB" << std::endl;

    dim3 dimBlock(32, 8);  // Based on comments on StackOverflow for 2D threads
    dim3 dimGrid(T_cols / dimBlock.x, kmer_count / dimBlock.y);
    std::cout << std::endl << "Grid/Block setup:" << std::endl;
    std::cout << dimGrid.x << ',' << dimGrid.y << ' ' << dimBlock.x << ',' << dimBlock.y << std::endl;

    // Set dimensions for reseting memory
    dim3 memBlock(kmer_count, 256 / kmer_count);
    dim3 memGrid(((T_cols - 1) / memBlock.x * memBlock.y) + 1);

    //////////////////
    // CUDA Streams //
    //////////////////
    // Keep track of training time
    uint64_t start, end;
    QueryPerformanceCounter(&start);

    // Check stream compatability
    cudaDeviceProp prop;
    int device_num;
    HANDLE_ERROR( cudaGetDevice( &device_num ) );
    HANDLE_ERROR( cudaGetDeviceProperties( &prop, device_num ) );

    if(!prop.deviceOverlap)
        std::cout << "Device streaming overlap disabled, therefore no speedup expected" << std::endl;

    // Initialize streams and copy event markers
    cudaStream_t stream0, stream1;
    cudaEvent_t cp0, cp1;
    HANDLE_ERROR( cudaStreamCreate( &stream0 ) );
    HANDLE_ERROR( cudaStreamCreate( &stream1 ) );
//    HANDLE_ERROR( cudaEventCreateWithFlags( &cp0, cudaEventDisableTiming | cudaEventBlockingSync ) );
//    HANDLE_ERROR( cudaEventCreateWithFlags( &cp1, cudaEventDisableTiming | cudaEventBlockingSync ) );
    HANDLE_ERROR( cudaEventCreateWithFlags( &cp0, cudaEventDisableTiming ) );
    HANDLE_ERROR( cudaEventCreateWithFlags( &cp1, cudaEventDisableTiming ) );

    // Set up pointers for zeroing memory using the driver API
    CUdeviceptr ptr1 = (CUdeviceptr) d_R1;
    CUdeviceptr ptr2 = (CUdeviceptr) d_R2;

    for(unsigned long r = 0; r < NUM_READS; ++r) {
        // Fill F1 and F2 with new data
        for (int m = 0; m < kmer_count; ++m) {
            for (int b = 0; b < k; ++b) {
                F1[(m * k) + b] = (unsigned char) r1[m + b];
                F2[(m * k) + b] = (unsigned char) r2[m + b];
//                F1[(m * k) + b] = 1;
//                F2[(m * k) + b] = 1;
            }
        }

        // Enque the memory streams in breadth-first order such that
        // the block scheduler launches kernels optimally
        HANDLE_ERROR(cudaMemcpyAsync(d_F1, F1, k * kmer_count * sizeof(unsigned char),
                                     cudaMemcpyHostToDevice, stream0));
        HANDLE_ERROR(cudaMemcpyAsync(d_F2, F2, k * kmer_count * sizeof(unsigned char),
                                     cudaMemcpyHostToDevice, stream1));

        // Insert event markers into the stream to tell host
        // when F1 and F2 are safe to overwrite in host memory
        HANDLE_ERROR( cudaEventRecord( cp0, stream0 ) );
        HANDLE_ERROR( cudaEventRecord( cp1, stream1 ) );

        // Enque the kernel launches
        MatHamm <<< dimGrid, dimBlock, 0, stream0 >>> (d_F1, d_T, d_R1, kmer_count, k, k, T_cols, kmer_count, T_cols);
        MatHamm <<< dimGrid, dimBlock, 0, stream1 >>> (d_F2, d_T, d_R2, kmer_count, k, k, T_cols, kmer_count, T_cols);

        // Enque copy back to host
        HANDLE_ERROR(cudaMemcpyAsync(R1, d_R1, kmer_count * T_cols * sizeof(unsigned char),
                                     cudaMemcpyDeviceToHost, stream0));
        HANDLE_ERROR(cudaMemcpyAsync(R2, d_R2, kmer_count * T_cols * sizeof(unsigned char),
                                     cudaMemcpyDeviceToHost, stream1));
        // Block host from proceeding until copy to GPU is complete
        HANDLE_ERROR( cudaEventSynchronize( cp0 ) );
        HANDLE_ERROR( cudaEventSynchronize( cp1 ) );

        // Clear previous values
        cuMemsetD8Async( ptr1, 0, kmer_count * T_cols, stream0);
        cuMemsetD8Async( ptr2, 0, kmer_count * T_cols, stream1);
    }

    // Synchronize to ensure work is complete
    HANDLE_ERROR( cudaStreamSynchronize( stream0 ) );
    HANDLE_ERROR( cudaStreamSynchronize( stream1 ) );

    QueryPerformanceCounter(&end);
    std::cout << "GPU pipeline took: " << double(end - start) / 1000 / kmer_count << " usec per seq" << std::endl;

    /////////////
    // Cleanup //
    /////////////
    // Free device memory
    HANDLE_ERROR( cudaFree(d_F1) );
    HANDLE_ERROR( cudaFree(d_F2) );
    HANDLE_ERROR( cudaFree(d_T) );
    HANDLE_ERROR( cudaFree(d_R1) );
    HANDLE_ERROR( cudaFree(d_R2) );
    HANDLE_ERROR( cudaStreamDestroy( stream0 ) );
    HANDLE_ERROR( cudaStreamDestroy( stream1 ) );
    HANDLE_ERROR( cudaEventDestroy( cp0 ) );
    HANDLE_ERROR( cudaEventDestroy( cp1 ) );



    // Output to test results
    write_matrix(R1, 1, T_cols);
    std::cout << std::endl;
    // Produce gold standard on CPU
    MatHammOnHost(F1, T, R1, kmer_count, k, k, T_cols, kmer_count, T_cols);
    write_matrix(R1, 1, T_cols);
    std::free(T);

    // Free host memory here if dynamically allocated
    HANDLE_ERROR( cudaFreeHost(F1) );
    HANDLE_ERROR( cudaFreeHost(F2) );
    HANDLE_ERROR( cudaFreeHost(R1) );
    HANDLE_ERROR( cudaFreeHost(R2) );

    return 0;
}
