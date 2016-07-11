#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include <iostream>
#include <cuda_runtime.h>
#include <exception>
#include <pthread.h>
#include <sys/time.h>
#include <functional>

void MatHammOnHost(unsigned char * A, unsigned char * B, unsigned char * C, int numARows,
                       int numAColumns, int numBRows, int numBColumns,
                       int numCRows, int numCColumns)
{
    for (int i=0; i < numARows; i++)
    {
        for (int j = 0; j < numAColumns; j++)
        {
            C[i*numCColumns + j ] = 0;
            for (int k = 0; k < numCColumns; k++)
            {
                C[i*numCColumns + j ] += ((A[i*numAColumns + k] ^ B[k*numBColumns + j]) && (A[i*numAColumns + k] ^ B[k*numBColumns + j] > 0);
            }
        }
    }
    return;
}

// For performance timings
void QueryPerformanceCounter( uint64_t* val )
{
    timeval tv;
    struct timezone tz = {0, 0};
    gettimeofday( &tv, &tz );
    *val = tv.tv_sec * 1000000 + tv.tv_usec;
}

void write_matrix(const unsigned char* X, const long m, const long n) {
    for(int i = 0; i < m; ++i) {
        for(int j = 0; j < n; ++j) {
            std::cout << (int)X[(i * n) + j] << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void cuda_error(cudaError_t e, int code_line) {
    if(e != cudaSuccess) {
        std::cerr << "CUDA execution error: " << cudaGetErrorString(e) << " at line " << code_line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(e) (cuda_error(e, __LINE__))

template <typename T>
void swap (T &x, T &y) {
    T temp = x;
    x = y;
    y = temp;
}

// For thread handling
typedef pthread_t CUDAThread;
typedef void *(*CUT_THREADROUTINE)(void *);
#define CUDA_THREADPROC void
#define CUDA_THREADEND
CUDAThread cuda_start_thread(CUT_THREADROUTINE, void *data);
void cuda_end_thread(CUDAThread thread);
void cuda_destroy_thread(CUDAThread thread);
void cuda_wait_multiple_threads(const CUDAThread *threads, int n);

// Function for creation of threads
CUDAThread cuda_start_thread(CUT_THREADROUTINE f, void *data) {
    pthread_t thread;
    pthread_create(&thread, NULL, f, data);
    return thread;
}

// Function for waiting on a single thread to finish
void cuda_end_thread(CUDAThread thread) {
    pthread_join(thread, NULL);
}

// Function to destroy a single thread
void cuda_destroy_thread(CUDAThread thread) {
    pthread_cancel(thread);
}

// Function to wait for multiple threads
void cuda_wait_multiple_threads(const CUDAThread *threads, int n) {
    for(int i = 0; i < n; ++i) {
        cuda_end_thread(threads[i]);
    }
}

// Function to gather information about devices prior to memory allocation
void print_cuda_device_properties() {
    cudaDeviceProp prop;
    int count;
    HANDLE_ERROR(cudaGetDeviceCount(&count));
    for(int i = 0; i < count; ++i) {
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
        std::cout << "===== Graphics Device Information: GPU" << i << " =====    " << std::endl;
        std::cout << "Name: " << prop.name << std::endl;
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Clock rate: " << prop.clockRate << std::endl;
        std::cout << "Device copy overlap: ";
        if(prop.deviceOverlap)
            std::cout << "Enabled" << std::endl;
        else
            std::cout << "Disabled" << std::endl;
        std::cout << "Kernel execition timeout: ";
        if(prop.kernelExecTimeoutEnabled)
            std::cout << "Enabled" << std::endl << std::endl;
        else
            std::cout << "Disabled" << std::endl << std::endl;

        std::cout << "===== Graphics Device Memory Information: GPU #" << i << " =====    " << std::endl;
        std::cout << "Total global memory: " << (float)prop.totalGlobalMem / 1000000000 << " GB" << std::endl;
        std::cout << "Total constant memory: " << (float)prop.totalConstMem / 1000 << " KB" << std::endl;
        std::cout << "Max memory pitch: " << (float)prop.memPitch / 1000000000 << " GB" << std::endl;
        std::cout << "Texture alignment: " << prop.textureAlignment << std::endl << std::endl;

        std::cout << "===== Graphics Device MultiProcessor Information: GPU #" << i << " =====    " << std::endl;
        std::cout << "Multiprocessor count: " << prop.multiProcessorCount << std::endl;
        std::cout << "Shared memory per MP: " << (float)prop.sharedMemPerBlock / 1000 << " KB" << std::endl;
        std::cout << "Registers per MP: " << prop.regsPerBlock << std::endl;
        std::cout << "Threads in warp: " << prop.warpSize << std::endl;
        std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Max thread dimensions: " << prop.maxThreadsDim[0] << ' ' << prop.maxThreadsDim[1] << ' ';
        std::cout << prop.maxThreadsDim[2] << std::endl;
        std::cout << "Max grid dimensions: " << prop.maxGridSize[0] << ' ' << prop.maxGridSize[1] << ' ';
        std::cout << prop.maxGridSize[2] << std::endl << std::endl << std::endl << std::endl;
    }

}

#endif // GPU_UTILS_H