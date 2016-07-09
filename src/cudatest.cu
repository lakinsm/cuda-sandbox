#include "gpu_utils.h"

class Neq {
public:
    __device__ bool operator() (float a, float b) const {return a != b;}
};

template <class O> __global__
void VectorizedOperation(const float * A, const float * B, float * C, unsigned int N, O op) {
    // The index of where we start is the block size times
    // the block we are in plus the thread we're using within that block
    unsigned int nElement = blockDim.x * blockIdx.x + threadIdx.x;
    if (nElement < N)
        C[nElement] = op(A[nElement], B[nElement]);
}

int main() {
    // Size of vectors
    const int N = 6;
    size_t size = N * sizeof(float);

    // Testing vectors
    float v1[N];
    float v2[N];
    float v3[N];
    for(int i = 0; i < N; ++i) {
        v1[i] = (float)i;
        v2[i] = (float)(N - i);
    }

    // Device allocation
    float* d_v1;
    HANDLE_ERROR( cudaMalloc(&d_v1, size) );
    float* d_v2;
    HANDLE_ERROR( cudaMalloc(&d_v2, size) );
    float* d_v3;
    HANDLE_ERROR( cudaMalloc(&d_v3, size) );

    // Copy to device
    cudaMemcpy(d_v1, v1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2, v2, size, cudaMemcpyHostToDevice);

    /* Usually the below would be something like this:
     * int N_THREADS_PER_BLOCK = 256;
     * int N_BLOCKS_PER_GRID = (N + threadsPerBlock - 1) / threadsPerBlock;
     */
    int N_THREADS_PER_BLOCK = N;
    int N_BLOCKS_PER_GRID = 1;


    // Kernel launch
    VectorizedOperation<<<N_BLOCKS_PER_GRID, N_THREADS_PER_BLOCK>>>(d_v1, d_v2, d_v3, N, Neq());
    cudaError_t err = cudaGetLastError();
    std::cout << err << std::endl << std::endl;

    // Copy back to host
    HANDLE_ERROR( cudaMemcpy(v3, d_v3, size, cudaMemcpyDeviceToHost) );

    // Free device memory
    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_v3);

    // Free host memory here if dynamically allocated

    // Output to test results
    for(int i = 0; i < N; ++i) {
        std::cout << v1[i] << ' ' << v2[i] << ' ' << v3[i] << std::endl;
    }

    return 0;
}