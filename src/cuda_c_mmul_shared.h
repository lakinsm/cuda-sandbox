#include <cuda_runtime.h>

// Thread block size
#define BLOCK_SIZE 32

__global__ void MatHamm(float* A, float* B, float* C, long ARows, long ACols, long BRows, long BCols, long CRows, long CCols) {

    float CValue = 0;
    int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
    int Col = blockIdx.x*BLOCK_SIZE + threadIdx.x;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    for (int k = 0; k < (BLOCK_SIZE + ACols - 1)/BLOCK_SIZE; k++) {
        if (k*BLOCK_SIZE + threadIdx.x < ACols && Row < ARows)
            As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*BLOCK_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0;

        if(k*BLOCK_SIZE + threadIdx.y < BRows && Col < BCols)
            Bs[threadIdx.y][threadIdx.x] = B[(k*BLOCK_SIZE + threadIdx.y)*BCols + Col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        __syncthreads();
        for (int n = 0; n < BLOCK_SIZE; ++n) {
            CValue += (As[threadIdx.y][n] != Bs[n][threadIdx.x]);
        }
        __syncthreads();
    }

    if (Row < CRows && Col < CCols)
        C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols)+(blockIdx.x*blockDim.x)+threadIdx.x]=CValue;
}

// Some other things I'm playing with
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