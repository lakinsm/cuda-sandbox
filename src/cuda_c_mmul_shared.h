#include <cuda_runtime.h>

// Thread block size
#define BLOCK_SIZE 32

__global__ void MatHamm(unsigned char* A, unsigned char* B, unsigned char* C, long ARows, long ACols, long BRows, long BCols, long CRows, long CCols) {
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    unsigned char CValue = 0;
    int Row = by*blockDim.y + ty;
    int Col = bx*blockDim.x + tx;

    __shared__ unsigned char As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned char Bs[BLOCK_SIZE][BLOCK_SIZE];

    for (int k = 0; k < (BLOCK_SIZE + ACols - 1) / BLOCK_SIZE; ++k) {
        if (k*BLOCK_SIZE + tx < ACols && Row < ARows)
            As[ty][tx] = 0;
        else
            As[ty][tx] = A[(Row*ACols) + (k*BLOCK_SIZE) + tx];

        if(k*BLOCK_SIZE + ty < BRows && Col < BCols)
            Bs[ty][tx] = 0;
        else
            Bs[ty][tx] = B[((k*BLOCK_SIZE) + ty)*BCols + Col];
        __syncthreads();

        for (int n = 0; n < BLOCK_SIZE; ++n) {
            CValue += (As[ty][n] ^ Bs[n][tx]) && (As[ty][n] & Bs[n][tx] > 0);
        }
        __syncthreads();
    }

    if ((Row < CRows) && (Col < CCols))
        C[((by * blockDim.y + ty)*CCols)+(bx*blockDim.x)+tx] = CValue;
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
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    unsigned int nElement = blockDim.x * bx + tx;
    if (nElement < N)
        C[nElement] = op(A[nElement], B[nElement]);
}
