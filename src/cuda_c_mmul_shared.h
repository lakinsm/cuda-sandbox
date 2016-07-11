#include <cuda_runtime.h>

// Tile size and vector size within that tile for the tiling
#define BLOCK_SIZE 32
#define VECTOR_SIZE 4

__global__ void MatHamm(unsigned char* A, unsigned char* B, unsigned char* C, long ACols, long BCols) {

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ unsigned char As[BLOCK_SIZE * BLOCK_SIZE];

    unsigned char cv[BLOCK_SIZE] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    // Index of the first sub-matrix of A processed by the block
    int aBegin = ACols * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + ACols - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * VECTOR_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * BCols;

    int cBegin = BCols * BLOCK_SIZE * by + VECTOR_SIZE * BLOCK_SIZE * bx;

    // TODO: Bounds checking here
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Load the matrices from device memory to shared memory
        // each thread loads one element of each matrix
        unsigned char *Ap = &A[a + ACols * ty + tx];
        unsigned char *ap = &As[ty + BLOCK_SIZE * tx];
#pragma unroll
        for(int i = 0; i < BLOCK_SIZE; i+=VECTOR_SIZE){
            ap[i] = Ap[ACols * i];
        }
        __syncthreads();

        ap = &As[0];
        unsigned char *bp = &B[b + BLOCK_SIZE * ty + tx];

        // Calculate hamming distance for each element of shared memory
#pragma unroll
        for(int i = 0; i < BLOCK_SIZE; i++){
            unsigned char bv = bp[0];
            cv[0] +=  ((ap[0] ^ bv) && (ap[0] ^ bv > 0));
            cv[1] +=  ((ap[1] ^ bv) && (ap[1] ^ bv > 0));
            cv[2] +=  ((ap[2] ^ bv) && (ap[2] ^ bv > 0));
            cv[3] +=  ((ap[3] ^ bv) && (ap[3] ^ bv > 0));
            cv[4] +=  ((ap[4] ^ bv) && (ap[4] ^ bv > 0));
            cv[5] +=  ((ap[5] ^ bv) && (ap[5] ^ bv > 0));
            cv[6] +=  ((ap[6] ^ bv) && (ap[6] ^ bv > 0));
            cv[7] +=  ((ap[7] ^ bv) && (ap[7] ^ bv > 0));
            cv[8] +=  ((ap[8] ^ bv) && (ap[8] ^ bv > 0));
            cv[9] +=  ((ap[9] ^ bv) && (ap[9] ^ bv > 0));
            cv[10] +=  ((ap[10] ^ bv) && (ap[10] ^ bv > 0));
            cv[11] +=  ((ap[11] ^ bv) && (ap[11] ^ bv > 0));
            cv[12] +=  ((ap[12] ^ bv) && (ap[12] ^ bv > 0));
            cv[13] +=  ((ap[13] ^ bv) && (ap[13] ^ bv > 0));
            cv[14] +=  ((ap[14] ^ bv) && (ap[14] ^ bv > 0));
            cv[15] +=  ((ap[15] ^ bv) && (ap[15] ^ bv > 0));
            ap += BLOCK_SIZE;
            bp += BCols;
        }
        // Synchronize to make sure the matrices are loaded
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    unsigned char *Cp = &C[cBegin];
    Cp += BLOCK_SIZE * ty + tx;
    int cStep = BCols;
#pragma unroll
    for(int i=0; i<BLOCK_SIZE; i++){
        Cp[0] = cv[i]; Cp += cStep;
    }
}

//__global__ void MatHamm(unsigned char* A, unsigned char* B, unsigned char* C, long ARows, long ACols, long BRows, long BCols, long CRows, long CCols) {
//    int tx = threadIdx.x;
//    int ty = threadIdx.y;
//    int bx = blockIdx.x;
//    int by = blockIdx.y;
//
//    unsigned char CValue = 0;
//    int Row = by*blockDim.y + ty;
//    int Col = bx*blockDim.x + tx;
//
//    __shared__ unsigned char As[BLOCK_SIZE][BLOCK_SIZE];
//    __shared__ unsigned char Bs[BLOCK_SIZE][BLOCK_SIZE];
//
//    for (int k = 0; k < (BLOCK_SIZE + ACols - 1) / BLOCK_SIZE; ++k) {
//        if (k*BLOCK_SIZE + tx < ACols && Row < ARows)
//            As[ty][tx] = 0;
//        else
//            As[ty][tx] = A[(Row*ACols) + (k*BLOCK_SIZE) + tx];
//
//        if(k*BLOCK_SIZE + ty < BRows && Col < BCols)
//            Bs[ty][tx] = 0;
//        else
//            Bs[ty][tx] = B[((k*BLOCK_SIZE) + ty)*BCols + Col];
//        __syncthreads();
//
//        for (int n = 0; n < BLOCK_SIZE; ++n) {
//            CValue += (As[ty][n] ^ Bs[n][tx]) && (As[ty][n] & Bs[n][tx] > 0);
//        }
//        __syncthreads();
//    }
//
//    if ((Row < CRows) && (Col < CCols))
//        C[((by * blockDim.y + ty)*CCols)+(bx*blockDim.x)+tx] = CValue;
//}

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
