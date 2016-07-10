#include <device_functions.h>
#include <device_launch_parameters.h>
#include <cuda_texture_types.h>
#include <texture_fetch_functions.h>

typdef texture<float4, cudaTextureType1D, cudaReadModeElementType> floatTex;

floatTex texA(0, cudaFilterModePoint, cudaAddressModeBorder);
floatTex texB(0, cudaFilterModePoint, cudaAddressModeBorder);

extern "C"
// Kernel for 256x1x1 threads per block
__global__ void __launch_bounds__(256) sgemh_kernel_128(
    float *C,
    const int m,    const int n,    const int k,
    const int lda,  const int ldb,  const int ldc,
    float alpha, int *D) {

    // Declare shared memory for kernel
    __shared__ float4 share[1024];

    int tid = threadIdx.x;

    // Indirect texture references
    floatTex tex = tid > 127 ? texB : texA;

    // Use the shared textures to prevent compiler optimizing them out
    share[tid] = tex1Dfetch(tex, tid);

    __syncthreads();

    // Again, output something to prevent optimizing out
    C[tid] = share[255-tid].x;
}

extern "C"
__global__ void __launch_bounds__(64) sgemh_kernel_64(
    float *C,
    const int m,    const int n,    const int k,
    const int lda,  const int ldb,  const int ldc,
    float alpha, int *D) {

    __shared__ float4 share[512];

    int tid = threadIdx.x;

    floatTex tex = tid > 127 ? texB : texA;

    share[tid] = tex1Dfetch(tex, tid);

    __synchthreads();

    C[tid] = share[255-tid].x;
}