#ifndef TEST_CUH__
#define TEST_CUH__

#include <stdio.h>
#define  USE_TEXTURE 1
#define POWER_OF_TWO 1
 
 
#if(USE_TEXTURE)
#define   LOAD_FLOAT(i) tex1Dfetch<float>(texFloat, i)
#define  SET_FLOAT_BASE
#else
#define  LOAD_FLOAT(i) d_Src[i]
#define SET_FLOAT_BASE
#endif

////////////////////////////////////////////////////////////////////////////////
/// Position convolution kernel center at (0, 0) in the image
////////////////////////////////////////////////////////////////////////////////
__global__ void padKernel_kernel(
    float *d_Dst,
    float *d_Src,
    int fftH,
    int fftW,
    int kernelH,
    int kernelW,
    int kernelY,
    int kernelX
#if (USE_TEXTURE)
    , cudaTextureObject_t texFloat
#endif
);

void fft_2d_input_conversion_wrapper();
void fft_2d_kernel_wrapper(float* in_img, float* out_img);

#endif
