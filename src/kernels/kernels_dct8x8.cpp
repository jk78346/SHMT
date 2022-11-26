#include <string>
#include <stdio.h>
#include "kernels_cpu.h"
#include "kernels_gpu.h"

/*
    CPU dct8x8
    Reference: samples/3_Imaging/dct8x8/dct8x8.cu
*/
void CpuKernel::dct8x8_2d(Params params, float* input, float* output){
}

/*
    GPU dct8x8
    Reference: samples/3_Imaging/dct8x8/dct8x8.cu
*/
void GpuKernel::dct8x8_2d(Params params, float* in_img, float* out_img){
}


