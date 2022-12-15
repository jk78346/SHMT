#include <string>
#include <stdio.h>
#include "cuda_utils.h"
#include "kernels_cpu.h"
#include "kernels_gpu.h"
#include "kernels_fft.cuh"

#include <cuda_runtime.h>
#include <cufft.h>
//#include <cuda_runtime_api.h>
//#include <cuda.h>

float fft_2d_kernel_array[7*6] = {
    13, 12, 13,  0,  1,  1,
     0,  7,  8,  2,  8,  0,
     5,  9,  1, 11, 11,  3,
    14, 14,  8, 11,  0,  3,
     6,  8, 14, 13,  0, 10,
    10, 11, 14,  1,  2,  0,
     5, 15,  7,  5,  1,  7
};
/*
    CPU convolveFFT2D, this kernel used a fixed 7x6 convolving kernel.
    Reference: samples/3_Imaging/convolutionFFT2D/convolutionFFT2D_gold.cpp
*/
void CpuKernel::fft_2d(Params params, float* input, float* output){
    float *h_Result = output;
    float *h_Data = input;
    float *h_Kernel = fft_2d_kernel_array;
    int dataH = params.get_kernel_size();
    int dataW = params.get_kernel_size();
    int kernelH = 7;
    int kernelW = 6;
    int kernelY = 3;
    int kernelX = 4;

    for (int y = 0; y < dataH; y++)
        for (int x = 0; x < dataW; x++)
        {
            double sum = 0;

            for (int ky = -(kernelH - kernelY - 1); ky <= kernelY; ky++)
                for (int kx = -(kernelW - kernelX - 1); kx <= kernelX; kx++)
                {
                    int dy = y + ky;
                    int dx = x + kx;

                    if (dy < 0) dy = 0;

                    if (dx < 0) dx = 0;

                    if (dy >= dataH) dy = dataH - 1;
 
                    if (dx >= dataW) dx = dataW - 1;
                    
                    sum += h_Data[dy * dataW + dx] * h_Kernel[(kernelY - ky) * kernelW + (kernelX - kx)];
                }

            h_Result[y * dataW + x] = (float)sum;
        }
}

void GpuKernel::fft_2d_input_conversion(Params params, float* input_array){
// ***** start to integrating fft_2d as the first integration trial *****
    const int kernelH = 7;
    const int kernelW = 6;
//            const int kernelY = 3;
//            const int kernelX = 4;
    const int   dataH = params.get_kernel_size();
    const int   dataW = params.get_kernel_size();
 
    const int fftH = snapTransformSize(dataH + kernelH - 1);
    const int fftW = snapTransformSize(dataW + kernelW - 1);
 
    float* h_Data = input_array;
    // Need to fill in the pre-determined kernel matrix for fair comparision
    float* h_Kernel = (float *)malloc(kernelH * kernelW * sizeof(float));
    float* h_ResultGPU = (float *)malloc(fftH    * fftW * sizeof(float));
    float *d_Data;
    float *d_Kernel;
    float* d_PaddedData;
    float* d_PaddedKernel;
 
    fComplex
    *d_DataSpectrum,
    *d_KernelSpectrum;
 
    cufftHandle
    fftPlanFwd,
    fftPlanInv;
 
    cudaMalloc((void **)&d_Data, dataH * dataW * sizeof(float));
    cudaMalloc((void **)&d_Kernel, kernelH * kernelW * sizeof(float));
    cudaMemcpy(d_Kernel, h_Kernel, kernelH * kernelW * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Data,   h_Data,   dataH   * dataW *   sizeof(float), cudaMemcpyHostToDevice);
 
    cudaMalloc((void **)&d_PaddedData,   fftH * fftW * sizeof(float));
    cudaMalloc((void **)&d_PaddedKernel, fftH * fftW * sizeof(float));
 
    cudaMalloc((void **)&d_DataSpectrum,   fftH * (fftW / 2 + 1) * sizeof(fComplex));
    cudaMalloc((void **)&d_KernelSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex));
    cudaMemset(d_KernelSpectrum, 0, fftH * (fftW / 2 + 1) * sizeof(fComplex));
 
    printf("...creating R2C & C2R FFT plans for %i x %i\n", fftH, fftW);
    cufftPlan2d(&fftPlanFwd, fftH, fftW, CUFFT_R2C);
    cufftPlan2d(&fftPlanInv, fftH, fftW, CUFFT_C2R);

//    padKernel(
//        d_PaddedKernel,
//        d_Kernel,
//        fftH,
//        fftW,
//        kernelH,
//        kernelW,
//        kernelY,
//        kernelX
//    );

//    fft_2d_input_conversion_wrapper();
}

/*
    GPU convolveFFT2D, this kernel used a fixed 7x6 convolving kernel.
    Reference: samples/3_Imaging/convolutionFFT2D/convolutionFFT2D.cu
*/
void GpuKernel::fft_2d(Params params, float* in_img, float* out_img){
    fft_2d_kernel_wrapper(in_img, out_img);
//    //Not including kernel transformation into time measurement,
//    //since convolution kernel is not changed very frequently
//    printf("...transforming convolution kernel\n");
//    timing kernel_fft_s = clk::now();
//    checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_PaddedKernel, (cuff    tComplex *)d_KernelSpectrum));
//    timing kernel_fft_e = clk::now();
// 
//    printf("...running GPU FFT convolution: ");
//    checkCudaErrors(cudaDeviceSynchronize());
//    sdkResetTimer(&hTimer);
//    sdkStartTimer(&hTimer);
//
//    for(int i = 0 ; i < iter ; i++){
//        checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_PaddedData, (cu    fftComplex *)d_DataSpectrum));
//        modulateAndNormalize(d_DataSpectrum, d_KernelSpectrum, fftH, fftW, 1);
//        checkCudaErrors(cufftExecC2R(fftPlanInv, (cufftComplex *)d_DataSpectrum    , (cufftReal *)d_PaddedData));
// 
//        checkCudaErrors(cudaDeviceSynchronize());
//    }
//    sdkStopTimer(&hTimer);
//    double gpuTime = sdkGetTimerValue(&hTimer)/iter;
//    printf("%f MPix/s (%f ms), averaged over %d time(s)\n", (double)dataH * (do    uble)dataW * 1e-6 / (gpuTime * 0.001), gpuTime, iter);
// 
//    printf("...reading back GPU convolution results\n");
//    checkCudaErrors(cudaMemcpy(h_ResultGPU, d_PaddedData, fftH * fftW * sizeof(    float), cudaMemcpyDeviceToHost));
}


