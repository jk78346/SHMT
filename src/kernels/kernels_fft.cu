#include <string>
#include <stdio.h>
#include "cuda_utils.h"
#include "kernels_cpu.h"
#include "kernels_gpu.h"
#include "kernels_fft.cuh"
#include "kernels_fft_wrapper.cu"

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

void GpuKernel::fft_2d_input_conversion(KernelParams& kernel_params, float* h_Data, float* d_PaddedData){
// ***** start to integrating fft_2d as the first integration trial *****
    std::cout << __func__ << " starts" << std::endl;
    
    //StopWatchInterface *hTimer = NULL;
    //sdkCreateTimer(&hTimer);
    
    const int kernelH = 7;
    const int kernelW = 6;
    const int kernelY = 3;
    const int kernelX = 4;
    const int   dataH = kernel_params.params.get_kernel_size();
    const int   dataW = kernel_params.params.get_kernel_size();
    const int    fftH = snapTransformSize(dataH + kernelH - 1); //
    const int    fftW = snapTransformSize(dataW + kernelW - 1); //

    float* h_Kernel;
    float* d_Data;
    float* d_Kernel;
    float* d_PaddedKernel; 
 
    fComplex* d_DataSpectrum; //
    fComplex* d_KernelSpectrum; //

//    cufftHandle fftPlanFwd, fftPlanInv;

    // assign input data
    h_Kernel = fft_2d_kernel_array;

    printf("...allocating memory\n");
    timing start = clk::now();
    checkCudaErrors(cudaMalloc((void **)&d_Data, dataH * dataW * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_Kernel, kernelH * kernelW * sizeof(float)));
    
    checkCudaErrors(cudaMalloc((void **)&d_PaddedData,   fftH * fftW * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_PaddedKernel, fftH * fftW * sizeof(float)));
    
    checkCudaErrors(cudaMalloc((void **)&d_DataSpectrum,   fftH * (fftW / 2 + 1) * sizeof(fComplex)));
    checkCudaErrors(cudaMalloc((void **)&d_KernelSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex)));
    checkCudaErrors(cudaMemset(d_KernelSpectrum, 0, fftH * (fftW / 2 + 1) * sizeof(fComplex)));
    timing end = clk::now();

    printf("...creating R2C & C2R FFT plans for %i x %i\n", fftH, fftW);
    checkCudaErrors(cufftPlan2d(&(kernel_params.cuda_kernel_args.fft_kernel_args.fftPlanFwd), fftH, fftW, CUFFT_R2C));
    checkCudaErrors(cufftPlan2d(&(kernel_params.cuda_kernel_args.fft_kernel_args.fftPlanInv), fftH, fftW, CUFFT_C2R));
    timing end2 = clk::now();

    double malloc_ms = get_time_ms(end, start);
    double cufft_ms = get_time_ms(end2, end);

    std::cout << __func__ << ": malloc time: " << malloc_ms << " (ms), cufftPlan2d time: "  << cufft_ms << " (ms)" << std::endl;
    printf("...uploading to GPU and padding convolution kernel and input data\n");
    checkCudaErrors(cudaMemcpy(d_Kernel, h_Kernel, kernelH * kernelW * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Data,   h_Data,   dataH   * dataW *   sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_PaddedKernel, 0, fftH * fftW * sizeof(float)));
    checkCudaErrors(cudaMemset(d_PaddedData,   0, fftH * fftW * sizeof(float)));

    padKernel(
        d_PaddedKernel,
        d_Kernel,
        fftH,
        fftW,
        kernelH,
        kernelW,
        kernelY,
        kernelX
    );

    padDataClampToBorder(
        d_PaddedData,
        d_Data,
        fftH,
        fftW,
        dataH,
        dataW,
        kernelH,
        kernelW,
        kernelY,
        kernelX
    );

    printf("...transforming convolution kernel\n");
    checkCudaErrors(cufftExecR2C(kernel_params.cuda_kernel_args.fft_kernel_args.fftPlanFwd, 
                 (cufftReal *)d_PaddedKernel, 
                 (cufftComplex *)d_KernelSpectrum));

    kernel_params.cuda_kernel_args.fft_kernel_args.fftH             = fftH;
    kernel_params.cuda_kernel_args.fft_kernel_args.fftW             = fftW;
    kernel_params.cuda_kernel_args.fft_kernel_args.d_PaddedData     = d_PaddedData;
    kernel_params.cuda_kernel_args.fft_kernel_args.d_DataSpectrum   = d_DataSpectrum;
    kernel_params.cuda_kernel_args.fft_kernel_args.d_KernelSpectrum = d_KernelSpectrum;

//    fft_2d_input_conversion_wrapper();
}

void GpuKernel::fft_2d_output_conversion(KernelParams& kernel_params, float* h_ResultGPU, float* d_PaddedData){
    int fftH = kernel_params.cuda_kernel_args.fft_kernel_args.fftH;
    int fftW = kernel_params.cuda_kernel_args.fft_kernel_args.fftW;
    
    printf("...reading back GPU convolution results\n");
    checkCudaErrors(cudaMemcpy(h_ResultGPU, d_PaddedData, fftH * fftW * sizeof(float), cudaMemcpyDeviceToHost)); 
}    

/*
    GPU convolveFFT2D, this kernel used a fixed 7x6 convolving kernel.
    Reference: samples/3_Imaging/convolutionFFT2D/convolutionFFT2D.cu
*/
void GpuKernel::fft_2d(KernelParams& kernel_params, float* in_device_fp, float* out_device_fp){
    int         fftH             = kernel_params.cuda_kernel_args.fft_kernel_args.fftH;
    int         fftW             = kernel_params.cuda_kernel_args.fft_kernel_args.fftW;
    float*      d_PaddedData     = kernel_params.cuda_kernel_args.fft_kernel_args.d_PaddedData;
    fComplex*   d_DataSpectrum   = kernel_params.cuda_kernel_args.fft_kernel_args.d_DataSpectrum;
    fComplex*   d_KernelSpectrum = kernel_params.cuda_kernel_args.fft_kernel_args.d_KernelSpectrum;

    d_PaddedData = in_device_fp;
    
    printf("...running GPU FFT convolution\n");
    checkCudaErrors(cudaDeviceSynchronize());
    //sdkResetTimer(&hTimer);
    //sdkStartTimer(&hTimer);

    checkCudaErrors(cufftExecR2C(kernel_params.cuda_kernel_args.fft_kernel_args.fftPlanFwd, 
                                 (cufftReal *)d_PaddedData, 
                                 (cufftComplex *)d_DataSpectrum));
    modulateAndNormalize(d_DataSpectrum, d_KernelSpectrum, fftH, fftW, 1);
    checkCudaErrors(cufftExecC2R(kernel_params.cuda_kernel_args.fft_kernel_args.fftPlanInv, 
                                 (cufftComplex *)d_DataSpectrum, 
                                 (cufftReal *)d_PaddedData));
 
    checkCudaErrors(cudaDeviceSynchronize());
    //sdkStopTimer(&hTimer);
    //double gpuTime = sdkGetTimerValue(&hTimer)/iter;
    //printf("%f MPix/s (%f ms)\n", (double)dataH * (double)dataW * 1e-6 / (gpuTime * 0.001), gpuTime);
    out_device_fp = d_PaddedData;

//    fft_2d_kernel_wrapper(in_img, out_img);
}


