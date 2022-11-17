#include <string>
#include <stdio.h>
#include <unordered_map>
#include <opencv2/cudaarithm.hpp> // addWeighted()
#include <opencv2/cudafilters.hpp> // create[XXX]Filter()
#include "utils.h"
#include "kernels_gpu.h"

void GpuKernel::minimum_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img){
    out_img = in_img;
}

void GpuKernel::sobel_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img){

    cuda::GpuMat grad_x, grad_y;
    cuda::GpuMat abs_grad_x, abs_grad_y;

    auto sobel_dx = cuda::createSobelFilter(in_img.type(), in_img.type(), 1, 0, 3);
    auto sobel_dy = cuda::createSobelFilter(in_img.type(), in_img.type(), 0, 1, 3);

    sobel_dx->apply(in_img, grad_x);
    sobel_dy->apply(in_img, grad_y);

    cuda::abs(grad_x, grad_x);
    cuda::abs(grad_y, grad_y);

    grad_x.convertTo(grad_x, CV_8U);
    grad_y.convertTo(grad_y, CV_8U);

    cuda::addWeighted(grad_x, 0.5, grad_y, 0.5, 0, out_img);
}

void GpuKernel::mean_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img){
    auto median = cuda::createBoxFilter(in_img.type(), in_img.type(), Size(3, 3), Point(-1, -1), BORDER_DEFAULT);
    median->apply(in_img, out_img);
}

void GpuKernel::laplacian_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img){
    auto laplacian = cuda::createLaplacianFilter(in_img.type(), in_img.type(), 3/*kernel size*/, 1/*scale*/, BORDER_DEFAULT);
    laplacian->apply(in_img, out_img);

    cuda::abs(out_img, out_img);
    out_img.convertTo(out_img, CV_8U);
}

/*
    GPU convolveFFT2D, this kernel used a fixed 7x6 convolving kernel.
    Reference: samples/3_Imaging/convolutionFFT2D/convolutionFFT2D.cu
*/
void GpuKernel::fft_2d(Params params, float* in_img, float* out_img){
//    //Not including kernel transformation into time measurement,
//    //since convolution kernel is not changed very frequently
//    printf("...transforming convolution kernel\n");
//    timing kernel_fft_s = clk::now();
//    checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_PaddedKernel, (cufftComplex *)d_KernelSpectrum));
//    timing kernel_fft_e = clk::now();
// 
//    printf("...running GPU FFT convolution: ");
//    checkCudaErrors(cudaDeviceSynchronize());
//    sdkResetTimer(&hTimer);
//    sdkStartTimer(&hTimer);
//
//    for(int i = 0 ; i < iter ; i++){
//        checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_PaddedData, (cufftComplex *)d_DataSpectrum));
//        modulateAndNormalize(d_DataSpectrum, d_KernelSpectrum, fftH, fftW, 1);
//        checkCudaErrors(cufftExecC2R(fftPlanInv, (cufftComplex *)d_DataSpectrum, (cufftReal *)d_PaddedData));
// 
//        checkCudaErrors(cudaDeviceSynchronize());
//    }
//    sdkStopTimer(&hTimer);
//    double gpuTime = sdkGetTimerValue(&hTimer)/iter;
//    printf("%f MPix/s (%f ms), averaged over %d time(s)\n", (double)dataH * (double)dataW * 1e-6 / (gpuTime * 0.001), gpuTime, iter);
// 
//    printf("...reading back GPU convolution results\n");
//    checkCudaErrors(cudaMemcpy(h_ResultGPU, d_PaddedData, fftH * fftW * sizeof(float), cudaMemcpyDeviceToHost));
}




