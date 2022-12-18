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

void GpuKernel::fft_2d_input_conversion(){
    this->input_array_type.device_fp  = this->input_array_type.host_fp;
}

void GpuKernel::fft_2d_output_conversion(){
// crop h_resultGPU[fftW x fftH] to this-<output_array_type.device_fp[dataW x dataH]
    Mat result;
    const int kernelH = 7;//this->kernel_params.cuda_kernel_args.fft_kernel_args.kernelH;
    const int kernelW = 6;//this->kernel_params.cuda_kernel_args.fft_kernel_args.kernelW;
    const int   dataH = kernel_params.params.get_kernel_size();
    const int   dataW = kernel_params.params.get_kernel_size();
    const int    fftH = snapTransformSize(dataH + kernelH - 1); //
    const int    fftW = snapTransformSize(dataW + kernelW - 1); //

    assert(this->output_array_type.device_fp != NULL);

    array2mat(result, this->output_array_type.device_fp, fftH, fftW);
    Mat cropped = result(Range(0, dataH), Range(0, dataW)); 
    mat2array(cropped, this->output_array_type.host_fp);
}    

/*
    GPU convolveFFT2D, this kernel used a fixed 7x6 convolving kernel.
    Reference: samples/3_Imaging/convolutionFFT2D/convolutionFFT2D.cu
*/
void GpuKernel::fft_2d(KernelParams& kernel_params, void** in_array, void** out_array){
// ***** start to integrating fft_2d as the first integration trial *****
    float* h_Data      = reinterpret_cast<float*>(*in_array);
    float* h_ResultGPU = reinterpret_cast<float*>(*out_array);
    
    float* h_Kernel;
    
    float* d_Data ;//        = this->kernel_params.cuda_kernel_args.fft_kernel_args.d_Data;
    float* d_PaddedData;//   = this->kernel_params.cuda_kernel_args.fft_kernel_args.d_PaddedData;
    float* d_Kernel;//       = this->kernel_params.cuda_kernel_args.fft_kernel_args.d_Kernel;
    float* d_PaddedKernel;// = this->kernel_params.cuda_kernel_args.fft_kernel_args.d_PaddedKernel;

    fComplex* d_DataSpectrum;//   = this->kernel_params.cuda_kernel_args.fft_kernel_args.d_DataSpectrum;
    fComplex* d_KernelSpectrum;// = this->kernel_params.cuda_kernel_args.fft_kernel_args.d_KernelSpectrum;

    cufftHandle fftPlanFwd, fftPlanInv;

    const int kernelH = 7;//this->kernel_params.cuda_kernel_args.fft_kernel_args.kernelH;
    const int kernelW = 6;//this->kernel_params.cuda_kernel_args.fft_kernel_args.kernelW;
    const int kernelY = 3;//this->kernel_params.cuda_kernel_args.fft_kernel_args.kernelY;
    const int kernelX = 4;//this->kernel_params.cuda_kernel_args.fft_kernel_args.kernelX;
    const int   dataH = kernel_params.params.get_kernel_size();
    const int   dataW = kernel_params.params.get_kernel_size();
    const int    fftH = snapTransformSize(dataH + kernelH - 1); //
    const int    fftW = snapTransformSize(dataW + kernelW - 1); //

    printf("...allocating memory\n");
    timing start = clk::now();
    h_Kernel = fft_2d_kernel_array;

    checkCudaErrors(cudaMalloc((void **)&(/*this->kernel_params.cuda_kernel_args.fft_kernel_args.*/d_Data), dataH * dataW * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&(/*this->kernel_params.cuda_kernel_args.fft_kernel_args.*/d_Kernel), kernelH * kernelW * sizeof(float)));
    
    checkCudaErrors(cudaMalloc((void **)&(/*this->kernel_params.cuda_kernel_args.fft_kernel_args.*/d_PaddedData),   fftH * fftW * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&(/*this->kernel_params.cuda_kernel_args.fft_kernel_args.*/d_PaddedKernel), fftH * fftW * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&(/*this->kernel_params.cuda_kernel_args.fft_kernel_args.*/d_DataSpectrum),   fftH * (fftW / 2 + 1) * sizeof(fComplex)));
    checkCudaErrors(cudaMalloc((void **)&(/*this->kernel_params.cuda_kernel_args.fft_kernel_args.*/d_KernelSpectrum), fftH * (fftW / 2 + 1) * sizeof(fComplex)));
    checkCudaErrors(cudaMemset(/*this->kernel_params.cuda_kernel_args.fft_kernel_args.*/d_KernelSpectrum, 0, fftH * (fftW / 2 + 1) * sizeof(fComplex)));
    timing end = clk::now();

    printf("...creating R2C & C2R FFT plans for %i x %i\n", fftH, fftW);
    checkCudaErrors(cufftPlan2d(&(/*this->kernel_params.cuda_kernel_args.fft_kernel_args.*/fftPlanFwd), fftH, fftW, CUFFT_R2C));
    checkCudaErrors(cufftPlan2d(&(/*this->kernel_params.cuda_kernel_args.fft_kernel_args.*/fftPlanInv), fftH, fftW, CUFFT_C2R));
    timing end2 = clk::now();

    double malloc_ms = get_time_ms(end, start);
    double cufft_ms = get_time_ms(end2, end);

    std::cout << __func__ << ": malloc time: " << malloc_ms << " (ms), cufftPlan2d time: "  << cufft_ms << " (ms)" << std::endl;
    printf("...uploading to gpu and padding convolution kernel and input data\n");
    checkCudaErrors(cudaMemcpy(/*this->kernel_params.cuda_kernel_args.fft_kernel_args.*/d_Kernel, 
                               h_Kernel, kernelH * kernelW * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(/*this->kernel_params.cuda_kernel_args.fft_kernel_args.*/d_Data,   
                               h_Data,   dataH   * dataW *   sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(/*this->kernel_params.cuda_kernel_args.fft_kernel_args.*/d_PaddedKernel, 0, fftH * fftW * sizeof(float)));
    checkCudaErrors(cudaMemset(/*this->kernel_params.cuda_kernel_args.fft_kernel_args.*/d_PaddedData,   0, fftH * fftW * sizeof(float)));

    padKernel(
        /*this->kernel_params.cuda_kernel_args.fft_kernel_args.*/d_PaddedKernel,
        /*this->kernel_params.cuda_kernel_args.fft_kernel_args.*/d_Kernel,
        fftH,
        fftW,
        kernelH,
        kernelW,
        kernelY,
        kernelX
    );

    padDataClampToBorder(
        /*this->kernel_params.cuda_kernel_args.fft_kernel_args.*/d_PaddedData,
        /*this->kernel_params.cuda_kernel_args.fft_kernel_args.*/d_Data,
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
    checkCudaErrors(cufftExecR2C(/*this->kernel_params.cuda_kernel_args.fft_kernel_args.*/fftPlanFwd, 
                 (cufftReal *)d_PaddedKernel, 
                 (cufftComplex *)d_KernelSpectrum));
    
    printf("...running GPU FFT convolution\n");
    checkCudaErrors(cudaDeviceSynchronize());
    
    checkCudaErrors(cufftExecR2C(/*this->kernel_params.cuda_kernel_args.fft_kernel_args.*/fftPlanFwd, 
                                 (cufftReal *)/*this->kernel_params.cuda_kernel_args.fft_kernel_args.*/d_PaddedData, 
                                 (cufftComplex *)/*this->kernel_params.cuda_kernel_args.fft_kernel_args.*/d_DataSpectrum));
    modulateAndNormalize(/*this->kernel_params.cuda_kernel_args.fft_kernel_args.*/d_DataSpectrum, 
                         /*this->kernel_params.cuda_kernel_args.fft_kernel_args.*/d_KernelSpectrum, fftH, fftW, 1);
    checkCudaErrors(cufftExecC2R(/*this->kernel_params.cuda_kernel_args.fft_kernel_args.*/fftPlanInv, 
                                 (cufftComplex *)/*this->kernel_params.cuda_kernel_args.fft_kernel_args.*/d_DataSpectrum, 
                                 (cufftReal *)/*this->kernel_params.cuda_kernel_args.fft_kernel_args.*/d_PaddedData));
 
    checkCudaErrors(cudaDeviceSynchronize());
    
    float* tmp = (float *)malloc(fftH    * fftW * sizeof(float));;
    
    printf("...reading back GPU convolution results\n");
    checkCudaErrors(cudaMemcpy(tmp, d_PaddedData, fftH * fftW * sizeof(float), cudaMemcpyDeviceToHost)); 
    h_ResultGPU = tmp;
    *out_array = (void*)h_ResultGPU;
}

