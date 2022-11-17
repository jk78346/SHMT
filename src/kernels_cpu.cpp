#include <string>
#include <stdio.h>
#include "kernels_cpu.h"

/* A dummy kernel for testing only. */
void CpuKernel::minimum_2d(const Mat in_img, Mat& out_img){
    out_img = in_img;
}

void CpuKernel::sobel_2d(const Mat in_img, Mat& out_img){
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    
    int ddepth = CV_32F; // CV_8U, CV_16S, CV_16U, CV_32F, CV_64F
    Sobel(in_img, grad_x, ddepth, 1, 0, 3/*ksize*/, 1/*scale*/, 0/*delta*/, BORDER_DEFAULT);
    Sobel(in_img, grad_y, ddepth, 0, 1, 3/*ksize*/, 1/*scale*/, 0/*delta*/, BORDER_DEFAULT);

    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);

    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, out_img);
}

void CpuKernel::mean_2d(const Mat in_img, Mat& out_img){
    blur(in_img, out_img, Size(3, 3), Point(-1, -1), BORDER_DEFAULT);
}

void CpuKernel::laplacian_2d(const Mat in_img, Mat& out_img){
    int ddepth = CV_32F;
    Laplacian(in_img, out_img, ddepth, 3/*kernel size*/, 1/*scale*/, 0/*delta*/, BORDER_DEFAULT);
    convertScaleAbs(out_img, out_img);
}

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
    int dataH = params.block_size;
    int dataW = params.block_size;
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
