#include <string>
#include <stdio.h>
#include <kernels.h>

void CpuKernel::minimum_2d(Mat& in_img, Mat& out_img){
    out_img = in_img;
}

void CpuKernel::sobel_2d(Mat& in_img, Mat& out_img){
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    
    int ddepth = CV_32F; // CV_8U, CV_16S, CV_16U, CV_32F, CV_64F
    Sobel(in_img, grad_x, ddepth, 1, 0, 3/*ksize*/, 1/*scale*/, 0/*delta*/, BORDER_DEFAULT);
    Sobel(in_img, grad_y, ddepth, 0, 1, 3/*ksize*/, 1/*scale*/, 0/*delta*/, BORDER_DEFAULT);

    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);

    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, out_img);
}

void CpuKernel::mean_2d(Mat& in_img, Mat& out_img){
    blur(in_img, out_img, Size(3, 3), Point(-1, -1), BORDER_DEFAULT);
}

void CpuKernel::laplacian_2d(Mat& in_img, Mat& out_img){
    int ddepth = CV_32F;
    Laplacian(in_img, out_img, ddepth, 3/*kernel size*/, 1/*scale*/, 0/*delta*/, BORDER_DEFAULT);
    convertScaleAbs(out_img, out_img);
}

void CpuKernel::fft_2d(float* input, float* output){

}
