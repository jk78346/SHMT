#ifndef __KERNELS_H__
#define __KERNELS_H__
#include <opencv2/opencv.hpp>

using namespace cv;

// CPU kernels baseline
void minimum_2d_cpu(Mat& in_img, Mat& out_img);
void sobel_2d_cpu(Mat& in_img, Mat& out_img);
void mean_2d_cpu(Mat& in_img, Mat& out_img);
void laplacian_2d_cpu(Mat& in_img, Mat& out_img);

//GPU kernels baseline
void minimum_2d_gpu(cuda::GpuMat& in_img, cuda::GpuMat& out_img);
void sobel_2d_gpu(cuda::GpuMat& in_img, cuda::GpuMat& out_img);
void mean_2d_gpu(cuda::GpuMat& in_img, cuda::GpuMat& out_img);
void laplacian_2d_gpu(cuda::GpuMat& in_img, cuda::GpuMat& out_img);

// function pointers
typedef void (*func_ptr_cpu)(Mat&, Mat&);
typedef void (*func_ptr_gpu)(cuda::GpuMat&, cuda::GpuMat&);

#endif
