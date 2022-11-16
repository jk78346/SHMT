#ifndef __KERNELS_H__
#define __KERNELS_H__
#include <opencv2/opencv.hpp>

using namespace cv;

// CPU kernels baseline
void minimum_2d_cpu(Mat& in_img, Mat& out_img);
void sobel_2d_cpu(Mat& in_img, Mat& out_img);
void mean_2d_cpu(Mat& in_img, Mat& out_img);
void laplacian_2d_cpu(Mat& in_img, Mat& out_img);
void fft_2d_cpu(float* input, float* output);

//GPU kernels baseline
void minimum_2d_gpu(cuda::GpuMat& in_img, cuda::GpuMat& out_img);
void sobel_2d_gpu(cuda::GpuMat& in_img, cuda::GpuMat& out_img);
void mean_2d_gpu(cuda::GpuMat& in_img, cuda::GpuMat& out_img);
void laplacian_2d_gpu(cuda::GpuMat& in_img, cuda::GpuMat& out_img);

// function pointers
typedef void (*func_ptr_float)(float*, float*);
typedef void (*func_ptr_cpu)(Mat&, Mat&);
typedef void (*func_ptr_gpu)(cuda::GpuMat&, cuda::GpuMat&);

typedef std::unordered_map<std::string, func_ptr_float> float_func_ptr_table;
typedef std::unordered_map<std::string, func_ptr_cpu>   opencv_func_ptr_table;
typedef std::unordered_map<std::string, func_ptr_gpu>   opencv_cuda_func_ptr_table;


/* 
To check if given app_name is supported as a kernel or not.
Since template is used, this definition needed to be located in this header file for successful compilation.
*/
template <typename T>
void kernel_existence_checking(std::unordered_map<std::string, T> func_table, const std::string app_name){
    if(func_table.find(app_name) == func_table.end()){
        std::cout << "app_name: " << app_name << " not found" << std::endl;
        std::cout << "supported app: " << std::endl;
        for(auto const &pair: func_table){
            std::cout << "{" << pair.first << ": " << pair.second << "}" << std::endl;
        }
        exit(0);
    }
}

#endif
