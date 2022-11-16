#ifndef __KERNELS_H__
#define __KERNELS_H__
#include <iostream>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include "types.h"
#include "utils.h"

using namespace cv;

/* 
A helper function to check if given app_name is supported as a kernel or not.
Since template is used, this definition needed to be located in this header 
file for successful compilation.
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

/* Base class for kernels*/
class KernelBase{
public:
    /*
        The base run_kernel() API returns kernel time in millisecond in double type.
    */
    virtual double run_kernel(){
        std::cout << "virtual kernel run is not implemented yet." << std::endl;
        exit(0);
    };
};

/* CPU kernel class */
class CpuKernel : public KernelBase{
public:
    /* opencv type of input/output */
    virtual double run_kernel(const std::string app_name, Mat& in_img, Mat& out_img){
        kernel_existence_checking(this->func_table_cv, app_name);
        timing start = clk::now();
        this->func_table_cv[app_name](in_img, out_img);
        timing end = clk::now();
        return get_time_ms(end, start);
    }
    /* float type of input/output */
    virtual float run_kernel(const std::string app_name, float* in_img, float* out_img){
        kernel_existence_checking(this->func_table_fp, app_name);
        timing start = clk::now();
        this->func_table_fp[app_name](in_img, out_img);
        timing end = clk::now();
        return get_time_ms(end, start);
    }
private:
    typedef void (*func_ptr_opencv)(Mat&, Mat&);
    typedef void (*func_ptr_float)(float*, float*);
    typedef std::unordered_map<std::string, func_ptr_opencv> func_table_opencv;
    typedef std::unordered_map<std::string, func_ptr_float>  func_table_float;
    func_table_opencv func_table_cv = {
        std::make_pair<std::string, func_ptr_opencv> ("minimum_2d", this->minimum_2d),
        std::make_pair<std::string, func_ptr_opencv> ("sobel_2d", this->sobel_2d),
        std::make_pair<std::string, func_ptr_opencv> ("mean_2d", this->mean_2d),
        std::make_pair<std::string, func_ptr_opencv> ("laplacian_2d", this->laplacian_2d)
    };
    func_table_float func_table_fp = {
        std::make_pair<std::string, func_ptr_float> ("laplacian_2d", this->fft_2d)
    };

    // kernels
    static void minimum_2d(Mat& in_img, Mat& out_img);
    static void sobel_2d(Mat& in_img, Mat& out_img);
    static void mean_2d(Mat& in_img, Mat& out_img);
    static void laplacian_2d(Mat& in_img, Mat& out_img);
    static void fft_2d(float* input, float* output);
};

//GPU kernels baseline
void minimum_2d_gpu(cuda::GpuMat& in_img, cuda::GpuMat& out_img);
void sobel_2d_gpu(cuda::GpuMat& in_img, cuda::GpuMat& out_img);
void mean_2d_gpu(cuda::GpuMat& in_img, cuda::GpuMat& out_img);
void laplacian_2d_gpu(cuda::GpuMat& in_img, cuda::GpuMat& out_img);

// function pointers
typedef void (*func_ptr_float)(float*, float*);
typedef void (*func_ptr_gpu)(cuda::GpuMat&, cuda::GpuMat&);

typedef std::unordered_map<std::string, func_ptr_gpu>   opencv_cuda_func_ptr_table;
#endif
