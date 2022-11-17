#ifndef __KERNELS_GPU_H__
#define __KERNELS_GPU_H__
#include <iostream>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include "types.h"
#include "utils.h"
#include "params.h"
#include "kernels_base.h"

using namespace cv;

/* GPU kernel class 
TODO: optimize function table searching algorithm.
*/
class GpuKernel : public KernelBase{
public:
    virtual ~GpuKernel(){};
    /* input conversion - search over func_tables to do correct input conversion */
    virtual double input_conversion(Params params, void* input, void* output){
        timing start = clk::now();
        std::string app_name = params.app_name;
        if(if_kernel_in_table(this->func_table_cv_cuda, app_name)){
            float* input_array  = reinterpret_cast<float*>(input);
            Mat img_host;
            array2mat(img_host, input_array, CV_32F, params.problem_size, params.problem_size);
            this->input_array_type.gpumat.upload(img_host); // convert from Mat to GpuMat
        }else if(if_kernel_in_table(this->func_table_fp, app_name)){
            float* input_array  = reinterpret_cast<float*>(input);
            float* output_array = reinterpret_cast<float*>(output);
            this->input_array_type.fp  = input_array;
            this->output_array_type.fp = output_array; 
        }else{
            // app_name not found in any table. 
        }
        timing end = clk::now();
        return get_time_ms(end, start);
    }
    
    /* output conversion - search over func_tables to do correct output conversion */
    virtual double output_conversion(Params params, void* converted_output){
        timing start = clk::now();
        std::string app_name = params.app_name;
        if(if_kernel_in_table(this->func_table_cv_cuda, app_name)){
            float* output_array = reinterpret_cast<float*>(converted_output);
            Mat out_img;
            this->output_array_type.gpumat.download(out_img); // convert GpuMat to Mat
            out_img.convertTo(out_img, CV_8U);
            mat2array(out_img, output_array);
        }else if(if_kernel_in_table(this->func_table_fp, app_name)){
            // no need to convert from float* to float*, pass
        }else{
            // app_name not found in any table. 
        }
        timing end = clk::now();
        return get_time_ms(end, start);
    }

    virtual double run_kernel(const std::string app_name, Params params){
        if(if_kernel_in_table(this->func_table_cv_cuda, app_name)){
            return this->run_kernel(app_name, 
                                    this->input_array_type.gpumat, 
                                    this->output_array_type.gpumat);
        }else if(if_kernel_in_table(this->func_table_fp, app_name)){
            return this->run_kernel(app_name, 
                                    params,
                                    this->input_array_type.fp, 
                                    this->output_array_type.fp);
        }else{
            // app_name not found in any table. 
        }
        return 0.0; // kernel execution is skipped.
    }
    /* opencv type of input/output */
    double run_kernel(const std::string app_name, const cuda::GpuMat in_img, cuda::GpuMat& out_img){
        kernel_existence_checking(this->func_table_cv_cuda, app_name);
        timing start = clk::now();
        this->func_table_cv_cuda[app_name](in_img, out_img);
        timing end = clk::now();
        return get_time_ms(end, start);
    }
    /* float type of input/output */
    double run_kernel(const std::string app_name, Params params, float* input, float* output){
        kernel_existence_checking(this->func_table_fp, app_name);
        timing start = clk::now();
        this->func_table_fp[app_name](params, input, output);
        timing end = clk::now();
        return get_time_ms(end, start);
    }
private:
    struct ArrayType{
        cuda::GpuMat gpumat;
        float* fp = NULL;
    };
    ArrayType input_array_type, output_array_type;
    
    typedef void (*func_ptr_opencv_cuda)(const cuda::GpuMat, cuda::GpuMat&); // const cuda::GpuMat: input, cuda::GpuMat& : input/output
    typedef void (*func_ptr_float)(Params, float*, float*);
    typedef std::unordered_map<std::string, func_ptr_opencv_cuda> func_table_opencv_cuda;
    typedef std::unordered_map<std::string, func_ptr_float>  func_table_float;
    func_table_opencv_cuda func_table_cv_cuda = {
        std::make_pair<std::string, func_ptr_opencv_cuda> ("minimum_2d", this->minimum_2d),
        std::make_pair<std::string, func_ptr_opencv_cuda> ("sobel_2d", this->sobel_2d),
        std::make_pair<std::string, func_ptr_opencv_cuda> ("mean_2d", this->mean_2d),
        std::make_pair<std::string, func_ptr_opencv_cuda> ("laplacian_2d", this->laplacian_2d)
    };
    func_table_float func_table_fp = {
        std::make_pair<std::string, func_ptr_float> ("fft_2d", this->fft_2d)
    };

    // kernels
    static void minimum_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img);
    static void sobel_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img);
    static void mean_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img);
    static void laplacian_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img);
    static void fft_2d(Params params, float* input, float* output);
};
#endif
