#ifndef __KERNELS_CPU_H__
#define __KERNELS_CPU_H__
#include <iostream>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include "types.h"
#include "utils.h"
#include "params.h"
#include "kernels_base.h"

using namespace cv;

/* CPU kernel class 
TODO: optimize function table searching algorithm.
*/
class CpuKernel : public KernelBase{
public:
    /* input conversion - search over func_tables to do correct input conversion */
    virtual double input_conversion(Params params, void* input, void* output){
        timing start = clk::now();
        std::string app_name = params.app_name;
        if(if_kernel_in_table(this->func_table_cv, app_name)){
            float* input_array  = reinterpret_cast<float*>(input);
            array2mat(this->input_array_type.mat, 
                      input_array, 
                      CV_32F, 
                      params.problem_size, 
                      params.problem_size);
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
        if(if_kernel_in_table(this->func_table_cv, app_name)){
            float* output_array = reinterpret_cast<float*>(converted_output);
            this->output_array_type.mat.convertTo(this->output_array_type.mat, CV_8U);
            mat2array(this->output_array_type.mat, output_array);
        }else if(if_kernel_in_table(this->func_table_fp, app_name)){
            // no need to convert from float* to float*, pass
        }else{
            // app_name not found in any table. 
        }
        timing end = clk::now();
        return get_time_ms(end, start);
    }

    virtual double run_kernel(const std::string app_name, Params params){
        if(if_kernel_in_table(this->func_table_cv, app_name)){
            return this->run_kernel(app_name, 
                                    this->input_array_type.mat, 
                                    this->output_array_type.mat);
        }else if(if_kernel_in_table(this->func_table_fp, app_name)){
            return this->run_kernel(app_name, 
                                    params, 
                                    this->input_array_type.fp, 
                                    this->output_array_type.fp);
        }else{
            // app_name not found in any table. 
        }
    }
    /* opencv type of input/output */
    double run_kernel(const std::string app_name, const Mat in_img, Mat& out_img){
        kernel_existence_checking(this->func_table_cv, app_name);
        timing start = clk::now();
        this->func_table_cv[app_name](in_img, out_img);
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
        Mat mat;
        float* fp = NULL;
    };
    ArrayType input_array_type, output_array_type;
    
    typedef void (*func_ptr_opencv)(const Mat, Mat&); // const Mat: input, Mat& : input/output
    typedef void (*func_ptr_float)(Params, float*, float*);
    typedef std::unordered_map<std::string, func_ptr_opencv> func_table_opencv;
    typedef std::unordered_map<std::string, func_ptr_float>  func_table_float;
    func_table_opencv func_table_cv = {
        std::make_pair<std::string, func_ptr_opencv> ("minimum_2d", this->minimum_2d),
        std::make_pair<std::string, func_ptr_opencv> ("sobel_2d", this->sobel_2d),
        std::make_pair<std::string, func_ptr_opencv> ("mean_2d", this->mean_2d),
        std::make_pair<std::string, func_ptr_opencv> ("laplacian_2d", this->laplacian_2d)
    };
    func_table_float func_table_fp = {
        std::make_pair<std::string, func_ptr_float> ("fft_2d", this->fft_2d)
    };

    // kernels
    static void minimum_2d(const Mat in_img, Mat& out_img);
    static void sobel_2d(const Mat in_img, Mat& out_img);
    static void mean_2d(const Mat in_img, Mat& out_img);
    static void laplacian_2d(const Mat in_img, Mat& out_img);
    static void fft_2d(Params params, float* input, float* output);
};

#endif
