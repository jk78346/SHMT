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
    GpuKernel(Params params, void* input, void* output){
        this->params = params;
        this->input_array_type.ptr = input;
        this->output_array_type.ptr = output;
    };

    virtual ~GpuKernel(){};

    /* input conversion - search over func_tables to do correct input conversion */
    virtual double input_conversion(){
        timing start = clk::now();
        std::string app_name = params.app_name;
        if(if_kernel_in_table(this->func_table_cv_cuda, app_name)){
            uint8_t* input_array  = 
                reinterpret_cast<uint8_t*>(this->input_array_type.ptr);
            array2mat(this->input_array_type.gpumat, 
                      input_array, 
                      this->params.get_kernel_size(), 
                      this->params.get_kernel_size());
        }else if(if_kernel_in_table(this->func_table_fp, app_name)){
            float* input_array  = 
                reinterpret_cast<float*>(this->input_array_type.ptr);
            float* output_array = 
                reinterpret_cast<float*>(this->output_array_type.ptr);
            this->input_array_type.fp  = input_array;
            this->output_array_type.fp = output_array; 
        }else{
            // app_name not found in any table. 
        }
        timing end = clk::now();
        return get_time_ms(end, start);
    }
    
    /* output conversion - search over func_tables to do correct output conversion */
    virtual double output_conversion(){
        timing start = clk::now();
        std::string app_name = this->params.app_name;
        if(if_kernel_in_table(this->func_table_cv_cuda, app_name)){
            uint8_t* output_array = 
                reinterpret_cast<uint8_t*>(this->output_array_type.ptr);
            this->output_array_type.gpumat.convertTo(
                this->output_array_type.gpumat, 
                CV_8U);
            mat2array(this->output_array_type.gpumat, output_array); // more than 95% of conversion time
        }else if(if_kernel_in_table(this->func_table_fp, app_name)){
            // no need to convert from float* to float*, pass
        }else{
            // app_name not found in any table. 
        }
        timing end = clk::now();
        return get_time_ms(end, start);
    }

    virtual double run_kernel(unsigned int iter){
        std::string app_name = this->params.app_name;
        if(if_kernel_in_table(this->func_table_cv_cuda, app_name)){
            return this->run_kernel_opencv_cuda(iter);
        }else if(if_kernel_in_table(this->func_table_fp, app_name)){
            return this->run_kernel_float(iter);
        }else{
            // app_name not found in any table. 
            std::cout << __func__ << ": kernel name: " << app_name 
                      << " not found, program exists." << std::endl;
            std::exit(0);
        }
        return 0.0; // kernel execution is skipped.
    }

private:
    /* opencv type of input/output */
    double run_kernel_opencv_cuda(unsigned int iter){
        kernel_existence_checking(this->func_table_cv_cuda, this->params.app_name);
        timing start = clk::now();
        for(unsigned int i = 0 ; i < iter ; i++){
            this->func_table_cv_cuda[this->params.app_name](this->input_array_type.gpumat, 
                                                            this->output_array_type.gpumat);
        }
        timing end = clk::now();
        return get_time_ms(end, start);
    }

    /* float type of input/output */
    double run_kernel_float(unsigned int iter){
        kernel_existence_checking(this->func_table_fp, this->params.app_name);
        timing start = clk::now();
        for(unsigned int i = 0 ; i < iter ; i++){
            this->func_table_fp[this->params.app_name](this->params, 
                                                       this->input_array_type.fp, 
                                                       this->output_array_type.fp);
        }
        timing end = clk::now();
        return get_time_ms(end, start);
    }

    // arrays
    struct ArrayType{
        void* ptr = NULL;
        cuda::GpuMat gpumat;
        float* fp = NULL;
        float* device_fp = NULL; 
    };
    Params params;
    ArrayType input_array_type, output_array_type;
    
    // function tables
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
        std::make_pair<std::string, func_ptr_float> ("fft_2d", this->fft_2d),
        std::make_pair<std::string, func_ptr_float> ("dct8x8_2d", this->dct8x8_2d),
        std::make_pair<std::string, func_ptr_float> ("blackscholes_2d", this->blackscholes_2d)
    };

    // kernels
    static void minimum_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img);
    static void sobel_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img);
    static void mean_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img);
    static void laplacian_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img);
    static void fft_2d(Params params, float* input, float* output);
    static void dct8x8_2d(Params params, float* input, float* output);
    static void blackscholes_2d(Params params, float* input, float* output);
};
#endif
