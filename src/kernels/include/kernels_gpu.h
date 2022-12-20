#ifndef __KERNELS_GPU_H__
#define __KERNELS_GPU_H__
#include <iostream>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include "types.h"
#include "utils.h"
#include "params.h"
#include "cuda_utils.h"
#include "kernels_base.h"

#include <cuda_runtime.h>
#include <cufft.h>

//#include <helper_functions.h>
#include <helper_cuda.h>

using namespace cv;

/* GPU kernel class 
TODO: optimize function table searching algorithm.
*/
class GpuKernel : public KernelBase{
public:
    GpuKernel(Params params, void* input, void* output){
        this->kernel_params.params = params;
        this->input_array_type.ptr = input;
        this->output_array_type.ptr = output;
//        timing start = clk::now();
        findCudaDevice(1, (const char **)NULL);
//        timing end = clk::now();
//        std::cout << __func__ << " - findCudaDevice() time: " << get_time_ms(end, start) << " (ms) " << std::endl;
    };

    virtual ~GpuKernel(){};

    /* input conversion - search over func_tables to do correct input conversion */
    virtual double input_conversion(){
        timing start = clk::now();
        std::string app_name = this->kernel_params.params.app_name;
        if(if_kernel_in_table(this->func_table_cv_cuda, app_name)){
            uint8_t* input_array  = 
                reinterpret_cast<uint8_t*>(this->input_array_type.ptr);
            array2mat(this->input_array_type.gpumat, 
                      input_array, 
                      this->kernel_params.params.get_kernel_size(), 
                      this->kernel_params.params.get_kernel_size());
        }else if(if_kernel_in_table(this->func_table_fp, app_name)){
            float* input_array  = 
                reinterpret_cast<float*>(this->input_array_type.ptr);
            float* output_array = 
                reinterpret_cast<float*>(this->output_array_type.ptr);
            this->input_array_type.host_fp  = input_array;
            this->output_array_type.host_fp = output_array; 

            // ***** integrating fft_2d conversion as the first trial *****
            if(app_name == "fft_2d"){
                this->fft_2d_input_conversion();
            }else{
                /* TODO: curently this is true for both fft and dct8x8, need to
                    see if separated input conversion functions are needed.
                 */
                   this->input_array_type.device_fp = this->input_array_type.host_fp;
                // other fp type of kernels' input conversions
            }
        }else{
            // app_name not found in any table. 
        }
        timing end = clk::now();
        return get_time_ms(end, start);
    }
    
    /* output conversion - search over func_tables to do correct output conversion */
    virtual double output_conversion(){
        timing start = clk::now();
        std::string app_name = this->kernel_params.params.app_name;
        if(if_kernel_in_table(this->func_table_cv_cuda, app_name)){
            uint8_t* output_array = 
                reinterpret_cast<uint8_t*>(this->output_array_type.ptr);
            this->output_array_type.gpumat.convertTo(
                this->output_array_type.gpumat, 
                CV_8U);
            mat2array(this->output_array_type.gpumat, output_array); // more than 95% of conversion time
        }else if(if_kernel_in_table(this->func_table_fp, app_name)){
            if(app_name == "fft_2d"){
                this->fft_2d_output_conversion(); 
            }else{
                // TODO: currently this is true for dct8x8
                this->output_array_type.host_fp = this->output_array_type.device_fp;
                // other fp type of kernels' output conversion
            }
        }else{
            // app_name not found in any table. 
        }
        timing end = clk::now();
        return get_time_ms(end, start);
    }

    virtual double run_kernel(unsigned int iter){
        std::string app_name = this->kernel_params.params.app_name;
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
        kernel_existence_checking(this->func_table_cv_cuda, this->kernel_params.params.app_name);
        timing start = clk::now();
        for(unsigned int i = 0 ; i < iter ; i++){
            this->func_table_cv_cuda[this->kernel_params.params.app_name](this->input_array_type.gpumat, 
                                                            this->output_array_type.gpumat);
        }
        timing end = clk::now();
        return get_time_ms(end, start);
    }

    /* float type of input/output */
    double run_kernel_float(unsigned int iter){
        kernel_existence_checking(this->func_table_fp, this->kernel_params.params.app_name);
        timing start = clk::now();
        for(unsigned int i = 0 ; i < iter ; i++){
            this->func_table_fp[this->kernel_params.params.app_name](this->kernel_params, 
                                                       (void**)&this->input_array_type.device_fp, 
                                                       (void**)&this->output_array_type.device_fp);
        }
        timing end = clk::now();
        return get_time_ms(end, start);
    }

    // arrays
    struct ArrayType{
        void* ptr = NULL;
        cuda::GpuMat gpumat;
        float* host_fp = NULL; // reinterpreted float pointer of void* ptr
        float* device_fp = NULL; 
    };
    ArrayType input_array_type, output_array_type;

    struct KernelParams{
        Params params;
        // other kernel related params or struct
    };
    KernelParams kernel_params;

    // function tables
    typedef void (*func_ptr_opencv_cuda)(const cuda::GpuMat, cuda::GpuMat&); // const cuda::GpuMat: input, cuda::GpuMat& : input/output
    typedef void (*func_ptr_float)(KernelParams&, void**, void**);
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
    
    // kernel-specific input/output conversion wrappers
    /*
        Note: arrays.cpp is in charge of allocating and populating h_Data of this tiling block.
        According to mat and gpumat 's way, the allocation of d_paddedData should be done within input conversion.   
        Since it is more like an internal memory object that explicitly separated out for staging measurment.
        But the necessity of the input conversion is part of the behavior of this application design.
        
        So, for general standard, kinda need to integrate d_PaddedData into ArrayType ??
        And the general name should be sth like: d_input_array
     
        Oh, it's the: float* fp and float* device_fp in ArrayType. Think of how to leverage them

        Furthermore, float* fp seems to be avoid-able. All input are from void** pointer type and
        casted into proper types such as mat, gpumat, and float* device_fp that is cuda device memory pointer
    
        Be aware of the case that input and output device_fp point to the same memory address. 
        (a.k.a need to avoid duplicate allocation and fails the program.)
     */

    void fft_2d_input_conversion();
    void fft_2d_output_conversion();

    // kernels
    static void minimum_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img);
    static void sobel_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img);
    static void mean_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img);
    static void laplacian_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img);
    static void fft_2d(KernelParams& kernel_params, void** input, void** output);
    static void dct8x8_2d(KernelParams& kernel_params, void** input, void** output);
    static void blackscholes_2d(KernelParams& kernel_params, void** input, void** output);
};
#endif
