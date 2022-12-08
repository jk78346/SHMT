#ifndef __KERNELS_TPU_H__
#define __KERNELS_TPU_H__
#include <iostream>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include "gptpu.h"
#include "types.h"
#include "utils.h"
#include "params.h"
#include "kernels_base.h"

using namespace cv;

/* edgeTPU kernel class 
TODO: optimize function table searching algorithm.
*/
class TpuKernel : public KernelBase{
public:
    TpuKernel(Params params, void* input, void* output){
        this->params = params;
        this->input = input;
        this->output = output;
        this->kernel_path = get_edgetpu_kernel_path(params.app_name,
                                                    params.get_kernel_size(),
                                                    params.get_kernel_size());
    };

    virtual ~TpuKernel(){};
    
    /* input conversion */
    virtual double input_conversion(){
        timing start = clk::now();
        std::string app_name = this->params.app_name;
        this->in_size  = 
            this->params.get_kernel_size() * this->params.get_kernel_size();        
        this->out_size = 
            this->params.get_kernel_size() * this->params.get_kernel_size();        

        // tflite model input array initialization
        this->input_kernel  = (int*) malloc(this->in_size * sizeof(int));
        this->output_kernel = (int*) calloc(this->out_size, sizeof(int));
        if( std::find(this->kernel_table_uint8.begin(),
                      this->kernel_table_uint8.end(),
                      app_name) !=
            this->kernel_table_uint8.end() ){
            uint8_t* input_array  = reinterpret_cast<uint8_t*>(this->input);

            // input array conversion
            for(unsigned int i = 0 ; i < this->in_size ; i++){
                this->input_kernel[i] = ((int)(input_array[i] + 128)) % 256; // uint8_t to int conversion
            }
        }else if( std::find(this->kernel_table_fp.begin(),
                            this->kernel_table_fp.end(),
                            app_name) !=
                  this->kernel_table_fp.end() ){
            float* input_array  = reinterpret_cast<float*>(this->input);

            // input array conversion
            for(unsigned int i = 0 ; i < this->in_size ; i++){
                this->input_kernel[i] = ((int)(input_array[i] + 128)) % 256; // float to int conversion
            }
        }else{
            // app_name not found in any table. 
        }
        timing end = clk::now();
        return get_time_ms(end, start);
    }   

    /* output conversion */
    virtual double output_conversion(){
        timing start = clk::now();
        std::string app_name = this->params.app_name;
        if( std::find(this->kernel_table_uint8.begin(),
                      this->kernel_table_uint8.end(),
                      app_name) !=
            this->kernel_table_uint8.end()){
            uint8_t* output_array = reinterpret_cast<uint8_t*>(this->output);
            for(unsigned int i = 0 ; i < this->out_size ; i++){
                output_array[i] = this->output_kernel[i]; // int to uint8 conversion
            }
        }else if( std::find(this->kernel_table_fp.begin(),
                            this->kernel_table_fp.end(),
                            app_name) !=
                  this->kernel_table_fp.end() ){
            this->output = this->output_kernel; // simply forward float pointer
        }else{
            // app_name not found in any table. 
        }
        openctpu_clean_up();
        timing end = clk::now();
        return get_time_ms(end, start);
    }

    /*
        TODO: separate out all stage timings of a single run_a_model() call
        will do into input_conversion, actual kernel, output conversion.
        Now measure the e2e time as kernel time as return to align with
        partitioning runtime design. This is a must ToDo as a future performance 
        improvement.
    */
    virtual double run_kernel(unsigned int iter){
        timing start = clk::now();
        run_a_model(this->kernel_path,
                    iter,
                    this->input_kernel,
                    this->in_size,
                    this->output_kernel,
                    this->out_size,
                    1/*scale*/
                    );
        timing end = clk::now();
        return get_time_ms(end, start);
    }

private:
    // arrays
    void* input = NULL;
    void* output = NULL;
    int* input_kernel = NULL;
    int* output_kernel = NULL;
    unsigned int in_size = 0;
    unsigned int out_size = 0;
    Params params;
    std::string kernel_path;

    // kernel table
    std::vector<std::string> kernel_table_uint8 = {
        "minimal_2d",
        "sobel_2d",
        "mean_2d",
        "laplacian_2d"
    };
    std::vector<std::string> kernel_table_fp = {
        "fft_2d",
        "dct8x8_2d",
        "blackscholes_2d"
    };
};

#endif
