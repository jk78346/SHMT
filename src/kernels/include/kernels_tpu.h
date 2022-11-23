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
        float* input_array  = reinterpret_cast<float*>(this->input);
        this->in_size  = this->params.get_kernel_size() * params.get_kernel_size();        
        this->out_size = this->params.get_kernel_size() * params.get_kernel_size();        

        // tflite model input array initialization
        this->input_kernel  = (int*) malloc(this->in_size * sizeof(int));
        this->output_kernel = (int*) calloc(this->out_size, sizeof(int));

        // input array conversion
        for(unsigned int i = 0 ; i < this->in_size ; i++){
            this->input_kernel[i] = ((int)(input_array[i] + 128)) % 256; // float to int conversion
        }
        timing end = clk::now();
        return get_time_ms(end, start);
    }   

    /* output conversion */
    virtual double output_conversion(){
        timing start = clk::now();
        float* output_array = reinterpret_cast<float*>(this->output);
        for(unsigned int i = 0 ; i < this->out_size ; i++){
            output_array[i] = this->output_kernel[i]; // int to float conversion
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
    virtual double run_kernel(){
        timing start = clk::now();
        run_a_model(this->kernel_path,
                    this->params.iter,
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
};

#endif
