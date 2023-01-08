#ifndef __KERNELS_TPU_H__
#define __KERNELS_TPU_H__
#include <cassert>
#include <iostream>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include "types.h"
#include "utils.h"
#include "params.h"
#include "gptpu_utils.h"
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
        this->device_handler = new gptpu_utils::EdgeTpuHandler;
        bool verbose = true;
        this->dev_cnt = this->device_handler->list_devices(verbose); 
        for(unsigned int tpuid = 0 ; tpuid < this->dev_cnt ; tpuid++){
            this->device_handler->open_device(tpuid, verbose);
        }
    };

    virtual ~TpuKernel(){
        delete this->device_handler;
    };
   
    unsigned int get_opened_dev_cnt(){
        return this->dev_cnt;
    }

    /*
        Assign this class instance to 'tpuid' edgetpu in system.
     */
    void set_tpuid(unsigned int tpuid){
        assert(tpuid < this->dev_cnt);
        this->tpuid = tpuid;
    };

    /* input conversion */
    virtual double input_conversion(){
        timing start = clk::now();
        std::string app_name = this->params.app_name;
        this->in_size  = 
            this->params.get_kernel_size() * this->params.get_kernel_size();        
        this->out_size = 
            this->params.get_kernel_size() * this->params.get_kernel_size();        

        // tflite model input array initialization
        this->input_kernel  = (uint8_t*) malloc(this->in_size * sizeof(uint8_t));
        this->output_kernel = (uint8_t*) calloc(this->out_size, sizeof(uint8_t));
        if( std::find(this->kernel_table_uint8.begin(),
                      this->kernel_table_uint8.end(),
                      app_name) !=
            this->kernel_table_uint8.end() ){
            uint8_t* input_array  = reinterpret_cast<uint8_t*>(this->input);

            // input array conversion
            for(unsigned int i = 0 ; i < this->in_size ; i++){
                this->input_kernel[i] = ((int)(input_array[i] /*+ 128*/)) % 256; // uint8_t to int conversion
            }
            this->output_kernel = reinterpret_cast<uint8_t*>(this->output);
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
        
        this->model_id = this->device_handler->build_model(this->kernel_path);
        this->device_handler->build_interpreter(this->tpuid, this->model_id);
        this->device_handler->populate_input(this->input_kernel, this->in_size, this->model_id);

        timing end = clk::now();
        return get_time_ms(end, start);
    }   

    /* output conversion */
    virtual double output_conversion(){
        timing start = clk::now();
        std::string app_name = this->params.app_name;
        float scale;
        uint8_t zero_point;
        this->device_handler->get_raw_output(this->output_kernel, 
                                             this->out_size, 
                                             this->model_id, 
                                             zero_point,
                                             scale);
        if( std::find(this->kernel_table_uint8.begin(),
                      this->kernel_table_uint8.end(),
                      app_name) !=
            this->kernel_table_uint8.end()){
            //this->output = this->output_kernel; // uint8_t to uint8_t pointer forwarding
        }else if( std::find(this->kernel_table_fp.begin(),
                            this->kernel_table_fp.end(),
                            app_name) !=
                  this->kernel_table_fp.end() ){
            for(unsigned int i = 0 ; i < this->out_size ; i++){
                float* tmp = reinterpret_cast<float*>(this->output);
                tmp[i] = (float)( this->output_kernel[i] - zero_point ) * scale;
            }
        }else{
            // app_name not found in any table. 
        }
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
        this->device_handler->model_invoke(this->model_id, iter);
        timing end = clk::now();
        return get_time_ms(end, start);
    }

private:
    // arrays
    void* input = NULL;
    void* output = NULL;
    uint8_t* input_kernel = NULL;
    uint8_t* output_kernel = NULL;
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
    gptpu_utils::EdgeTpuHandler* device_handler;
    unsigned int dev_cnt = 0;
    unsigned int tpuid;
    unsigned int model_id;
};

#endif
