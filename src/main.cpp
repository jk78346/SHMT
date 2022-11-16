#include <iostream>
#include <opencv2/opencv.hpp>
#include "gptpu.h"
#include "types.h"
#include "utils.h"
#include "arrays.h"
#include "params.h"
#include "quality.h"
#include "kernels.h"
#include "conversion.h"

using namespace cv;

extern std::unordered_map<std::string, func_ptr_cpu> cpu_func_table; 
extern std::unordered_map<std::string, func_ptr_gpu> gpu_func_table; 

float run_kernel_on_cpu(Params params, void* input, void* output){
    
    kernel_existence_checking(cpu_func_table, params.app_name);

    // kernel-specifc input/output data type.    
    Mat in_img, out_img;
    cpu_kernel_input_conversion(params.app_name, params, input, in_img);
    
    // Actual kernel call
    printf("CPU kernel starts.\n");
    timing start = clk::now();
    for(int i = 0 ; i < params.iter ; i ++){
        cpu_func_table[params.app_name](in_img, out_img);
    }
    timing end   = clk::now(); 
    printf("CPU kernel ends.\n");
    
    cpu_kernel_output_conversion(params.app_name, params, out_img, output);

    return get_time_ms(end, start);
}

float run_kernel_on_cpu_tiling(Params params, void* input, void* output){
    
    kernel_existence_checking(cpu_func_table, params.app_name);

    // input array partitioning
    void** input_pars;
    void** output_pars;
    input_array_partition_initialization(params, input, input_pars);    

    // input partition conversion
    Mat* in_img_pars  = new Mat[params.block_size];
    Mat* out_img_pars = new Mat[params.block_size];
    int row_cnt = params.problem_size / params.block_size; 
    int col_cnt = params.problem_size / params.block_size; 
    unsigned int block_size = params.block_size * params.block_size;
    for(int i = 0 ; i < row_cnt ; i++){
        for(int j = 0 ; j < col_cnt ; j++){
            int idx = i * col_cnt + j;
            cpu_kernel_input_conversion(params.app_name, params, input_pars[idx], in_img_pars[idx]);
        }
    }   

    // Actual kernel call
    printf("CPU tiling kernel starts.\n");
    timing start = clk::now();
    for(int iter = 0 ; iter < params.iter ; iter++){
        // per iteration calls
        for(int i = 0 ; i < row_cnt ; i++){
            for(int j = 0 ; j < col_cnt ; j++){
                int idx = i * col_cnt + j;
                std::cout << "i: " << i << ",j: " << j << ", cpu_func_table starting..." << std::endl;
                cpu_func_table[params.app_name](in_img_pars[idx], out_img_pars[idx]);
                std::cout << "end" << std::endl;
            }
        }
    }
    timing end   = clk::now(); 
    printf("CPU tiling kernel ends.\n");

    // output partition conversion    
    for(int i = 0 ; i < row_cnt ; i++){
        for(int j = 0 ; j < col_cnt ; j++){
            int idx = i * col_cnt + j;
            std::cout << "i: " << i << ",j: " << j << ": cpu_kernel_output_conversion starts..." << std::endl;
            cpu_kernel_output_conversion(params.app_name, params, out_img_pars[idx], output_pars[idx]);
            std::cout << "end" << std::endl;    
        }
    }

    // output partition gathering
    output_array_partition_gathering(params, output, output_pars);    
    std::cout << "done" << std::endl;
    return get_time_ms(end, start);
}

float run_kernel_on_gpu(Params params, void* input, void* output){
    
    kernel_existence_checking(gpu_func_table, params.app_name);

    // kernel-specifc input/output data type.    
	cuda::GpuMat in_img_gpu;
    cuda::GpuMat out_img_gpu;

    gpu_kernel_input_conversion(params.app_name, params, input, in_img_gpu);
    
    // Actual kernel call
    printf("GPU kernel starts.\n");
    timing start = clk::now();
    for(int i = 0 ; i < params.iter ; i++){
        gpu_func_table[params.app_name](in_img_gpu, out_img_gpu);
    }
    timing end   = clk::now(); 
    printf("GPU kernel ends.\n");

    gpu_kernel_output_conversion(params.app_name, params, out_img_gpu, output);

    return get_time_ms(end, start);
}

float run_kernel_on_tpu(Params params, void* input, void* output){
    float* input_array  = reinterpret_cast<float*>(input);
    float* output_array = reinterpret_cast<float*>(output);
    int in_size  = params.problem_size * params.problem_size;
    int out_size = params.problem_size * params.problem_size;
    
    // tflite model input array initialization
    int* in  = (int*) malloc(in_size * sizeof(int));
    int* out = (int*) calloc(out_size, sizeof(int));
    
    // input array conversion
    for(int i = 0 ; i < in_size ; i++){
        in[i] = ((int)(input_array[i] + 128)) % 256; // float to int conversion
    }
    std::string kernel_path = get_edgetpu_kernel_path(params.app_name, params.problem_size, params.problem_size);
    printf("TPU kernel starts.\n");
    run_a_model(kernel_path, params.iter, in, in_size, out, out_size, params.iter);
    printf("TPU kernel ends.\n");
    
    // output array conversion
    for(int i = 0 ; i < out_size ; i++){
        output_array[i] = out[i]; // int to float conversion
    }
    openctpu_clean_up();
}

float run_kernel(const std::string& mode, Params& params, void* input, void* output){
    float kernel_ms = 0.0;
    std::cout << __func__ << ": start running kernel in " << mode << " mode" 
              << " with iter = " << params.iter << std::endl;
    if(mode == "cpu"){
        kernel_ms = run_kernel_on_cpu(params, input, output);        
    }else if(mode == "cpu_p"){ // cpu partition mode
        kernel_ms = run_kernel_on_cpu_tiling(params, input, output);        
    }else if(mode == "gpu"){
	    kernel_ms = run_kernel_on_gpu(params, input, output);        
    }else if(mode == "tpu"){
	    kernel_ms = run_kernel_on_tpu(params, input, output);        
    }else{
        std::cout << "undefined execution mode: " << mode << ", execution is skipped." << std::endl;
    }
    return kernel_ms;
}

    
int main(int argc, char* argv[]){
    if(argc != 6){
        std::cout << "Usage: " << argv[0] 
                  << " <application name> <problem_size> <iter> <baseline mode> <proposed mode>" 
                  << std::endl;
        return 0;
    }else{
        // print program arguments
        for(int i = 0 ; i < argc ; i++){
            std::cout << argv[i] << " ";
        }
        std::cout << std::endl;
    }

    // program arguments assignment
    std::string app_name = argv[1];
    int problem_size     = atoi(argv[2]);
    int iter             = atoi(argv[3]);
    std::string baseline_mode = argv[4];
    std::string proposed_mode = argv[5];

    Params params(app_name, 
                  problem_size, 
                  problem_size/*block size*/, 
                  iter);

    void* input_array = NULL;
    void* output_array_baseline = NULL;
    void* output_array_proposed = NULL;

    // input/output array allocation and inititalization
    data_initialization(params, 
                        &input_array,
                        &output_array_baseline,
                        &output_array_proposed);

    float baseline_kernel_ms = 0;
    float proposed_kernel_ms = 0;
    
    // Start to run baseline version of the application's implementation.
    timing baseline_start = clk::now();
    baseline_kernel_ms = run_kernel(baseline_mode, 
                                    params, 
                                    input_array, 
                                    output_array_baseline);
    timing baseline_end = clk::now();
    
    // Start to run proposed version of the application's implementation.
    timing proposed_start = clk::now();
    proposed_kernel_ms = run_kernel(proposed_mode, 
                                    params, 
                                    input_array, 
                                    output_array_proposed);
    timing proposed_end = clk::now();
    
    // Get quality measurements
    printf("Getting quality reseults...\n");
    Quality quality(params.problem_size, // m
                    params.problem_size, // n
                    params.problem_size, // ldn
                    (float*)output_array_proposed, 
                    (float*)output_array_baseline);
    quality.print_results(1/*verbose*/);

    // Calculate end to end latency of each implementation
    double baseline_e2e_ms = get_time_ms(baseline_end, baseline_start);
    double proposed_e2e_ms = get_time_ms(proposed_end, proposed_start);
    
    std::cout << "===== Latency =====" << std::endl;
    std::cout << baseline_mode << " kernel time: " << baseline_kernel_ms/iter 
              << " (ms), averaged over " << iter << " time(s)." << std::endl;
    std::cout << proposed_mode << " kernel time: " << proposed_kernel_ms/iter
              << " (ms), averaged over " << iter << " time(s)." << std::endl;
    std::cout << baseline_mode << "    e2e time: " << baseline_e2e_ms << " (ms)" << std::endl;
    std::cout << proposed_mode << "    e2e time: " << proposed_e2e_ms << " (ms)" << std::endl;
    return 0;
}
