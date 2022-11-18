#include <iostream>
#include <opencv2/opencv.hpp>
#include "gptpu.h"
#include "types.h"
#include "utils.h"
#include "arrays.h"
#include "params.h"
#include "quality.h"
#include "conversion.h"
#include "kernels_cpu.h"
#include "kernels_gpu.h"

using namespace cv;

float run_kernel_on_cpu(Params params, void* input, void* output){
    double kernel_ms = 0.0;
    CpuKernel* cpu_kernel = new CpuKernel(params, input, output);

    // input array conversion from void* input
    cpu_kernel->input_conversion();
    
    // Actual kernel call
    printf("CPU kernel starts.\n");
    for(int i = 0 ; i < params.iter ; i ++){
        kernel_ms += cpu_kernel->run_kernel();
    }
    printf("CPU kernel ends.\n");
    
    // output array conversion back to void* output
    cpu_kernel->output_conversion();

    delete cpu_kernel;
    return (float)kernel_ms;
}

float run_kernel_on_cpu_tiling(Params params, void* input, void* output){
    double kernel_ms = 0.0;

    // arrays partition
    std::vector<void*> input_pars, output_pars;
    array_partition_initialization(params, false, &input, input_pars);
    array_partition_initialization(params, true/*skip_init*/, &output, output_pars);

    // kernel handler init
    unsigned block_cnt = params.get_block_cnt();
    CpuKernel** cpu_kernels = new CpuKernel*[block_cnt];
    for(unsigned int i = 0 ; i < block_cnt ; i++){
        cpu_kernels[i] = new CpuKernel(params, input_pars[i], output_pars[i]);
    }

    // input array conversion from void* input
    for(unsigned int i = 0 ; i < block_cnt ; i++){
        cpu_kernels[i]->input_conversion();
    }
    
    // Actual kernel call
    printf("CPU kernel starts.\n");
    for(int i = 0 ; i < params.iter ; i ++){
        // per iteration tiling calls
        for(unsigned int j = 0 ; j < block_cnt ; j++){
            kernel_ms += cpu_kernels[j]->run_kernel();
        }    
    }
    printf("CPU kernel ends.\n");
    
    // output array conversion back to void* output
    for(unsigned int i = 0 ; i < block_cnt ; i++){
        cpu_kernels[i]->output_conversion();
    }

    // output partition gathering
    output_array_partition_gathering(params, &output, output_pars);

    // clean up
    for(unsigned int i = 0 ; i < block_cnt ; i++){
        delete cpu_kernels[i];
    }
    delete cpu_kernels;
    input_pars.clear();
    output_pars.clear();
    
    return (float)kernel_ms;
}

float run_kernel_on_gpu(Params params, void* input, void* output){
    double kernel_ms = 0.0;
    GpuKernel* gpu_kernel = new GpuKernel(params, input, output);

    // input array conversion from void* input
    gpu_kernel->input_conversion();
    
    // Actual kernel call
    printf("GPU kernel starts.\n");
    for(int i = 0 ; i < params.iter ; i ++){
        kernel_ms += gpu_kernel->run_kernel();
    }
    printf("GPU kernel ends.\n");
    
    // output array conversion back to void* output
    gpu_kernel->output_conversion();

    delete gpu_kernel;
    return (float)kernel_ms;
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
    return 0.0; //TODO: integrating kernel timing as return here
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
    if(argc != 7){
        std::cout << "Usage: " << argv[0] 
                  << " <application name>"
                  << " <problem_size>"
                  << " <block_size>"
                  << " <iter>"
                  << " <baseline mode>"
                  << " <proposed mode>" 
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
    int idx = 1;
    std::string app_name = argv[idx++];
    int problem_size     = atoi(argv[idx++]);
    int block_size       = atoi(argv[idx++]);
    int iter             = atoi(argv[idx++]);
    std::string baseline_mode = argv[idx++];
    std::string proposed_mode = argv[idx++];

    Params params(app_name, 
                  problem_size, 
                  block_size, 
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
    std::cout << baseline_mode << "\t kernel time: " << baseline_kernel_ms/iter 
              << " (ms), averaged over " << iter << " time(s)." << std::endl;
    std::cout << proposed_mode << "\t kernel time: " << proposed_kernel_ms/iter
              << " (ms), averaged over " << iter << " time(s)." << std::endl;
    std::cout << baseline_mode << "\t    e2e time: " << baseline_e2e_ms << " (ms)" << std::endl;
    std::cout << proposed_mode << "\t    e2e time: " << proposed_e2e_ms << " (ms)" << std::endl;
    return 0;
}
