#include <iostream>
#include <opencv2/opencv.hpp>
#include "types.h"
#include "utils.h"
#include "arrays.h"
#include "params.h"
#include "quality.h"
#include "partition.h"
#include "conversion.h"
#include "kernels_cpu.h"
#include "kernels_gpu.h"
#include "kernels_tpu.h"

using namespace cv;

float run_kernel_on_single_device(const std::string& mode, Params& params, void* input, void* output){
    double kernel_ms = 0.0;
    KernelBase* kernel = NULL;
    if(mode == "cpu"){
        kernel = new CpuKernel(params, input, output);
    }else if(mode == "gpu"){
        kernel = new GpuKernel(params, input, output);
    }else if(mode == "tpu"){
        kernel = new TpuKernel(params, input, output);
    }else{
        std::cout << __func__ << ": undefined execution mode: " << mode 
                  << ", execution is skipped." << std::endl;
    }

    // input array conversion from void* input
    kernel->input_conversion();
    
    // Actual kernel call
    std::cout << mode << " kernel starts." << std::endl;
    for(int i = 0 ; i < params.iter ; i ++){
        kernel_ms += kernel->run_kernel();
    }
    std::cout << mode << " kernel ends." << std::endl;
    
    // output array conversion back to void* output
    kernel->output_conversion();

    delete kernel;
    return (float)kernel_ms;
}

float run_kernel_partition(const std::string& mode, Params params, void* input, void* output){
    double kernel_ms = 0.0;

    PartitionRuntime* p_run = new PartitionRuntime(params,
                                                   mode,
                                                   input,
                                                   output);
    p_run->prepare_partitions();
    
    // Actual kernel call
    std::cout << mode << " kernel starts." << std::endl;
    for(int i = 0 ; i < params.iter ; i ++){
        // per iteration tiling calls
        kernel_ms += p_run->run_partitions();
    }
    std::cout << mode << " kernel ends." << std::endl;

    p_run->transform_output();
    p_run->show_device_sequence();

    delete p_run;
    return (float)kernel_ms;
}

float run_kernel(const std::string& mode, Params& params, void* input, void* output){
    float kernel_ms = 0.0;
    std::cout << __func__ << ": start running kernel in " << mode << " mode" 
              << " with iter = " << params.iter << std::endl;
    if(mode == "cpu" || mode == "gpu" || mode == "tpu"){
        kernel_ms = run_kernel_on_single_device(mode, params, input, output); 
    }else{
        kernel_ms = run_kernel_partition(mode, params, input, output);
    }
    return kernel_ms;
}
    
int main(int argc, char* argv[]){
    if(argc != 7){
        std::cout << "Usage: " << argv[0] 
                  << " <application name>" // kernel's name
                  << " <problem_size>" // given problem size
                  << " <block_size>" // desired blocking size (effective only if tiling mode(s) is chosen.)
                  << " <iter>" // number of iteration
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
                  false, // default no tiling mode. can be reset anytime later
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
    std::cout << "Getting quality reseults..." << std::endl;
    
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
