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

struct TimeBreakDown{
    double input_time_ms;
    double kernel_time_ms;
    double output_time_ms;
};

void run_kernel_on_single_device(const std::string& mode, 
                                  Params& params, 
                                  void* input, 
                                  void* output,
                                  struct TimeBreakDown& time_breakdown){
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
    time_breakdown.input_time_ms = kernel->input_conversion();
    
    // Actual kernel call
    std::cout << mode << " mode of kernel starts." << std::endl;
    time_breakdown.kernel_time_ms = kernel->run_kernel(params.iter);
    std::cout << mode << " mode of kernel ends." << std::endl;
    
    // output array conversion back to void* output
    time_breakdown.output_time_ms = kernel->output_conversion();

    delete kernel;
}

void run_kernel_partition(const std::string& mode, 
                          Params params, 
                          void* input, 
                          void* output,
                          struct TimeBreakDown& time_breakdown){
    PartitionRuntime* p_run = new PartitionRuntime(params,
                                                   mode,
                                                   input,
                                                   output);
    time_breakdown.input_time_ms = p_run->prepare_partitions();
    
    // Actual kernel call
    std::cout << mode << " mode of kernel starts." << std::endl;
    time_breakdown.kernel_time_ms = p_run->run_partitions();
    std::cout << mode << " mode of kernel ends." << std::endl;

    time_breakdown.output_time_ms = p_run->transform_output();
    p_run->show_device_sequence();

    delete p_run;
}

void run_kernel(const std::string& mode, 
                 Params& params, 
                 void* input, 
                 void* output,
                 struct TimeBreakDown& time_breakdown){
    std::cout << __func__ << ": start running kernel in " << mode << " mode" 
              << " with iter = " << params.iter << std::endl;
    if(mode == "cpu" || mode == "gpu" || mode == "tpu"){
        run_kernel_on_single_device(mode, 
                                    params, 
                                    input, 
                                    output,
                                    time_breakdown); 
    }else{
        run_kernel_partition(mode, 
                             params, 
                             input, 
                             output,
                             time_breakdown);
    }
}
   

int main(int argc, char* argv[]){
    if(argc != 7){
        std::cout << "Usage: " << argv[0] 
                  << " <application name>" // kernel's name
                  << " <problem_size>" // given problem size
                  << " <block_size>" // desired blocking size (effective only if tiling mode(s) is chosen.)
                  << " <iter>" // number of iteration on kernel execution
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
    
    struct TimeBreakDown baseline_time_breakdown;
    struct TimeBreakDown proposed_time_breakdown;

    // Start to run baseline version of the application's implementation.
    timing baseline_start = clk::now();
    run_kernel(baseline_mode, 
               params, 
               input_array, 
               output_array_baseline,
               baseline_time_breakdown);
    timing baseline_end = clk::now();
    
    // Start to run proposed version of the application's implementation.
    timing proposed_start = clk::now();
    run_kernel(proposed_mode, 
               params, 
               input_array, 
               output_array_proposed,
               proposed_time_breakdown);
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
    
    std::cout << "=============== Latency ===============" << std::endl;
    std::cout << std::setprecision(7);
    std::cout << "        modes compared: \t" << baseline_mode << "\t" 
                                              << proposed_mode << std::endl;
    std::cout << " input conversion time: " 
              << baseline_time_breakdown.input_time_ms << " (ms), " 
              << proposed_time_breakdown.input_time_ms << " (ms)" << std::endl;
    std::cout << "           kernel time: " 
              << baseline_time_breakdown.kernel_time_ms/iter << " (ms), "
              << proposed_time_breakdown.kernel_time_ms/iter << " (ms)"
              << ", averaged over " << iter << " time(s)." << std::endl;
    std::cout << "output conversion time: " 
              << baseline_time_breakdown.output_time_ms << " (ms), " 
              << proposed_time_breakdown.output_time_ms << " (ms)" << std::endl;
    std::cout << "--------------- Summary ---------------" << std::endl;
    std::cout << "              e2e time: " << baseline_e2e_ms << " (ms), " 
                                            << proposed_e2e_ms << " (ms)" 
                                            << std::endl;
    return 0;
}
