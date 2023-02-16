#include <ctime>
#include <fstream>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
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
#include "performance.h"

using namespace cv;

std::vector<DeviceType> run_kernel_on_single_device(const std::string& mode, 
                                                    Params params, 
                                                    void* input, 
                                                    void* output,
                                                    TimeBreakDown* time_breakdown,
                                                    std::vector<bool>& criticality){
    std::vector<DeviceType> ret;
    KernelBase* kernel = NULL;
    if(mode == "cpu"){
        kernel = new CpuKernel(params, input, output);
        ret.push_back(cpu);
    }else if(mode == "gpu"){
        kernel = new GpuKernel(params, input, output);
        ret.push_back(gpu);
    }else if(mode == "tpu"){
        kernel = new TpuKernel(params, input, output);
        ret.push_back(tpu);
    }else{
        std::cout << __func__ << ": undefined execution mode: " << mode 
                  << ", execution is skipped." << std::endl;
    }
    criticality.push_back(true); // dummy

    // input array conversion from void* input
    time_breakdown->input_time_ms = kernel->input_conversion();
    
    // Actual kernel call
    std::cout << mode << " mode of kernel starts." << std::endl;
    time_breakdown->kernel_time_ms = kernel->run_kernel(params.iter);
    std::cout << mode << " mode of kernel ends." << std::endl;
    
    // output array conversion back to void* output
    time_breakdown->output_time_ms = kernel->output_conversion();

    delete kernel;
    return ret;
}

std::vector<DeviceType> run_kernel_partition(const std::string& mode, 
                                             Params params, 
                                             void* input, 
                                             void* output,
                                             TimeBreakDown* time_breakdown,
                                             std::vector<bool>& criticality){
    PartitionRuntime* p_run = new PartitionRuntime(params,
                                                   mode,
                                                   input,
                                                   output);
    time_breakdown->input_time_ms = p_run->prepare_partitions();
    
    // Actual kernel call
    std::cout << mode << " mode of kernel starts." << std::endl;
    time_breakdown->kernel_time_ms = p_run->run_partitions();
    std::cout << mode << " mode of kernel ends." << std::endl;

    time_breakdown->output_time_ms = p_run->transform_output();
    p_run->show_device_sequence();
    std::vector<DeviceType> ret = p_run->get_device_sequence();
    criticality = p_run->get_criticality();
    std::cout << __func__ << ": criticality seq:" << std::endl;
    for(auto c: criticality){
        std::cout << c << " ";
    }
    std::cout << std::endl;
    delete p_run;
    return ret;
}

std::vector<DeviceType> run_kernel(const std::string& mode, 
                                   Params& params, 
                                   void* input, 
                                   void* output,
                                   TimeBreakDown* time_breakdown,
                                   std::vector<bool>& criticality){
    std::cout << __func__ << ": start running kernel in " << mode << " mode" 
              << " with iter = " << params.iter << std::endl;
    std::vector<DeviceType> ret;
    if(mode == "cpu" || mode == "gpu" || mode == "tpu"){
        ret = run_kernel_on_single_device(mode, 
                                          params, 
                                          input, 
                                          output,
                                          time_breakdown,
                                          criticality); 
    }else{
        ret = run_kernel_partition(mode, 
                                   params, 
                                   input, 
                                   output,
                                   time_breakdown,
                                   criticality);
    }
    return ret;
}
   
int main(int argc, char* argv[]){
    if(argc < 7){
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
    std::string testing_img_path = 
        (argc == 8)?argv[idx++]:"../data/lena_gray_2Kx2K.bmp";
    std::string testing_img_file_name = 
        testing_img_path.substr(testing_img_path.find_last_of("/") + 1);

    Params baseline_params(app_name,
                           problem_size, 
                           block_size, 
                           false, // default no tiling mode. can be reset anytime later
                           1/*iter*/,
                           testing_img_path); 
    Params proposed_params(app_name,
                           problem_size, 
                           block_size, 
                           false, // default no tiling mode. can be reset anytime later
                           iter,
                           testing_img_path);

    void* input_array = NULL;
    void* output_array_baseline = NULL;
    void* output_array_proposed = NULL;

    /* input/output array allocation and inititalization
        All arrays will be casted to corresponding data type
        depending on application.
     */
    std::cout << __func__ << ": data init..." << std::endl;
    data_initialization(proposed_params, 
                        &input_array,
                        &output_array_baseline,
                        &output_array_proposed);
    
    TimeBreakDown* baseline_time_breakdown = new TimeBreakDown;
    TimeBreakDown* proposed_time_breakdown = new TimeBreakDown;

    std::vector<DeviceType> baseline_device_sequence; // typically will be gpu mode
    std::vector<DeviceType> proposed_device_sequence;

    std::vector<bool> baseline_criticality_sequence;
    std::vector<bool> proposed_criticality_sequence;

    // Start to run baseline version of the application's implementation.
    timing baseline_start = clk::now();
    baseline_device_sequence = run_kernel(baseline_mode, 
                               baseline_params, 
                               input_array, 
                               output_array_baseline,
                               baseline_time_breakdown,
                               baseline_criticality_sequence);
    timing baseline_end = clk::now();
    
    // Start to run proposed version of the application's implementation
    timing proposed_start = clk::now();
    proposed_device_sequence = run_kernel(proposed_mode, 
                                          proposed_params, 
                                          input_array, 
                                          output_array_proposed,
                                          proposed_time_breakdown,
                                          proposed_criticality_sequence);
    timing proposed_end = clk::now();

    // convert device sequence type 
    std::vector<int> proposed_device_type;
    for(unsigned int i = 0 ; i < proposed_device_sequence.size() ; i++){
        proposed_device_type.push_back(int(proposed_device_sequence[i]));
    }
    
    /* A quick plugin test of histogram matching on laplacian_2d */
    if(0 && app_name == "laplacian_2d"){
        histogram_matching(output_array_baseline, 
                           output_array_proposed,
                           problem_size,
                           problem_size,
                           block_size,
                           block_size,
                           proposed_device_type);
    }

    std::cout << "Converting output array to float type for quality measurement..." 
              << std::endl;
    std::cout << "[WARN] now input and output mats are assumed to have the "
              << "same params setting, " 
              << "quality result will fail if not so." << std::endl;

    UnifyType* unify_input_type = 
        new UnifyType(baseline_params, input_array);
    UnifyType* unify_baseline_type = 
        new UnifyType(baseline_params, output_array_baseline);    
    UnifyType* unify_proposed_type = 
        new UnifyType(proposed_params, output_array_proposed);    

    // Get quality measurements
    std::cout << "Getting quality results..." << std::endl;
    std::cout << __func__ << ": criticality size: " << proposed_criticality_sequence.size() << std::endl;
    Quality* quality = new Quality(proposed_params.problem_size, // m
                                   proposed_params.problem_size, // n
                                   proposed_params.problem_size, // ldn
                                   proposed_params.block_size,
                                   proposed_params.block_size,
                                   unify_input_type->float_array,
                                   unify_proposed_type->float_array, 
                                   unify_baseline_type->float_array,
                                   proposed_criticality_sequence,
                                   proposed_device_type);
    bool is_tiling = 
        (baseline_params.problem_size > baseline_params.block_size)?true:false;

    quality->print_results(is_tiling, 1/*verbose*/);
    
    /* print hist of input */
    //std::cout << __func__ << ": print hist of input array" << std::endl;
    //quality->print_histogram(unify_input_type->float_array);

    // save result arrays as image files
    const std::string path_prefix = proposed_params.app_name + "/"
                                   +std::to_string(proposed_params.problem_size) + "x"
                                   +std::to_string(proposed_params.problem_size) + "/"
                                   +proposed_params.app_name + "_" 
                                   +std::to_string(proposed_params.problem_size) + "_"
                                   +std::to_string(proposed_params.block_size) + "_"
                                   +std::to_string(proposed_params.iter) + "_"
                                   +baseline_mode + "_"
                                   +proposed_mode;
     
    // dump output images
    auto t = std::chrono::system_clock::now();
    std::time_t ts = std::chrono::system_clock::to_time_t(t);
    std::string ts_str = std::ctime(&ts);
    
    std::string cmd = "mkdir -p ../log/" + path_prefix;
    system(cmd.c_str());    

    // save as png images
    std::cout << "saving output results as images..." << std::endl;
    unify_baseline_type->save_as_img("../log/"+path_prefix+"/"+testing_img_file_name+"_"+ts_str+"_baseline.png", 
                                    baseline_params.problem_size,
                                    baseline_params.problem_size,
                                    output_array_baseline);
    unify_proposed_type->save_as_img("../log/"+path_prefix+"/"+testing_img_file_name+"_"+ts_str+"_proposed.png", 
                                    proposed_params.problem_size,
                                    proposed_params.problem_size,
                                    output_array_proposed);
    
    // save as pixel arrays
    //std::cout << "saving output results in txt files..." << std::endl;
    //unify_baseline_type->save_as_csv("../log/"+path_prefix+"/"+testing_img_file_name+"_"+ts_str+"_baseline.csv", 
    //                                baseline_params.problem_size,
    //                                baseline_params.problem_size,
    //                                output_array_baseline);
    //unify_proposed_type->save_as_csv("../log/"+path_prefix+"/"+testing_img_file_name+"_"+ts_str+"_proposed.csv", 
    //                                proposed_params.problem_size,
    //                                proposed_params.problem_size,
    //                                output_array_proposed);
    
    std::string log_file_path = "../log/" + path_prefix + "/"
                               + app_name + "_"
                               +std::to_string(problem_size) + "_"
                               +std::to_string(block_size) + "_"
                               +std::to_string(iter) + "_"
                               +baseline_mode + "_"
                               +proposed_mode + ".csv";

    // Calculate end to end latency of each implementation
    double baseline_e2e_ms = get_time_ms(baseline_end, baseline_start);
    double proposed_e2e_ms = get_time_ms(proposed_end, proposed_start);
    
    std::cout << "=============== Latency ===============" << std::endl;
    std::cout << std::setprecision(7);
    std::cout << "        modes compared: \t" << baseline_mode << "\t" 
                                              << proposed_mode << std::endl;
    std::cout << " input conversion time: " 
              << baseline_time_breakdown->input_time_ms << " (ms), " 
              << proposed_time_breakdown->input_time_ms << " (ms)" << std::endl;
    std::cout << "           kernel time: " 
              << baseline_time_breakdown->kernel_time_ms/baseline_params.iter 
              << " (ms), "
              << proposed_time_breakdown->kernel_time_ms/proposed_params.iter 
              << " (ms)"
              << ", averaged over " << proposed_params.iter 
              << " time(s)." << std::endl;
    std::cout << "output conversion time: " 
              << baseline_time_breakdown->output_time_ms << " (ms), " 
              << proposed_time_breakdown->output_time_ms << " (ms)" << std::endl;
    std::cout << "--------------- Summary ---------------" << std::endl;
    std::cout << "     averaged e2e time: " 
              << baseline_time_breakdown->get_total_time_ms(baseline_params.iter) 
              << " (ms), " 
              << proposed_time_breakdown->get_total_time_ms(proposed_params.iter) 
              << " (ms), (iteration averaged)" << std::endl;
    std::cout << "      overall e2e time: " 
              << baseline_e2e_ms << " (ms), " 
              << proposed_e2e_ms << " (ms), (iteration included)" << std::endl;
    
    
    // dump record to csv file
    std::cout << "dumping measurement results into file: " 
              << log_file_path << std::endl;

    float saliency_ratio, protected_saliency_ratio;
    quality->calc_saliency_accuracy(saliency_ratio, protected_saliency_ratio);
    
    dump_to_csv(log_file_path, 
                testing_img_file_name,
                proposed_params.app_name,
                baseline_mode,
                proposed_mode,
                proposed_params.problem_size,
                proposed_params.block_size,
                proposed_params.iter,
                quality, 
                baseline_time_breakdown,
                proposed_time_breakdown,
                proposed_device_type,
                saliency_ratio,
                protected_saliency_ratio);

    delete quality;
    delete baseline_time_breakdown;
    delete proposed_time_breakdown;
    delete unify_baseline_type;
    delete unify_proposed_type;
    return 0;
}
