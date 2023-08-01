#include <ctime>
#include <fstream>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include "shmt.h"
   
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
    
    void* input_array = NULL;
    void* output_array_baseline = NULL;
    void* output_array_proposed = NULL;

    TimeBreakDown* baseline_time_breakdown = new TimeBreakDown;
    TimeBreakDown* proposed_time_breakdown = new TimeBreakDown;

    std::vector<DeviceType> baseline_device_sequence; // typically will be gpu mode
    std::vector<DeviceType> proposed_device_sequence;

    std::vector<bool> baseline_criticality_sequence;
    std::vector<bool> proposed_criticality_sequence;

    // VOPS
    VOPS baseline_vops;
    VOPS proposed_vops;

    // create parameter instances
    Params baseline_params(app_name,
                           problem_size, 
                           block_size, 
                           false, // default no tiling mode. can be reset anytime later
                           iter,
                           testing_img_path); 
    Params proposed_params(app_name,
                           problem_size, 
                           block_size, 
                           false, // default no tiling mode. can be reset anytime later
                           iter,
                           testing_img_path);

    /* input/output array allocation and inititalization
        All arrays will be casted to corresponding data type
        depending on application.
     */
    std::cout << "data init..." << std::endl;
    data_initialization(proposed_params, 
                        &input_array,
                        &output_array_baseline,
                        &output_array_proposed);

    // Start to run baseline version of the application's implementation.
    std::cout << "run baseline... " << baseline_mode << std::endl; 
    timing baseline_start = clk::now();
    baseline_device_sequence = baseline_vops.run_kernel(baseline_mode,
                                                        baseline_params,
                                                        input_array,
                                                        output_array_baseline);
    timing baseline_end = clk::now();
    
    // Start to run proposed version of the application's implementation
    std::cout << "run experiment... " << proposed_mode << std::endl; 
    timing proposed_start = clk::now();
    proposed_device_sequence = proposed_vops.run_kernel(proposed_mode,
                                                        proposed_params,
                                                        input_array,
                                                        output_array_proposed);
    timing proposed_end = clk::now();

    // convert device sequence type 
    std::vector<int> proposed_device_type;
    for(unsigned int i = 0 ; i < proposed_device_sequence.size() ; i++){
        proposed_device_type.push_back(int(proposed_device_sequence[i]));
    }

    UnifyType* unify_input_type = 
        new UnifyType(baseline_params, input_array);
    UnifyType* unify_baseline_type = 
        new UnifyType(baseline_params, output_array_baseline);    
    UnifyType* unify_proposed_type = 
        new UnifyType(proposed_params, output_array_proposed);    

    // Get quality measurements
    std::cout << "Result evaluating..." << std::endl;
    Quality* quality = new Quality(app_name, 
                                   proposed_params.problem_size, // m
                                   proposed_params.problem_size, // n
                                   proposed_params.problem_size, // ldn
                                   proposed_params.block_size,
                                   proposed_params.block_size,
                                   unify_input_type->float_array,
                                   unify_proposed_type->float_array, 
                                   unify_baseline_type->float_array,
                                   proposed_device_type);

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

    float saliency_ratio, protected_saliency_ratio, precision;
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
