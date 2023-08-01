#include <iostream>
#include "shmt.h"

void VOPS::test_func(){
    std::cout << "vops test func" << std::endl;
}

std::vector<DeviceType> VOPS::run_kernel_on_single_device(
        const std::string& mode,
        Params params,
        void* input,
        void* output){
    std::vector<DeviceType> ret;
    HLOPBase* kernel = NULL;
    if(mode == "cpu"){
        kernel = new HLOPCpu(params, input, output);
        ret.push_back(cpu);
    }else if(mode == "gpu"){
        kernel = new HLOPGpu(params, input, output);
        ret.push_back(gpu);
    }else if(mode == "tpu"){
        kernel = new HLOPTpu(params, input, output);
        ret.push_back(tpu);
    }else{
        std::cout << __func__ << ": undefined execution mode: " << mode
                  << ", execution is skipped." << std::endl;
    }
    // input array conversion from void* input
    double input_time_ms = kernel->input_conversion();
 
    // Actual kernel call
    double kernel_time_ms = kernel->run_kernel(params.iter);
 
    // output array conversion back to void* output
    double output_time_ms = kernel->output_conversion();
 
    delete kernel;
    return ret;
}

std::vector<DeviceType> VOPS::run_kernel_partition(
        const std::string& mode,
        Params params,
        void* input,
        void* output){
    PartitionRuntime* p_run = new PartitionRuntime(params,
                                                   mode,
                                                   input,
                                                   output);
    double input_time_ms = p_run->prepare_partitions();
 
    // Actual kernel call
    double kernel_time_ms = p_run->run_partitions();
 
    double output_time_ms = p_run->transform_output();
 
    //p_run->show_device_sequence();
    std::vector<DeviceType> ret = p_run->get_device_sequence();
 
    delete p_run;
    return ret;
}

std::vector<DeviceType> VOPS::run_kernel(const std::string& mode,
                                   Params& params,
                                   void* input,
                                   void* output){
    std::vector<DeviceType> ret;
    if(mode == "cpu" || mode == "gpu" || mode == "tpu"){
        ret = this->run_kernel_on_single_device(mode,
                                                params,
                                                input,
                                                output);
    }else{
        ret = this->run_kernel_partition(mode,
                                         params,
                                         input,
                                         output);
    }
    return ret;
}
