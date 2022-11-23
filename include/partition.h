#ifndef __PARTITION_H__
#define __PARTITION_H__
#include <string>
#include <iostream>
#include "params.h"
#include "kernels_cpu.h"
#include "kernels_gpu.h"
#include "kernels_tpu.h"

typedef enum _DeviceType { undefine, cpu, gpu, tpu} DeviceType;

typedef struct {
    KernelBase* kernel_base;
    DeviceType device_type;
}GenericKernel;

class PartitionRuntime{
public:
    PartitionRuntime(Params params, 
                     std::string mode, 
                     void* input, 
                     void* output);
    
    ~PartitionRuntime();
    void prepare_partitions();
    double run_partitions();
    void transform_output();

private:
    /*
        The main algorithm to determine tiling tasks to specific device(s). 
    */
    DeviceType mix_policy(unsigned int i);
    
    unsigned int block_cnt = 1;
    Params params;
    std::string mode = "cpu_p"; // partition mode, default as cpu_p
    void* input;
    void* output;
    CpuKernel** cpu_kernels;
    GpuKernel** gpu_kernels;
    TpuKernel** tpu_kernels;

    //KernelBase** kernel_base;
    GenericKernel* generic_kernels;


    std::vector<void*> input_pars;
    std::vector<void*> output_pars;
};

#endif
