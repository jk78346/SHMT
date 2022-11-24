#ifndef __PARTITION_H__
#define __PARTITION_H__
#include <string>
#include <iostream>
#include <pthread.h>
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
    double prepare_partitions();
    double run_partitions();
    double transform_output();
    void show_device_sequence();
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
    unsigned int dev_type_cnt = 3;
    CpuKernel** cpu_kernels;
    GpuKernel** gpu_kernels;
    TpuKernel** tpu_kernels;

    GenericKernel* generic_kernels;
    DeviceType* dev_sequence; // device sequence storing device types of each tiling block
    
    std::vector<void*> input_pars;
    std::vector<void*> output_pars;
};

#endif
