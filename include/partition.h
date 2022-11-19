#ifndef __PARTITION_H__
#define __PARTITION_H__
#include <iostream>
#include "params.h"
#include "kernels_cpu.h"
#include "kernels_gpu.h"

class PartitionRuntime{
public:
    PartitionRuntime(Params params, 
                     std::string mode, 
                     void* input, 
                     void* output);
    
    ~PartitionRuntime();

    void partition_arrays();
    void init_kernel_handlers();
    void input_partition_conversion();
    double run_partitions();
    void output_partition_conversion();
    void output_summation();

private:
    unsigned int block_cnt = 1;
    Params params;
    std::string mode = "cpu_p"; // partition mode
    void* input;
    void* output;
    CpuKernel** cpu_kernels;
    GpuKernel** gpu_kernels;
    std::vector<void*> input_pars;
    std::vector<void*> output_pars;
};

#endif
