#ifndef __PARTITION_H__
#define __PARTITION_H__
#include <iostream>
#include "params.h"
#include "kernels_cpu.h"
#include "kernels_gpu.h"
#include "kernels_tpu.h"

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
    unsigned int block_cnt = 1;
    Params params;
    std::string mode = "cpu_p"; // partition mode, default as cpu_p
    void* input;
    void* output;
    CpuKernel** cpu_kernels;
    GpuKernel** gpu_kernels;
    TpuKernel** tpu_kernels;
    std::vector<void*> input_pars;
    std::vector<void*> output_pars;
};

#endif
