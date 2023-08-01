#ifndef __VOPS_H__
#define __VOPS_H__
#include <iostream>
#include <algorithm>
#include "types.h" // utils
#include "utils.h" // utils
#include "arrays.h"
#include "params.h"
#include "quality.h"
#include "hlop_cpu.h"
#include "hlop_gpu.h"
#include "hlop_tpu.h"
#include "partition.h"
#include "conversion.h"
#include "performance.h"

class VOPS{
public:
    void test_func();  
    std::vector<DeviceType> run_kernel(
            const std::string& mode,
            Params& params,
            void* input,
            void* output);
private:
    std::vector<DeviceType> run_kernel_on_single_device(
            const std::string& mode,
            Params params,
            void* input,
            void* output);
    std::vector<DeviceType> run_kernel_partition(
            const std::string& mode,
            Params params,
            void* input,
            void* output);
};
#endif
