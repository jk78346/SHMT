#ifndef __PARTITION_H__
#define __PARTITION_H__
#include <string>
#include <iostream>
#include <pthread.h>
#include "params.h"
#include "kernels_cpu.h"
#include "kernels_gpu.h"
#include "kernels_tpu.h"
#include "concurrentqueue.h"

typedef enum _DeviceType { undefine, cpu, gpu, tpu} DeviceType;

typedef struct {
    KernelBase* kernel_base = NULL;
    DeviceType device_type;
}GenericKernel;

/*
    structure of each SPMC FIFO node. 
 */
struct node_data{
    GenericKernel* generic_kernel;
    Params params;
    unsigned int block_id;
    unsigned int iter;
};

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
    unsigned int dev_type_cnt = 3; // cpu, gpu and tpu

private:
    /* The main algorithm to determine tiling tasks to specific device(s). */
    DeviceType mix_policy(unsigned int i);
    
    /* threading for each device type. */
    static void* RunDeviceThread(void* args);

    /* SPMC queue */
    moodycamel::ConcurrentQueue<struct node_data> q;
    
    void create_kernel_by_type(unsigned int block_id, DeviceType device_type);

    /* To determine if each tiling block is static or dynamic
        by setting the the following arrays:
        bool* is_dynamic_block
        bool* any_static_block_per_device
        bool* is_dynamic_device.
     */
    void populate_dynamic_flags();

    /*
        To indicate if the ith tiling block is under dynamic partitioning mode. 
        Dynamic block:
            a tiling block that will only be assigned to a device until 
            runtime scheduling happens such as SPMC.
        Static block:
            a tiling block that can be assigned to a device statically 
            before execution stage.
     */
    bool* is_dynamic_block;
    
    /*
        To indicate if the ith device participates in SPMC scheduling, 
        which means it's allowed to consume any dynamic tiling block.
     */
    bool* is_dynamic_device;

    unsigned int block_cnt = 1; // default no partition
    Params params;
    std::string mode = "cpu_p"; // partition mode, default as cpu_p
    void* input;
    void* output;
    CpuKernel** cpu_kernels;
    GpuKernel** gpu_kernels;
    TpuKernel** tpu_kernels;

    GenericKernel* generic_kernels;
    DeviceType* dev_sequence; // device sequence storing device types of each tiling block
    
    std::vector<void*> input_pars;
    std::vector<void*> output_pars;
};

/*
    pthread arguments
    Each thread here represents one type of device.
 */
struct thread_data{
    PartitionRuntime* p_run_ptr;
    GenericKernel* generic_kernels;
    unsigned int block_cnt;
    unsigned int iter;
    double kernel_ms;
    // The device type the thread is representing.
    DeviceType device_type;
};
#endif
