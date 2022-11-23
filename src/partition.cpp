#include <time.h>
#include <stdlib.h>
#include "arrays.h"
#include "partition.h"

PartitionRuntime::PartitionRuntime(Params params,
                                   std::string mode,
                                   void* input,
                                   void* output){
    params.set_tiling_mode(true);
    this->params = params;
    this->block_cnt = params.get_block_cnt();
    this->mode = mode;
    this->input = input;
    this->output = output;
    this->generic_kernels = new GenericKernel[this->block_cnt];
    this->dev_sequence = new DeviceType[this->block_cnt];
    // For rand_p partition mode
    srand(time(NULL));
};

PartitionRuntime::~PartitionRuntime(){
    for(unsigned int i = 0 ; i < this->block_cnt ; i++){
        delete this->generic_kernels[i].kernel_base;
    }   
    delete this->generic_kernels;
    delete this->dev_sequence;
    this->input_pars.clear();
    this->output_pars.clear();
}

void PartitionRuntime::prepare_partitions(){
    // allocate input partitions and initialization
    array_partition_initialization(this->params,
                                   false,
                                   &(this->input),
                                   this->input_pars);

    // allocate output partitions
    array_partition_initialization(this->params,
                                   true, // skip_init
                                   &(this->output),
                                   this->output_pars);

    // assign partitions to each type of kernel handler
    for(unsigned int i = 0 ; i < this->block_cnt ; i++){
        auto device_type = this->mix_policy(i);
        if(device_type == cpu){
            this->generic_kernels[i].kernel_base =
                new CpuKernel(this->params,
                          this->input_pars[i],
                          this->output_pars[i]);
        }else if(device_type == gpu){
            this->generic_kernels[i].kernel_base =
                new GpuKernel(this->params,
                          this->input_pars[i],
                          this->output_pars[i]);
        }else if(device_type == tpu){
            this->generic_kernels[i].kernel_base =
                new TpuKernel(this->params,
                          this->input_pars[i],
                          this->output_pars[i]);
        }else{
            std::cout << __func__ << ": undefined device type: "
                      << device_type << ", program exits."
                      << std::endl;
            exit(0);
        }
        this->generic_kernels[i].kernel_base->input_conversion();
    }
}

double PartitionRuntime::run_partitions(){
    double kernel_ms = 0.0;
    for(unsigned int  i = 0 ; i < this->block_cnt ; i++){
        kernel_ms += this->generic_kernels[i].kernel_base->run_kernel();
    }   
    return kernel_ms;
}

void PartitionRuntime::transform_output(){
    for(unsigned int  i = 0 ; i < this->block_cnt ; i++){
        this->generic_kernels[i].kernel_base->output_conversion();
    }   
    output_array_partition_gathering(this->params,
                                     &(this->output),
                                     this->output_pars);
}

DeviceType PartitionRuntime::mix_policy(unsigned i
        /*index of a tiling task, no larger than this->block_cnt*/){
    DeviceType ret = undefine;
    if(this->mode == "cpu_p"){
        ret = cpu;
    }else if(this->mode == "gpu_p"){
        ret = gpu;
    }else if(this->mode == "tpu_p"){
        ret = tpu;
    }else if(this->mode == "mix_p"){ // a default mixed GPU/edgeTPU concurrent mode
        ret = (i%2 == 0)?gpu:tpu;
    }else if(this->mode == "rand_p"){ // randomly choose a device among cpu, gpu and tpu
        int idx = rand()%3;
        ret = (idx == 0)?cpu:((idx == 1)?gpu:tpu);
    }else{
        std::cout << __func__ << ": undefined partition mode: "
                  << this->mode << ", program exits."
                  << std::endl;
        exit(0);
    }   
    this->dev_sequence[i] = ret;
    return ret;
}

void PartitionRuntime::show_device_sequence(){
    std::cout << __func__ << ": ";
    for(unsigned int  i = 0 ; i < this->block_cnt ; i++){
        std::cout << this->dev_sequence[i] << " ";
    }   
    std::cout << std::endl;
}
