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
};

PartitionRuntime::~PartitionRuntime(){
    for(unsigned int i = 0 ; i < this->block_cnt ; i++){
        delete this->cpu_kernels[i];
    }   
    delete this->cpu_kernels;
    this->input_pars.clear();
    this->output_pars.clear();
}

void PartitionRuntime::partition_arrays(){
    array_partition_initialization(this->params,
                                   false,
                                   &(this->input),
                                   this->input_pars);
    array_partition_initialization(this->params,
                                   true, // skip_init
                                   &(this->output),
                                   this->output_pars);
}

void PartitionRuntime::init_kernel_handlers(){
    if(this->mode == "cpu_p"){
        this->cpu_kernels = new CpuKernel*[this->block_cnt];
        for(unsigned int i = 0 ; i < this->block_cnt ; i++){
            this->cpu_kernels[i] =
                new CpuKernel(this->params,
                              this->input_pars[i],
                              this->output_pars[i]);
        }
    }else{
        std::cout << __func__ << ": undefined partition mode: "
                  << this->mode << ", program exits."
                  << std::endl;
        exit(0);
    }
}

void PartitionRuntime::input_partition_conversion(){
    for(unsigned int  i = 0 ; i < this->block_cnt ; i++){
        this->cpu_kernels[i]->input_conversion();
    }
}

double PartitionRuntime::run_partitions(){
    double kernel_ms = 0.0;
    for(unsigned int  i = 0 ; i < this->block_cnt ; i++){
        kernel_ms += this->cpu_kernels[i]->run_kernel();
    }   
    return kernel_ms;
}

void PartitionRuntime::output_partition_conversion(){
    for(unsigned int  i = 0 ; i < this->block_cnt ; i++){
        this->cpu_kernels[i]->output_conversion();
    }   
}

void PartitionRuntime::output_summation(){
    output_array_partition_gathering(this->params,
                                     &(this->output),
                                     this->output_pars);
}







