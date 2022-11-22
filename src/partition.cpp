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
    if(this->mode == "cpu_p"){
        for(unsigned int i = 0 ; i < this->block_cnt ; i++){
            delete this->cpu_kernels[i];
        }   
        delete this->cpu_kernels;
    }else if(this->mode == "gpu_p"){
        for(unsigned int i = 0 ; i < this->block_cnt ; i++){
            delete this->gpu_kernels[i];
        }   
        delete this->gpu_kernels;
    }else if(this->mode == "tpu_p"){
        for(unsigned int i = 0 ; i < this->block_cnt ; i++){
            delete this->tpu_kernels[i];
        }   
        delete this->tpu_kernels;
    }else{
        std::cout << __func__ << ": undefined partition mode: "
                  << this->mode << ", program exits."
                  << std::endl;
        exit(0);
    }
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

    // assign partitions to each kernel handler
    if(this->mode == "cpu_p"){
        this->cpu_kernels = new CpuKernel*[this->block_cnt];
        for(unsigned int i = 0 ; i < this->block_cnt ; i++){
            this->cpu_kernels[i] =
                new CpuKernel(this->params,
                              this->input_pars[i],
                              this->output_pars[i]);
            this->cpu_kernels[i]->input_conversion();
        }
    }else if(this->mode == "gpu_p"){
        this->gpu_kernels = new GpuKernel*[this->block_cnt];
        for(unsigned int i = 0 ; i < this->block_cnt ; i++){
            this->gpu_kernels[i] =
                new GpuKernel(this->params,
                              this->input_pars[i],
                              this->output_pars[i]);
            this->gpu_kernels[i]->input_conversion();
        }
    }else if(this->mode == "tpu_p"){
        this->tpu_kernels = new TpuKernel*[this->block_cnt];
        for(unsigned int i = 0 ; i < this->block_cnt ; i++){
            this->tpu_kernels[i] =
                new TpuKernel(this->params,
                              this->input_pars[i],
                              this->output_pars[i]);
            this->tpu_kernels[i]->input_conversion();
        }
    }else{
        std::cout << __func__ << ": undefined partition mode: "
                  << this->mode << ", program exits."
                  << std::endl;
        exit(0);
    }
}

double PartitionRuntime::run_partitions(){
    double kernel_ms = 0.0;
    if(this->mode == "cpu_p"){
        for(unsigned int  i = 0 ; i < this->block_cnt ; i++){
            kernel_ms += this->cpu_kernels[i]->run_kernel();
        }   
    }else if(this->mode == "gpu_p"){
        for(unsigned int  i = 0 ; i < this->block_cnt ; i++){
            kernel_ms += this->gpu_kernels[i]->run_kernel();
        }   
    }else if(this->mode == "tpu_p"){
        for(unsigned int  i = 0 ; i < this->block_cnt ; i++){
            kernel_ms += this->tpu_kernels[i]->run_kernel();
        }   
    }else{
        std::cout << __func__ << ": undefined partition mode: "
                  << this->mode << ", program exits."
                  << std::endl;
        exit(0);
    }
    return kernel_ms;
}

void PartitionRuntime::transform_output(){
    if(this->mode == "cpu_p"){
        for(unsigned int  i = 0 ; i < this->block_cnt ; i++){
            this->cpu_kernels[i]->output_conversion();
        }   
    }else if(this->mode == "gpu_p"){
        for(unsigned int  i = 0 ; i < this->block_cnt ; i++){
            this->gpu_kernels[i]->output_conversion();
        }   
    }else if(this->mode == "tpu_p"){
        for(unsigned int  i = 0 ; i < this->block_cnt ; i++){
            this->tpu_kernels[i]->output_conversion();
        }   
    }else{
        std::cout << __func__ << ": undefined partition mode: "
                  << this->mode << ", program exits."
                  << std::endl;
        exit(0);
    }   
    output_array_partition_gathering(this->params,
                                     &(this->output),
                                     this->output_pars);
}

