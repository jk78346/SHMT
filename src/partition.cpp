#include <time.h>
#include <stdlib.h>
#include "arrays.h"
#include "partition.h"

#define CONCURRENT 1

struct thread_data{
    GenericKernel* generic_kernels;
    unsigned int block_cnt;
    unsigned int iter;
    double kernel_ms;
    // The device type the thread is representing.
    DeviceType device_type;
};

void *RunDeviceThread(void *my_args){
    // getting argument(s)
    struct thread_data *args = (struct thread_data*) my_args;
    GenericKernel* generic_kernels = args->generic_kernels;
    unsigned int block_cnt = args->block_cnt;
    unsigned int iter = args->iter;
    double kernel_ms = args->kernel_ms;
    DeviceType device_type = args->device_type;
    
    kernel_ms = 0.0;
    for(unsigned int  i = 0 ; i < block_cnt ; i++){
        /* Check if the device type this kernel was assigned to is the same as
           the type this consumer thread is representing. 
         */
        if(generic_kernels[i].device_type == device_type){
            kernel_ms += generic_kernels[i].kernel_base->run_kernel(iter);
        }
    }
    args->kernel_ms = kernel_ms;
    pthread_exit(NULL);
}

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

double PartitionRuntime::prepare_partitions(){
    timing start = clk::now();
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
            this->generic_kernels[i].device_type = cpu;
        }else if(device_type == gpu){
            this->generic_kernels[i].kernel_base =
                new GpuKernel(this->params,
                          this->input_pars[i],
                          this->output_pars[i]);
            this->generic_kernels[i].device_type = gpu;
        }else if(device_type == tpu){
            this->generic_kernels[i].kernel_base =
                new TpuKernel(this->params,
                          this->input_pars[i],
                          this->output_pars[i]);
            this->generic_kernels[i].device_type = tpu;
        }else{
            std::cout << __func__ << ": undefined device type: "
                      << device_type << ", program exits."
                      << std::endl;
            exit(0);
        }
        this->generic_kernels[i].kernel_base->input_conversion();
    }
    timing end = clk::now();
    return get_time_ms(end, start);
}

double PartitionRuntime::run_partitions(){
#ifdef CONCURRENT
    timing start = clk::now();
    
    //create pthreads for each device as runtime threading
    pthread_t threads[this->dev_type_cnt];
    struct thread_data td[this->dev_type_cnt];

    // CPU thread
    td[0].device_type = cpu;
    
    // GPU thread
    td[1].device_type = gpu;

    // edgeTPU thread
    td[2].device_type = tpu;
    
    // create device threads
    for(unsigned int i = 0 ; i < this->dev_type_cnt ; i++){
        td[i].generic_kernels = this->generic_kernels; 
        td[i].block_cnt = this->block_cnt;
        td[i].iter = this->params.iter;
        pthread_create(&threads[i], NULL, RunDeviceThread, (void *)&td[i]);
    }

    // wait for join
    for(unsigned int i = 0 ; i < this->dev_type_cnt ; i++){
        pthread_join(threads[i], NULL);
    }

    timing end = clk::now();
    std::cout << __func__ << ": CPU thread latency: " << td[0].kernel_ms << " (ms)" << std::endl;
    std::cout << __func__ << ": GPU thread latency: " << td[1].kernel_ms << " (ms)" << std::endl;
    std::cout << __func__ << ": TPU thread latency: " << td[2].kernel_ms << " (ms)" << std::endl;
    double e2e_kernel_ms = get_time_ms(end, start);
    std::cout << __func__ << ": e2e kernel time: " << e2e_kernel_ms << " (ms) (pthread overhead included)" << std::endl;
    return e2e_kernel_ms;
#else
    double kernel_ms = 0.0;
    for(unsigned int  i = 0 ; i < this->block_cnt ; i++){
        kernel_ms += this->generic_kernels[i].kernel_base->run_kernel(iter);
    }   
    return kernel_ms;
#endif
}

double PartitionRuntime::transform_output(){
    timing start = clk::now();
    for(unsigned int  i = 0 ; i < this->block_cnt ; i++){
        this->generic_kernels[i].kernel_base->output_conversion();
    }   
    output_array_partition_gathering(this->params,
                                     &(this->output),
                                     this->output_pars);
    timing end = clk::now();
    return get_time_ms(end, start);
}

DeviceType PartitionRuntime::mix_policy(unsigned i
        /*index of a tiling task, no larger than this->block_cnt*/){
    DeviceType ret = undefine;
    if(this->mode == "cpu_p"){ // all partitions on cpu
        ret = cpu;
    }else if(this->mode == "gpu_p"){ // all partitions on gpu
        ret = gpu;
    }else if(this->mode == "tpu_p"){ // all partitions on tpu
        ret = tpu;
    }else if(this->mode == "alls_p"){ // sequentially choose a device between cpu, gpu and tpu
        int idx = i%3;
        ret = (idx == 0)?cpu:((idx == 1)?gpu:tpu);
    }else if(this->mode == "cgs_p"){ // sequentially choose between cpu and gpu
        ret = (i%2 == 0)?cpu:gpu;
    }else if(this->mode == "gts_p"){ // sequentially choose between gpu and tpu
        ret = (i%2 == 0)?gpu:tpu;
    }else if(this->mode == "cts_p"){ // sequentially choose between cpu and tpu
        ret = (i%2 == 0)?cpu:tpu;
    }else if(this->mode == "allr_p"){ // randomly choose a device among cpu, gpu and tpu
        int idx = rand()%3;
        ret = (idx == 0)?cpu:((idx == 1)?gpu:tpu);
    }else if(this->mode == "cgr_p"){ // randomly choose between cpu and gpu
        int idx = rand()%2;
        ret = (idx == 0)?cpu:gpu;
    }else if(this->mode == "gtr_p"){ // randomly choose between gpu and tpu
        int idx = rand()%2;
        ret = (idx == 0)?gpu:tpu;
    }else if(this->mode == "cgr_p"){ // randomly choose between cpu and tpu
        int idx = rand()%2;
        ret = (idx == 0)?cpu:tpu;
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
