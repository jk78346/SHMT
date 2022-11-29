#include <time.h>
#include <stdlib.h>
#include "arrays.h"
#include "partition.h"
#include "concurrentqueue.h"

#define SERIAL 0
#define PTHREAD 1
#define CONCURRENT_QUEUE 2

#define CONCURRENT PTHREAD //CONCURRENT_QUEUE

#if CONCURRENT == PTHREAD || CONCURRENT == CONCURRENT_QUEUE
struct thread_data{
    GenericKernel* generic_kernels;
    unsigned int block_cnt;
    unsigned int iter;
    double kernel_ms;
    // The device type the thread is representing.
    DeviceType device_type;
};

#if CONCURRENT == CONCURRENT_QUEUE
using namespace moodycamel;
 
struct node_data{
    PartitionRuntime* p_run_ptr;
    GenericKernel* generic_kernel;
    Params params;
    unsigned int block_id;
    unsigned int iter;
};

ConcurrentQueue<struct node_data> q;
std::atomic<int> doneProducers(0);
std::atomic<int> doneConsumers(0);
#endif

void *RunDeviceThread(void *my_args){
    // getting argument(s)
    struct thread_data *args = (struct thread_data*) my_args;
    GenericKernel* generic_kernels = args->generic_kernels;
    unsigned int block_cnt = args->block_cnt;
    unsigned int iter = args->iter;
    double kernel_ms = args->kernel_ms;
    DeviceType device_type = args->device_type;
    
    kernel_ms = 0.0;
#if CONCURRENT == PTHREAD
    for(unsigned int  i = 0 ; i < block_cnt ; i++){
        /* Check if the device type this kernel was assigned to is the same as
           the type this consumer thread is representing. 
         */
        if(generic_kernels[i].device_type == device_type){
            kernel_ms += generic_kernels[i].kernel_base->run_kernel(iter);
        }
    }
#elif CONCURRENT == CONCURRENT_QUEUE
    struct node_data curr_node;
    bool itemsLeft;
    do{
        itemsLeft = doneProducers.load(std::memory_order_acquire) != 1;
        while(q.try_dequeue(curr_node)){
            itemsLeft = true;
        /*  Start to consume one tiling block.
            Current implementation has to includes inut conversion overhead 
            since device type is not determined until now.
         */
        unsigned int block_id = curr_node.block_id;
        curr_node.p_run_ptr->create_kernel_by_type(block_id, device_type);
        curr_node.generic_kernel->kernel_base->input_conversion();
        kernel_ms += curr_node.generic_kernel->kernel_base->run_kernel(curr_node.iter);
        }
    }while(itemsLeft || 
                doneConsumers.fetch_add(1, std::memory_order_acq_rel) + 1 == 
                (int)curr_node.p_run_ptr->dev_type_cnt);
#endif
    args->kernel_ms = kernel_ms;
    pthread_exit(NULL);
}
#endif

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

void PartitionRuntime::create_kernel_by_type(unsigned int i/*block_id*/, 
                                             DeviceType device_type){
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
/*
    Work-balancing type of partition modes require device type to be determined
    at SPMC runtime stage, but not pre-execution stage as here.
 */
#if CONCURRENT != CONCURENT_QUEUE
    // assign partitions to each type of kernel handler
    for(unsigned int i = 0 ; i < this->block_cnt ; i++){
        auto device_type = this->mix_policy(i);
        this->create_kernel_by_type(i, device_type);
        this->generic_kernels[i].kernel_base->input_conversion();
    }
#endif
    timing end = clk::now();
    return get_time_ms(end, start);
}

double PartitionRuntime::run_partitions(){
#if CONCURRENT == PTHREAD || CONCURRENT == CONCURRENT_QUEUE
    timing start = clk::now();

#if CONCURRENT == CONCURRENT_QUEUE
    /*
        This version uses SPMC where the main thread is producing tiling blocks
        and consumers are consuming tasks. Each consumer represents one type of
        computation hardware including CPU, GPU and edgeTPU.
    */
    // producer
    for(unsigned int i = 0 ; i < this->block_cnt ; i++){
        struct node_data curr_node;
        curr_node.p_run_ptr = this;
        curr_node.generic_kernel = &(this->generic_kernels[i]);
        curr_node.params = this->params;
        curr_node.block_id = i;
        curr_node.iter = this->params.iter;
        q.enqueue(curr_node);
    }
    doneProducers.fetch_add(1, std::memory_order_release);
#endif

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
#elif CONCURRENT == SERIAL 
    double kernel_ms = 0.0;
    for(unsigned int  i = 0 ; i < this->block_cnt ; i++){
        kernel_ms += this->generic_kernels[i].kernel_base->run_kernel(this->params.iter);
    }   
    return kernel_ms;
#else // any undefined version of implementation
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
    }else if(this->mode == "allb_p" ||
             this->mode == "cgb_p" ||
             this->mode == "gtb_p" ||
             this->mode == "ctb_p"){
        /*
           For work-balancing type of modes, device assignment of each tiling 
           block is determined by SPMC at runtime. No need to pre-determine
           here so do nothing but dummy.
         */
        ret = cpu;
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
