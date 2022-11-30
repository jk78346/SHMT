#include <time.h>
#include <stdlib.h>
#include "arrays.h"
#include "partition.h"

std::atomic<int> doneProducers(0);
std::atomic<int> doneConsumers(0);

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
    this->is_dynamic_block = new bool[this->block_cnt]; 
    this->is_dynamic_device = new bool[this->dev_type_cnt]; 
    // For rand_p partition mode
    srand(time(NULL));
    
};

PartitionRuntime::~PartitionRuntime(){
    for(unsigned int i = 0 ; i < this->block_cnt ; i++){
        delete this->generic_kernels[i].kernel_base;
    }   
    delete this->generic_kernels;
    delete this->dev_sequence;
    delete this->is_dynamic_block;
    delete this->is_dynamic_device;
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
    
    /* This is the latest moment to determine if each tiling block and device is 
       dynamic or static. */
    this->populate_dynamic_flags();
    
    // assign partitions to corresponding type of kernel handler if is static.
    for(unsigned int i = 0 ; i < this->block_cnt ; i++){
        if( !this->is_dynamic_block[i] ){
            auto device_type = this->mix_policy(i);
            this->create_kernel_by_type(i, device_type);
            this->generic_kernels[i].kernel_base->input_conversion();
        }
    }
    timing end = clk::now();
    return get_time_ms(end, start);
}

void* PartitionRuntime::RunDeviceThread(void *my_args){
    // getting argument(s)
    struct thread_data *args = (struct thread_data*) my_args;
    auto p_run_ptr = args->p_run_ptr; // pointer of 'this'
    GenericKernel* generic_kernels = args->generic_kernels;
    unsigned int block_cnt = args->block_cnt;
    unsigned int iter = args->iter;
    double kernel_ms = args->kernel_ms;
    DeviceType device_type = args->device_type;
    
    kernel_ms = 0.0;
    
    // To consume any tiling block that is assigned to this device statically.
    for(unsigned int i = 0 ; i < block_cnt ; i++){
        /* Check if the device type this kernel was assigned to is the same as
           the type this consumer thread is representing. 
         */
        if(p_run_ptr->is_dynamic_block[i] == false &&
            generic_kernels[i].device_type == device_type){
            kernel_ms += generic_kernels[i].kernel_base->run_kernel(iter);
        }
    }
    
    // device as dynamic consumer
    if(p_run_ptr->is_dynamic_device[device_type]){
        struct node_data curr_node;
        bool itemsLeft;
        do{
            itemsLeft = doneProducers.load(std::memory_order_acquire) != 1;
            while(p_run_ptr->q.try_dequeue(curr_node)){
                itemsLeft = true;
            /*  Start to consume one tiling block.
                Current implementation has to includes inut conversion overhead 
                since device type is not determined until now.
            */
            unsigned int block_id = curr_node.block_id;
            p_run_ptr->create_kernel_by_type(block_id, device_type);
            p_run_ptr->dev_sequence[block_id] = device_type;
            curr_node.generic_kernel->kernel_base->input_conversion();
            kernel_ms += 
                curr_node.generic_kernel->kernel_base->run_kernel(curr_node.iter);
            }
        }while(itemsLeft || 
                doneConsumers.fetch_add(1, std::memory_order_acq_rel) + 1 == 
                (int)p_run_ptr->dev_type_cnt);
    }
    args->kernel_ms = kernel_ms;
    pthread_exit(NULL);
}

double PartitionRuntime::run_partitions(){
    timing start = clk::now();
    /*
       Dynamic producer of SPMC scheduling.
       Any dynamic tiling block that is left un-assigned to any device during
       static assignment stage now will be push into SPMC FIFO queue for 
       dynamic scheduling.
    */
    for(unsigned int i = 0 ; i < this->block_cnt ; i++){
        if(this->is_dynamic_block[i]){
            struct node_data curr_node;
            curr_node.generic_kernel = &(this->generic_kernels[i]);
            curr_node.params = this->params;
            curr_node.block_id = i;
            curr_node.iter = this->params.iter;
            this->q.enqueue(curr_node);
        }
    }
    doneProducers.fetch_add(1, std::memory_order_release);

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
        td[i].p_run_ptr = this;
        td[i].generic_kernels = this->generic_kernels; 
        td[i].block_cnt = this->block_cnt;
        td[i].iter = this->params.iter;
        pthread_create(&threads[i], NULL, this->RunDeviceThread, (void *)&td[i]);
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

void PartitionRuntime::create_kernel_by_type(unsigned int i/*block_id*/, 
                                             DeviceType device_type){
    if(this->generic_kernels[i].kernel_base != NULL){
        std::cout << "[WARN] " << __func__ << ": generic_kenrels[" << i 
                  << "] has been instanciated as type " 
                  << this->generic_kernels[i].device_type 
                  << ", and now type " << device_type 
                  << " is wanted. Skip creating." << std::endl;
    }else{
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
}

DeviceType PartitionRuntime::mix_policy(unsigned i
        /*index of a tiling task, no larger than this->block_cnt*/){
    DeviceType ret = undefine;
    if(this->mode == "c_p"){ // all partitions on cpu
        ret = cpu;
    }else if(this->mode == "g_p"){ // all partitions on gpu
        ret = gpu;
    }else if(this->mode == "t_p"){ // all partitions on tpu
        ret = tpu;
    }else if(this->mode == "cgt_s"){ // sequentially choose a device between cpu, gpu and tpu
        int idx = i%3;
        ret = (idx == 0)?cpu:((idx == 1)?gpu:tpu);
    }else if(this->mode == "cg_s"){ // sequentially choose between cpu and gpu
        ret = (i%2 == 0)?cpu:gpu;
    }else if(this->mode == "gt_s"){ // sequentially choose between gpu and tpu
        ret = (i%2 == 0)?gpu:tpu;
    }else if(this->mode == "ct_s"){ // sequentially choose between cpu and tpu
        ret = (i%2 == 0)?cpu:tpu;
    }else if(this->mode == "cgt_b" ||
             this->mode == "cg_b" ||
             this->mode == "gt_b" ||
             this->mode == "ct_b"){
        /*
           For work-balancing type of modes, device assignment of each tiling 
           block is dynamic (determined by SPMC at runtime). No need to 
           pre-determine here so do nothing.
         */
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

void PartitionRuntime::populate_dynamic_flags(){
    unsigned int delimiter_loc = this->mode.find("_");
    
    // default as static
    for(unsigned int i = 0 ; i < this->block_cnt ; i++){
        this->is_dynamic_block[i] = false;
    }
    for(unsigned int i = 0 ; i < this->dev_type_cnt ; i++){
        this->is_dynamic_device[i] = false;
    }

    if(delimiter_loc != std::string::npos && 
        this->mode.length() > delimiter_loc &&
        this->mode.substr(delimiter_loc+1, 1) == "b"){
        
        // switch block to dynamic.
        for(unsigned int i = 0 ; i < this->block_cnt ; i++){
            this->is_dynamic_block[i] = true;
        }
        // switch device to dynamic if detected.
        std::string sub_mode = this->mode.substr(0, delimiter_loc);
        if(sub_mode.find("c") != std::string::npos){ // found cpu type
            this->is_dynamic_device[cpu] = true;
        }
        if(sub_mode.find("g") != std::string::npos){ // found gpu type
            this->is_dynamic_device[gpu] = true;
        }
        if(sub_mode.find("t") != std::string::npos){ // found tpu type
            this->is_dynamic_device[tpu] = true;
        }
    }// else: no partition mode(s). all static
}

