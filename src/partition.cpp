#include <time.h>
#include <stdlib.h>
#include "arrays.h"
#include "quality.h"
#include "partition.h"
#include "conversion.h"

std::atomic<int> doneProducers(0);
std::atomic<int> doneConsumers(0);

PartitionRuntime::PartitionRuntime(Params params,
                                   std::string mode,
                                   void* input,
                                   void* output){
    params.set_tiling_mode(true);
    this->params = params;
    this->row_cnt = params.get_row_cnt();
    this->col_cnt = params.get_col_cnt();
    this->block_cnt = params.get_block_cnt();
    this->criticality.resize(this->block_cnt);
    assert(this->row_cnt * this->col_cnt == this->block_cnt);
    this->mode = mode;
    this->input = input;
    this->output = output;
    this->generic_kernels = new GenericKernel[this->block_cnt];
    this->dev_sequence = new DeviceType[this->block_cnt];
    this->is_dynamic_block = new bool[this->block_cnt]; 
    this->is_dynamic_device = new bool[this->dev_type_cnt+1]; // enum is 1-index. 
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
    this->sampling_qualities.clear();
    this->criticality.clear();
}

bool sortByVal(const std::pair<int , float> &a, const std::pair<int, float> &b){
    return a.second < b.second;
}

/*
 This is the main function to determine criticality of each tiling block based on
 sampling quality.
 */
void PartitionRuntime::criticality_kernel(){
    std::vector<std::pair<int, float>> order;

    for(unsigned int i = 0 ; i < this->sampling_qualities.size() ; i++){
        std::cout << __func__ << ": i: " << i 
                  << ", rmse: " << this->sampling_qualities[i].rmse()
                  << ", rmse %: " << this->sampling_qualities[i].rmse_percentage()
                  << ", error rate: " << this->sampling_qualities[i].error_rate()
                  << ", error %: " << this->sampling_qualities[i].error_percentage()
                  << ", ssim: " << this->sampling_qualities[i].ssim()
                  << ", pnsr: " << this->sampling_qualities[i].pnsr() << std::endl;
        order.push_back(std::make_pair(i, this->sampling_qualities[i].error_rate()));
    }
    sort(order.begin(), order.end(), sortByVal);

    /*
        Current design: mark (no more than) one third of the worst blocks 
            (error_rate worst) to be critical.
        TODO: design the criticality decision 
     */
    int threshold = ceil(order.size() * (2./3.));
    int cnt = 0;
    for(auto p: order){
        this->criticality[p.first] = (cnt < threshold)?false:true;
        cnt++;
        std::cout << __func__ << ": i: " << p.first << ", error rate: " << p.second << std::endl;
    }

    // show criticality
    std::cout << __func__ << ": criticality: ";
    for(auto c: this->criticality){
        std::cout << c << " ";
    }
    std::cout << std::endl;
}

double PartitionRuntime::run_sampling(SamplingMode mode){
    this->params.set_sampling_mode(mode);
    std::cout << __func__ << ": start sampling run, mode: " 
              << this->params.get_sampling_mode() 
              << ", downsampling rate: "
              << this->params.get_downsampling_rate() << std::endl;
    /* Downsampling tiling blocks and assign them to edgetpu. */
    std::vector<void*> input_sampling_pars;
    std::vector<void*> cpu_output_sampling_pars;
    std::vector<void*> tpu_output_sampling_pars;
    
    array_partition_downsampling(this->params,
                                 false,
                                 this->input_pars,
                                 input_sampling_pars);
    
    array_partition_downsampling(this->params,
                                 true, // skip_init
                                 this->output_pars,
                                 cpu_output_sampling_pars);
    
    array_partition_downsampling(this->params,
                                 true, // skip_init
                                 this->output_pars,
                                 tpu_output_sampling_pars);

    double sampling_overhead = 0.0;

    /* run downsampled tiling blocks on edgetpu and get quality result. */
    for(unsigned int i = 0 ; i < this->row_cnt ; i++){
        for(unsigned int j = 0; j < this->col_cnt ; j++){
            unsigned int idx = i*this->col_cnt+j;
            Params params = this->params;
            params.block_size = this->params.block_size * params.get_downsampling_rate();

            // cpu part
             CpuKernel* cpu_kernel_ptr = new CpuKernel(params,
                                                       input_sampling_pars[idx],
                                                       cpu_output_sampling_pars[idx]);
            sampling_overhead += cpu_kernel_ptr->input_conversion();
            sampling_overhead += cpu_kernel_ptr->run_kernel(1);
            sampling_overhead += cpu_kernel_ptr->output_conversion();

            // tpu part
            TpuKernel* tpu_kernel_ptr = new TpuKernel(params,
                                                      input_sampling_pars[idx],
                                                      tpu_output_sampling_pars[idx]);
            sampling_overhead += tpu_kernel_ptr->input_conversion();
            sampling_overhead += tpu_kernel_ptr->run_kernel(1);
            sampling_overhead += tpu_kernel_ptr->output_conversion();

            params.problem_size = params.block_size;
            
            UnifyType* unify_input_type = 
                new UnifyType(params, input_sampling_pars[idx]);          
            UnifyType* unify_cpu_output_type =
                new UnifyType(params, cpu_output_sampling_pars[idx]);
            UnifyType* unify_tpu_output_type =
                new UnifyType(params, tpu_output_sampling_pars[idx]);
                
//            Quality* quality = new Quality(params.block_size, // m
//                                           params.block_size, // n
//                                           params.block_size, // ldn
//                                           params.block_size,
//                                           params.block_size,
//                                           unify_input_type->float_array,
//                                           unify_tpu_output_type->float_array,
//                                           unify_cpu_output_type->float_array);
            this->sampling_qualities.push_back(Quality(params.block_size,
                                                       params.block_size,
                                                       params.block_size,
                                                       params.block_size,
                                                       params.block_size,
                                                       unify_input_type->float_array,
                                                       unify_tpu_output_type->float_array,
                                                       unify_cpu_output_type->float_array));
//            std::cout << __func__ << ": block[" << i << ", " << j << "]: "
//                      << "rmse: " << quality->rmse()
//                      << ", error_rate: " << quality->error_rate()
//                      << ", error_percentage: " << quality->error_percentage()
//                      << ", ssim: " << quality->ssim()
//                      << ", pnsr: " << quality->pnsr() << std::endl;
        }
    }

    std::cout << __func__ << ": sampling timing overhead: " << sampling_overhead << " (ms)" << std::endl;
    /* criticality policy to determine which tiling block(s) are critical. */
    this->criticality_kernel(); 

    // If a block is critical, then it must be static and assigned to GPU.
    // And for non-critical blocks, it is dynamic and upto runtime to determine device type to run.
    // In this way, work stealing is used during runtime.
    
    return sampling_overhead;
}

double PartitionRuntime::prepare_partitions(){
    double ret = 0.0;
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

    if(is_criticality_mode()){
        SamplingMode mode = center_crop;
        ret += this->run_sampling(mode);
    }
    this->setup_dynamic_devices();

    /* This is the latest moment to determine if each tiling block and device is 
       dynamic or static. */
    this->setup_dynamic_blocks();
    
    // assign partitions to corresponding type of kernel handler if is static.
    for(unsigned int i = 0 ; i < this->row_cnt ; i++){
        for(unsigned int j = 0; j < this->col_cnt ; j++){
            unsigned int idx = i*this->col_cnt+j;
            if( !this->is_dynamic_block[idx] ){
                auto device_type = this->mix_policy(idx);
                this->create_kernel_by_type(idx, device_type);
                ret += this->generic_kernels[idx].kernel_base->input_conversion();
            }
        }
    }
    return ret;
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
                Current implementation has to include input conversion overhead 
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
    double ret = 0.0;
    for(unsigned int  i = 0 ; i < this->block_cnt ; i++){
        ret +=  this->generic_kernels[i].kernel_base->output_conversion();
    }  
    output_array_partition_gathering(this->params,
                                     &(this->output),
                                     this->output_pars);
    return ret;
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
                     << device_type 
                     << " on block id: " << i
                     << ", program exits."
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
    }else if(this->mode == "gt_c"){ // criticality mode on GPU/TPU mixing
        ret = (this->criticality[i] == true)?gpu:undefine; // non-critical blocks are dynamic
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
    std::cout << __func__ << ": (in [i, j] indexing)" << std::endl;
    for(unsigned int  i = 0 ; i < this->row_cnt ; i++){
        for(unsigned int j = 0 ; j < this->col_cnt ; j++){
            unsigned int idx = i*this->col_cnt +j;
            int tmp = this->dev_sequence[idx];
            if(tmp == cpu){
                std::cout << "c";
            }else if(tmp == gpu){
                std::cout << "g";
            }else if(tmp == tpu){
                std::cout << "t";
            }
            std::cout << " ";
        }
        std::cout << std::endl;
    }   
}

std::vector<DeviceType> PartitionRuntime::get_device_sequence(){
    std::vector<DeviceType> ret;
    for(unsigned int i = 0 ; i < this->row_cnt ; i++){
        for(unsigned int j = 0 ; j < this->col_cnt ; j++){
            unsigned int idx = i*this->col_cnt+j;
            ret.push_back(this->dev_sequence[idx]);
        }
    }
    return ret;
}

bool PartitionRuntime::is_criticality_mode(){
    bool ret = false;
    unsigned int delimiter_loc = this->mode.find("_");
    if(delimiter_loc != std::string::npos && 
        this->mode.length() > delimiter_loc &&
        this->mode.substr(delimiter_loc+1, 1) == "c"){
        ret = true;
    }
    return ret;
}

/* Setup default dynamic flag based on this->mode */
void PartitionRuntime::setup_dynamic_blocks(){
    unsigned int delimiter_loc = this->mode.find("_");
    
    if(is_criticality_mode()){
        assert(this->criticality.size() == this->block_cnt);    
        for(unsigned int i = 0 ; i < this->block_cnt ; i++){
            this->is_dynamic_block[i] = (this->criticality[i] == true)?false:true;
        }
    }else if(delimiter_loc != std::string::npos && 
             this->mode.length() > delimiter_loc &&
             this->mode.substr(delimiter_loc+1, 1) == "b"){
        // default as dynamic
        for(unsigned int i = 0 ; i < this->block_cnt ; i++){
            this->is_dynamic_block[i] = true;
        }
    
    }else{ // all other non criticality aware non-sampling policies
        // default as static
        for(unsigned int i = 0 ; i < this->block_cnt ; i++){
            this->is_dynamic_block[i] = false;
        }
    }
}

/* Setup default dynamic flag based on this->mode */
void PartitionRuntime::setup_dynamic_devices(){
    unsigned int delimiter_loc = this->mode.find("_");
    
    // default as static
    for(unsigned int i = 0 ; i < this->dev_type_cnt+1/*enum is 1-index*/ ; i++){
        this->is_dynamic_device[i] = false;
    }

    if((delimiter_loc != std::string::npos && 
        this->mode.length() > delimiter_loc &&
        this->mode.substr(delimiter_loc+1, 1) == "b") ||
        this->is_criticality_mode()){
        
        // switch each device to dynamic if detected.
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

