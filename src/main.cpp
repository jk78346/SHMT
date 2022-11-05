#include <iostream>
#include <opencv2/opencv.hpp>
#include "gptpu.h"
#include "types.h"
#include "utils.h"
#include "params.h"
#include "kernels.h"
#include "quality.h"

using namespace cv;

extern std::unordered_map<std::string, func_ptr_cpu> cpu_func_table; 
extern std::unordered_map<std::string, func_ptr_gpu> gpu_func_table; 

float run_kernel_on_cpu(std::string app_name, Mat& img, float* output_array){
    // kernel existence checking
    if(cpu_func_table.find(app_name) == cpu_func_table.end()){
        std::cout << "app_name: " << app_name << " not found" << std::endl;
        std::cout << "supported app: " << std::endl;
        for(auto const &pair: cpu_func_table){
            std::cout << "{" << pair.first << ": " << pair.second << "}" << std::endl;
        }
        exit(0);
    }

    Mat out_img;
    
    timing start = clk::now();
    // Actual kernel call
    cpu_func_table[app_name](img, out_img);
    timing end   = clk::now(); 
    
    out_img.convertTo(out_img, CV_8U);
    mat2array(out_img, output_array);

    return get_time_ms(end, start);
}

float run_kernel_on_gpu(std::string app_name, cuda::GpuMat& img, float* output_array){
    // kernel existence checking
    if(gpu_func_table.find(app_name) == gpu_func_table.end()){
        std::cout << "app_name: " << app_name << " not found" << std::endl;
        std::cout << "supported app: " << std::endl;
        for(auto const &pair: gpu_func_table){
            std::cout << "{" << pair.first << ": " << pair.second << "}" << std::endl;
        }
        exit(0);
    }

    cuda::GpuMat out_img_gpu;
    
    timing start = clk::now();
    // Actual kernel call
    gpu_func_table[app_name](img, out_img_gpu);
    timing end   = clk::now(); 
    
    Mat out_img;
    out_img_gpu.download(out_img); // convert GpuMat to Mat
    out_img.convertTo(out_img, CV_8U);
    mat2array(out_img, output_array);

    return get_time_ms(end, start);
}

float run_kernel(const std::string& mode, Params& params, float* input_array, float* output_array){
    float kernel_ms = 0.0;
    std::cout << __func__ << ": start running kernel in " << mode << " mode." << std::endl;
    if(mode == "cpu"){
        Mat img;
        array2mat(img, input_array, CV_32F, params.problem_size, params.problem_size);
        kernel_ms = run_kernel_on_cpu(params.app_name, img, output_array);        
    }else if(mode == "gpu"){
	Mat img_host;
        array2mat(img_host, input_array, CV_32F, params.problem_size, params.problem_size);
	cuda::GpuMat img;
	img.upload(img_host); // convert from Mat to GpuMat
	kernel_ms = run_kernel_on_gpu(params.app_name, img, output_array);        
    }else if(mode == "tpu"){
        int in_size  = params.problem_size * params.problem_size;
        int out_size = params.problem_size * params.problem_size;
        int* in  = (int*) malloc(in_size * sizeof(int));
        int* out = (int*) calloc(out_size, sizeof(int));
        for(int i = 0 ; i < in_size ; i++){
            in[i] = ((int)(input_array[i] + 128)) % 256; // float to int conversion
        }
        std::string kernel_path = get_edgetpu_kernel_path(params.app_name, params.problem_size, params.problem_size);
        run_a_model(kernel_path, params.iter, in, in_size, out, out_size, 1);
        for(int i = 0 ; i < out_size ; i++){
            output_array[i] = out[i]; // int to float conversion
        }
        openctpu_clean_up();
    }
    return kernel_ms;
}

int main(int argc, char* argv[]){
    if(argc != 6){
        std::cout << "Usage: " << argv[0] 
                  << " <application name> <problem_size> <iter> <baseline mode> <proposed mode>" 
                  << std::endl;
        return 0;
    }
    std::string app_name = argv[1];
    int problem_size     = atoi(argv[2]);
    int iter             = atoi(argv[3]);
    std::string baseline_mode = argv[4];
    std::string proposed_mode = argv[5];

    Params params(app_name, 
                  problem_size, 
                  problem_size/*block size*/, 
                  iter);

    int rows = params.problem_size;
    int cols = params.problem_size;
    unsigned int input_total_size = rows * cols;
    float* input_array = (float*) malloc(input_total_size * sizeof(float));
    
    unsigned int output_total_size = rows * cols;
    float* output_array_baseline = (float*) malloc(output_total_size * sizeof(float));
    float* output_array_proposed = (float*) malloc(output_total_size * sizeof(float));

    Mat in_img;
    Mat out_img;
    
    read_img(params.input_data_path, 
             rows, 
             cols, 
             in_img);
    mat2array(in_img, input_array);

    // peak input image for a cropped section
    std::cout << __func__ << ": peaking input image data..." << std::endl; 
    for(int i = 0 ; i < 5 ; i++){
        for(int j = 0 ; j < 5 ; j++){
            std::cout << input_array[i*params.problem_size+j] << " ";
        }
        std::cout << std::endl;
    }

    float baseline_kernel_ms = 0;
    float proposed_kernel_ms = 0;
    
    timing baseline_start = clk::now();
    // Start to run baseline version of the application's implementation.
    baseline_kernel_ms = run_kernel(baseline_mode, 
                                    params, 
                                    input_array, 
                                    output_array_baseline);
    timing baseline_end = clk::now();
    
    timing proposed_start = clk::now();
    // Start to run proposed version of the application's implementation.
    proposed_kernel_ms = run_kernel(proposed_mode, 
                                    params, 
                                    input_array, 
                                    output_array_proposed);
    timing proposed_end = clk::now();
    
    // Calculate end to end latency of each implementation
    double baseline_e2e_ms = get_time_ms(baseline_end, baseline_start);
    double proposed_e2e_ms = get_time_ms(proposed_end, proposed_start);

    // Get quality measurements
    Quality quality(params.problem_size, // m
                    params.problem_size, // n
                    params.problem_size, // ldn
                    output_array_proposed, 
                    output_array_baseline);
    quality.print_results(1);

    std::cout << "===== Latency =====" << std::endl;
    std::cout << baseline_mode << " kernel: " << baseline_kernel_ms << " (ms)" << std::endl;
    std::cout << proposed_mode << " kernel: " << proposed_kernel_ms << " (ms)" << std::endl;
    std::cout << baseline_mode << "    e2e: " << baseline_e2e_ms << " (ms)" << std::endl;
    std::cout << proposed_mode << "    e2e: " << proposed_e2e_ms << " (ms)" << std::endl;
    return 0;
}
