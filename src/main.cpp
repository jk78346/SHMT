#include <iostream>
#include <opencv2/opencv.hpp>
#include "gptpu.h"
#include "types.h"
#include "utils.h"
#include "params.h"
#include "kernels.h"
#include "quality.h"

using namespace cv;

float run_kernel_on_cpu(Mat& img, float* output_array){
    timing start = clk::now();
    
    Mat out_img;
    sobel_2d_cpu(img, out_img);
    mat2array(out_img, output_array);

    timing end   = clk::now(); 
    return get_time_ms(end, start);
}

float run_kernel(const std::string& mode, Params& params, float* input_array, float* output_array){
    float kernel_ms = 0.0;
    if(mode == "cpu"){
        Mat img;
        array2mat(img, input_array, CV_32F, params.problem_size, params.problem_size);
        kernel_ms = run_kernel_on_cpu(img, output_array);        
    }else if(mode == "tpu"){
        int in_size  = params.problem_size * params.problem_size;
        int out_size = params.problem_size * params.problem_size;
        int* in  = (int*) malloc(in_size * sizeof(int));
        int* out = (int*) calloc(out_size, sizeof(int));
        for(int i = 0 ; i < in_size ; i++){
            in[i] = ((int)(input_array[i] + 128)) % 256; // float to int conversion
        }
        std::string kernel_path = "../models/sobel_2d_2048x2048/sobel_2d_edgetpu.tflite";
        run_a_model(kernel_path, 1, in, in_size, out, out_size, 1);
        for(int i = 0 ; i < out_size ; i++){
            output_array[i] = out[i]; // int to float conversion
        }
        openctpu_clean_up();
    }
    

    return kernel_ms;
}

int main(){
    Params params;

    int rows = params.problem_size;
    int cols = params.problem_size;
    unsigned int input_total_size = rows * cols;
    float* input_array = (float*) malloc(input_total_size * sizeof(float));
    
    unsigned int output_total_size = rows * cols;
    float* output_array_baseline = (float*) malloc(output_total_size * sizeof(float));
    float* output_array_proposed = (float*) malloc(output_total_size * sizeof(float));

    Mat in_img;
    Mat out_img;

    std::string file_name = "../data/lena_gray_2Kx2K.bmp";
    
    read_img(file_name, rows, cols, in_img);
    mat2array(in_img, input_array);

    float proposed_kernel_ms = 0;
    float baseline_kernel_ms = 0;
    baseline_kernel_ms = run_kernel(params.baseline_mode, params, input_array, output_array_baseline);
    proposed_kernel_ms = run_kernel(params.target_mode,   params, input_array, output_array_proposed);

    Quality quality(params.problem_size, params.problem_size, params.problem_size, output_array_proposed, output_array_baseline);

    float rmse             = quality.rmse(0);
    float error_rate       = quality.error_rate(0);
    float error_percentage = quality.error_percentage(0);
    float ssim             = quality.ssim(0);
    float pnsr             = quality.pnsr(0);

    printf("rmse: %f %%\n", rmse);
    printf("error rate: %f %%\n", error_rate);
    printf("error percentage: %f %%\n", error_percentage);
    printf("ssim: %f\n", ssim);
    printf("pnsr: %f dB\n", pnsr);

    printf("baseline kernel: %f (ms)\n", baseline_kernel_ms);
    printf("proposed kernel: %f (ms)\n", proposed_kernel_ms);

    return 0;
}
