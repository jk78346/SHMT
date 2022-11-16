#ifndef __CONVERSION_H__
#define __CONVERSION_H__
#include <opencv2/opencv.hpp>
#include "params.h"

// Since template is used, this definitions needed to be located in this header file for successful compilation.

template <typename T>
void cpu_kernel_input_conversion(const std::string app_name, Params params, void* input, T& converted_input){
    //TODO: choose the way how to convert the input array based on app_name.
    float* input_array  = reinterpret_cast<float*>(input);
    array2mat(converted_input, input_array, CV_32F, params.problem_size, params.problem_size);
}

template <typename T>
void cpu_kernel_output_conversion(const std::string app_name, Params params, T& output, void* converted_output){
    float* output_array = reinterpret_cast<float*>(converted_output);
    output.convertTo(output, CV_8U);
    mat2array(output, output_array);
}

template <typename T>
void gpu_kernel_input_conversion(const std::string app_name, Params params, void* input, T& converted_input){
    //TODO: choose the way how to convert the input array based on app_name.
    float* input_array  = reinterpret_cast<float*>(input);
    Mat img_host;
    array2mat(img_host, input_array, CV_32F, params.problem_size, params.problem_size);
    converted_input.upload(img_host); // convert from Mat to GpuMat
}

template <typename T>
void gpu_kernel_output_conversion(const std::string app_name, Params params, T& output, void* converted_output){
    float* output_array = reinterpret_cast<float*>(converted_output);
    Mat out_img;
    output.download(out_img); // convert GpuMat to Mat
    out_img.convertTo(out_img, CV_8U);
    mat2array(out_img, output_array);
}


#endif

