#include <opencv2/opencv.hpp>
#include "arrays.h"
#include "utils.h"

/*
    Input array allocation and initialization.
    The data type of arrays depends on applications.
    EX:
        uint8_t: sobel_2d, mean_2d, laplacian_2d
        float:   fft_2d, dct8x8_2d, blackscholes
 */

std::vector<std::string> uint8_t_type_app = {
    "sobel_2d",
    "mean_2d",
    "laplacian_2d"
};

void data_initialization(Params params,
                         void** input_array,
                         void** output_array_baseline,
                         void** output_array_proposed){
    // TODO: choose corresponding initial depending on app_name.
    int rows = params.problem_size;
    int cols = params.problem_size;
    unsigned int input_total_size = rows * cols;
    unsigned int output_total_size = rows * cols;
    
    // image filter type of kernels
    if( std::find(uint8_t_type_app.begin(), 
                  uint8_t_type_app.end(), 
                  params.app_name) !=
        uint8_t_type_app.end() ){
        *input_array = (uint8_t*) malloc(input_total_size * sizeof(uint8_t));
        *output_array_baseline = (uint8_t*) malloc(output_total_size * sizeof(uint8_t));
        *output_array_proposed = (uint8_t*) malloc(output_total_size * sizeof(uint8_t));   
        Mat in_img;
        read_img(params.input_data_path,
                 rows,
                 cols,
                 in_img);
        mat2array(in_img, (uint8_t*)*input_array);
    }else{ // others are default as float type
        *input_array = (float*) malloc(input_total_size * sizeof(float));
        *output_array_baseline = (float*) malloc(output_total_size * sizeof(float));
        *output_array_proposed = (float*) malloc(output_total_size * sizeof(float));   
        Mat in_img;
        read_img(params.input_data_path,
                 rows,
                 cols,
                 in_img);
        mat2array(in_img, (float*)*input_array);
    }
}

/*
    partition array into partitions: allocation and initialization(optional).
*/
void array_partition_initialization(Params params,
                                    bool skip_init,
                                    void** input,
                                    std::vector<void*>& input_pars){
    // prepare for utilizing opencv roi() to do partitioning.
    Mat input_mat, tmp(params.block_size, params.block_size, CV_32F);
    if(!skip_init){
        array2mat(input_mat, (float*)*input, params.problem_size, params.problem_size);
    }
    unsigned int block_total_size = params.block_size * params.block_size;

    // vector of partitions allocation
    input_pars.resize(params.get_block_cnt());   
    for(unsigned int i = 0 ; i < params.get_row_cnt() ; i++){
        for(unsigned int j = 0 ; j < params.get_col_cnt() ; j++){ 
            unsigned int idx = i * params.get_col_cnt() + j;         
    
            // partition allocation
            input_pars[idx] = (float*) calloc(block_total_size, sizeof(float));
            
            // partition initialization
            if(!skip_init){
                Rect roi(i*params.block_size, j*params.block_size, params.block_size, params.block_size); 
                input_mat(roi).copyTo(tmp); 
                mat2array(tmp, (float*)((input_pars[idx])));
            }
        }
    }
}

/*
    Remap output partitions into one single output array.
*/
void output_array_partition_gathering(Params params,
                                      void** output,
                                      std::vector<void*>& output_pars){
    // prepare for utilizing opencv roi() to do gathering.
    Mat output_mat(params.problem_size, params.problem_size, CV_32F), tmp;
    
    for(unsigned int i = 0 ; i < params.get_row_cnt() ; i++){
        for(unsigned int j = 0 ; j < params.get_col_cnt() ; j++){
            unsigned int idx = i * params.get_col_cnt() + j;
            array2mat(tmp, (float*)((output_pars[idx])), params.block_size, params.block_size);
            Rect roi(i*params.block_size, j*params.block_size, params.block_size, params.block_size); 
            tmp.copyTo(output_mat(roi));
        }
    }
    mat2array(output_mat, (float*)*output);
}



