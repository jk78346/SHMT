#include <fstream>
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "conversion.h"

using namespace cv;

extern std::vector<std::string> uint8_t_type_app;

UnifyType::UnifyType(Params params, void* in){
    this->params = params;
    if( std::find(uint8_t_type_app.begin(),
                  uint8_t_type_app.end(),
                  params.app_name) !=
        uint8_t_type_app.end() ){
        uint8_t* tmp = reinterpret_cast<uint8_t*>(in);
        this->float_array = (float*) malloc(params.problem_size * 
                                            params.problem_size * 
                                            sizeof(float));
        
        for(int i = 0 ; i < params.problem_size ; i++){
            for(int j = 0 ; j < params.problem_size ; j++){
                this->float_array[i*params.problem_size+j] = 
                    tmp[i*params.problem_size+j]; // uint8_t to float conversion
            }
        }
        this->char_array = tmp; // record the uint8_t pointer
    }else{ // others are default as float type
        this->float_array = reinterpret_cast<float*>(in);
    }
}

void UnifyType::save_as_img(const std::string file_name, 
                            unsigned int rows, 
                            unsigned int cols, 
                            void* img){
    if( std::find(uint8_t_type_app.begin(),
                  uint8_t_type_app.end(),
                  this->params.app_name) !=
        uint8_t_type_app.end() ){
        Mat mat(rows, cols, CV_8U);
        array2mat(mat, this->char_array, rows, cols);
        assert(!mat.empty());
        imwrite(file_name.c_str(), mat);
    }else{
        Mat mat(rows, cols, CV_32F);
        // TODO: how to show float array?
        array2mat(mat, this->float_array, rows, cols);
        assert(!mat.empty());
        imwrite(file_name.c_str(), mat);
    }
}

void UnifyType::save_as_csv(const std::string file_name,
                            unsigned int rows,
                            unsigned int cols,
                            void* img){
    std::fstream myfile;
    myfile.open(file_name.c_str(), std::ios_base::out | std::ios::binary);
    assert(myfile.is_open());

    if(std::find(uint8_t_type_app.begin(),
                uint8_t_type_app.end(),
                this->params.app_name) !=
       uint8_t_type_app.end() ){
        uint8_t* tmp = reinterpret_cast<uint8_t*>(img);    
        for(unsigned int i = 0 ; i < rows ; i++){
            for(unsigned int j = 0 ; j < cols ; j++){
                myfile << std::hex << std::setfill('0') << std::setw(2) 
                       << (unsigned)tmp[i*cols+j] << " ";        
            }
            myfile << std::endl;
        }    
    }else{
        float* tmp = reinterpret_cast<float*>(img);    
        for(unsigned int i = 0 ; i < rows ; i++){
            for(unsigned int j = 0 ; j < cols ; j++){
                myfile << tmp[i*cols+j] << " ";        
            }
            myfile << std::endl;
        }    
    }

}


