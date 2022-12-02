#include <iostream>
#include <algorithm>
#include "conversion.h"

extern std::vector<std::string> uint8_t_type_app;

UnifyType::UnifyType(Params params, void** in){
    if( std::find(uint8_t_type_app.begin(),
                  uint8_t_type_app.end(),
                  params.app_name) !=
        uint8_t_type_app.end() ){
        std::cout << __func__ << ": uint8_t type" << std::endl;
        uint8_t* tmp = reinterpret_cast<uint8_t*>(in);
        this->float_array = (float*) malloc(params.problem_size * 
                                            params.problem_size * 
                                            sizeof(float));
        std::cout << "params.problem_size: " << params.problem_size << std::endl;
        
        for(int i = 0 ; i < params.problem_size ; i++){
            for(int j = 0 ; j < params.problem_size ; j++){
                this->float_array[i*params.problem_size+j] = 
                    tmp[i*params.problem_size+j]; // uint8_t to float conversion
            }
        }
    }else{ // others are default as float type
        std::cout << __func__ << ": float type" << std::endl;
        this->float_array = reinterpret_cast<float*>(in);
    }
}

UnifyType::~UnifyType(){
    free(this->float_array);
}

float* UnifyType::convert_to_float(){
    return this->float_array;
}
