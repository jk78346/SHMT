#include <cassert>
#include "params.h"

Params::Params(
    std::string app_name,
    int problem_size,
    int block_size,
    int iter,
    float mix_p,
    std::string input_data_path){

    this->app_name        = app_name;
    this->problem_size    = problem_size;
    this->block_size      = block_size;
    this->iter            = iter;
    this->mix_p           = mix_p; 
    this->input_data_path = input_data_path;

    assert(problem_size >= block_size);
    // A temporary aligned design, can be released later
    assert(problem_size % block_size == 0);
    this->row_cnt = problem_size / block_size; 
    this->col_cnt = problem_size / block_size; 
    this->block_cnt = this->row_cnt * this->col_cnt;
}

unsigned int Params::get_row_cnt(){
    return this->row_cnt;
}

unsigned int Params::get_col_cnt(){
    return this->col_cnt;
}

unsigned int Params::get_block_cnt(){
    return this->block_cnt;
}
