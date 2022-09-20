#include "params.h"

Params::Params(
    std::string app_name,
    int problem_size,
    int block_size,
    int iter,
    std::string baseline_mode,
    std::string target_mode,
    float mix_p,
    std::string input_data_path){
    this->app_name        = app_name;
    this->problem_size    = problem_size;
    this->block_size      = block_size;
    this->iter            = iter;
    this->baseline_mode   = baseline_mode;
    this->target_mode     = target_mode;
    this->mix_p           = mix_p; 
    this->input_data_path = input_data_path;
}
