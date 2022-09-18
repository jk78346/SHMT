#include "params.h"

Params::Params(){
    this->problem_size  = 2048;
    this->block_size    = 2048;
    this->iter          = 1;
    this->baseline_mode = "cpu";
    this->target_mode   = "tpu";
    this->mix_p         = 0.5; 
}
