#include "performance.h"

double TimeBreakDown::get_total_time_ms(){
    return this->input_time_ms +
           this->kernel_time_ms +
           this->output_time_ms;
};
