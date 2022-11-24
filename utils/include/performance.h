#ifndef __PERFORMANCE_H__
#define __PERFORMANCE_H__

class TimeBreakDown{
public:
    // useful APIs
    double get_total_time_ms();
    
    // section timing
    double input_time_ms;
    double kernel_time_ms;
    double output_time_ms;
};
#endif

