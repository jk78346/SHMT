#ifndef __ARRAYS_H__
#define __ARRAYS_H__

#include "params.h"

void input_array_partition_initialization(Params params, 
                                          void* input, 
                                          void** input_pars);
void output_array_partition_gathering(Params params, 
                                      void* output, 
                                      void** output_pars);
void data_initialization(Params params, 
                         void** input_array, 
                         void** output_array_baseline, 
                         void** output_array_proposed);
#endif

