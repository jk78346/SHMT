#ifndef __CONVERSION_H__
#define __CONVERSION_H__
#include "arrays.h"
#include "params.h"

class UnifyType{
public:
    UnifyType(Params params, void** in);
    ~UnifyType();
    float* convert_to_float();
    float* float_array;
};

#endif

