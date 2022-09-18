#ifndef __PARAMS_H__
#define __PARAMS_H__
#include <string>

class Params{
    public:
        Params();
        int problem_size;
        int block_size;
        int iter;
        std::string baseline_mode;
        std::string target_mode;
        float mix_p;
};
#endif
