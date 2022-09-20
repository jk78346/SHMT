#ifndef __PARAMS_H__
#define __PARAMS_H__
#include <string>

class Params{
    public:
        Params(std::string app_name="sobel_2d",
              int problem_size=2048,
              int block_size=2048,
              int iter=1,
              std::string baseline_mode="cpu",
              std::string target_mode="tpu",
              float mix_p=0.5,
              std::string input_data_path="../data/lena_gray_2Kx2K.bmp");
        std::string app_name;
        int problem_size;
        int block_size;
        int iter;
        std::string baseline_mode;
        std::string target_mode;
        float mix_p;
        std::string input_data_path;
};
#endif
