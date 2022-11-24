#ifndef __PARAMS_H__
#define __PARAMS_H__
#include <string>

class Params{
public:
    Params(std::string app_name="sobel_2d",
          int problem_size=2048,
          int block_size=2048,
          bool tiling_mode = false,
          int iter=1,
          float mix_p=0.5,
          std::string input_data_path="../data/lena_gray_2Kx2K.bmp");
    
    void set_tiling_mode(bool);
    bool get_tiling_mode();
    unsigned int get_row_cnt();
    unsigned int get_col_cnt();
    unsigned int get_block_cnt();
    unsigned int get_kernel_size();

    std::string app_name;
    int problem_size;
    int block_size;
    bool tiling_mode;
    unsigned int iter;
    float mix_p;
    std::string input_data_path;

private:        
    unsigned int row_cnt = 0;
    unsigned int col_cnt = 0;
    unsigned int block_cnt = 0;
};
#endif
