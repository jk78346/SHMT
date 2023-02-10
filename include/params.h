#ifndef __PARAMS_H__
#define __PARAMS_H__
#include <string>
#include <vector>

enum SamplingMode {cv_resize, center_crop, init_crop, random_crop};

class Params{
public:
    Params(std::string app_name="sobel_2d",
          int problem_size=2048,
          int block_size=2048,
          bool tiling_mode = false,
          int iter=1,
          //std::string input_data_path="../data/lena_gray_2Kx2K.bmp"
          std::string input_data_path="../data/super5.png"
          );
    
    void set_tiling_mode(bool);
    bool get_tiling_mode();
    unsigned int get_row_cnt();
    unsigned int get_col_cnt();
    unsigned int get_block_cnt();
    unsigned int get_kernel_size();
    void set_downsampling_rate(float r){this->downsampling_rate = r; };
    float get_downsampling_rate(){ return this->downsampling_rate; };
    void set_sampling_mode(SamplingMode mode){ this->sampling_mode = mode; };
    SamplingMode get_sampling_mode(){ return this->sampling_mode; };

    std::string app_name;
    int problem_size;
    int block_size;
    bool tiling_mode;
    unsigned int iter;
    std::string input_data_path; 

    std::vector<std::string> uint8_t_type_app = {
        "sobel_2d",
        "mean_2d",
        "laplacian_2d",
        "kmeans_2d"
    };
    
    /* Return maximun percentage of tiling blocks can be protected as critical
        before degrading latency.
     */
    void set_criticality_ratio(float val){ this->criticality_ratio = val; };
    float get_criticality_ratio(/*std::string app_name, int block_size*/){ return this->criticality_ratio; };

private:        
    unsigned int row_cnt = 0;
    unsigned int col_cnt = 0;
    unsigned int block_cnt = 0;
    float downsampling_rate = 0.25;
    SamplingMode sampling_mode = center_crop;
    
    float criticality_ratio = 1./3.;
};
#endif
