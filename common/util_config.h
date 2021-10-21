#ifndef UTIL_CONFIG_H_
#define UTIL_CONFIG_H_

struct _blk_dims{
    int w_cnt; // number of block counts in input's width  direction
    int h_cnt; // number of block counts in input's height direction
    int w_size; // per block size in input's width  direction (# of pixels)
    int h_size; // per block size in input's height direction (# of pixels)
    int blk_cnt;  // input's block counts in total
    int blk_size; // input's block size in total (# of pixels)
}blk_dims;

struct _blk_pemeter{
    struct _blk_dims in_dims;
    struct _blk_dims out_dims;
} blk_pemeter;

struct _CONFIG{
    int mode; // [0:full|1:blk|2:mix|-1:off]
    int dev;  // [0:gpu |1:trt|2:tpu]
    unsigned int w_cnt; // block count in width  direction, the  first params from [hxw]
    unsigned int h_cnt; // block count in height direction, the second params from [hxw]
    struct _blk_pemeter s_blk_pemeter;
    char* model; //TODO: later on this could be a list of models
}CONFIG;

void init_configs(struct _CONFIG* configs);
int get_configs(int argc, char* argv[], struct _CONFIG* configs);
void check_configs(struct _CONFIG* configs);

void select_model(struct _CONFIG config);
void configure_blk(struct _CONFIG config);
#endif
