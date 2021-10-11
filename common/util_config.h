#ifndef UTIL_CONFIG_H_
#define UTIL_CONFIG_H_

struct _CONFIG{
        int mode; // [0:full|1:blk|2:mix|-1:off]
        int dev;  // [0:gpu |1:trt|2:tpu]
        unsigned int w_cnt; // block count in width  direction, the  first params from [hxw]
        unsigned int h_cnt; // block count in height direction, the second params from [hxw]
}CONFIG;

void init_configs(struct _CONFIG* configs);
void get_configs(int argc, char* argv[], struct _CONFIG* configs);
void check_configs(struct _CONFIG* configs);

#endif
