#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include "string.h"
#include "util_config.h"

void init_configs(struct _CONFIG* configs){
        configs[0].mode = 0;
        configs[0].dev  = 0;
        configs[0].w_cnt = 1;
        configs[0].h_cnt = 1;
        configs[1].mode = -1; // default off
        configs[1].dev  = 0;
        configs[1].w_cnt = 1;
        configs[1].h_cnt = 1;
}

void get_configs(int argc, char* argv[], struct _CONFIG* configs){
        if(argc <= 3){
                printf("argc = %d, while the usage is the following. [first version is not completed]\n", argc);
                printf("./exe [full|blk|mix] [gpu|trt|tpu] [hxw] None                               // for running one version \n");
                printf("./exe [full|blk|mix] [gpu|trt|tpu] [hxw] [full|blk|mix] [gpu|trt|tpu] [hxw] // for running two versions\n");
                exit(0);
        }
        if(strncmp(argv[1], "full", sizeof("full")) == 0){
                configs[0].mode = 0;
        }else if(strncmp(argv[1], "blk", sizeof("blk")) == 0){
                configs[0].mode = 1;
        }else if(strncmp(argv[1], "mix", sizeof("mix")) == 0){
                configs[0].mode = 2;
        }else{
                printf("version 1 mode is undefined: %s\n", argv[1]); exit(0);
        }
        if(strncmp(argv[2], "gpu", sizeof("gpu")) == 0){
                configs[0].dev = 0;
        }else if(strncmp(argv[2], "trt", sizeof("trt")) == 0){
                configs[0].dev = 1;
        }else if(strncmp(argv[2], "tpu", sizeof("tpu")) == 0){
                configs[0].dev = 2;
        }else{
                printf("version 1 device is undefined: %s\n", argv[2]); exit(0);
        }
        unsigned int h_cnt = atoi(strtok(argv[3], "x"));
        unsigned int w_cnt = atoi(strtok(NULL   , "x"));
        configs[0].h_cnt = h_cnt;
        configs[0].w_cnt = w_cnt;
        if(argc == 4 || argc == 5){ // indicate that only one version is specified.
                return;
        }
        if(argc <= 6){
                printf("argc = %d, while the usage is the following. [second version is not completed]\n", argc);
                printf("./exe [full|blk|mix] [gpu|trt|tpu] [hxw] None                               // for running one version \n");
                printf("./exe [full|blk|mix] [gpu|trt|tpu] [hxw] [full|blk|mix] [gpu|trt|tpu] [hxw] // for running two versions\n");
                exit(0);
        }
        if(strncmp(argv[4], "full", sizeof("full")) == 0){
                configs[1].mode = 0;
        }else if(strncmp(argv[4], "blk", sizeof("blk")) == 0){
                configs[1].mode = 1;
        }else if(strncmp(argv[4], "mix", sizeof("mix")) == 0){
                configs[1].mode = 2;
        }else{
                printf("version 2 mode is undefined: %s\n", argv[4]); exit(0);
        }
        if(strncmp(argv[5], "gpu", sizeof("gpu")) == 0){
                configs[1].dev = 0;
        }else if(strncmp(argv[5], "trt", sizeof("trt")) == 0){
                configs[1].dev = 1;
        }else if(strncmp(argv[5], "tpu", sizeof("tpu")) == 0){
                configs[1].dev = 2;
        }else{
                printf("version 2 device is undefined: %s\n", argv[5]); exit(0);
	}
        h_cnt = atoi(strtok(argv[6], "x"));
        w_cnt = atoi(strtok(NULL   , "x"));
        configs[1].h_cnt = h_cnt;
        configs[1].w_cnt = w_cnt;
}

void check_configs(struct _CONFIG* configs){
        if(configs[0].mode < 0 || configs[0].mode > 3){
                printf("version 1 mode invalid: %d\n", configs[0].mode);
        }
        if(configs[0].dev < 0 || configs[0].dev > 3){
                printf("version 1 device invalid: %d\n", configs[0].dev);
        }
        if(configs[0].mode == 0 && (configs[0].w_cnt != 1 || configs[0].h_cnt != 1)){
                printf("version 1 uses full mode while blk cnt: (h:%d, w:%d) is not 1x1\n", configs[0].h_cnt, configs[0].w_cnt);
        }
        if(configs[1].mode < -1 || configs[1].mode > 3){
                printf("version 2 mode invalid: %d\n", configs[1].mode);
        }
        if(configs[1].dev < 0 || configs[1].dev > 3){
                printf("version 2 device invalid: %d\n", configs[1].dev);
        }
        if(configs[1].mode == 0 && (configs[1].w_cnt != 1 || configs[1].h_cnt != 1)){
                printf("version 2 uses full mode while blk cnt: (h:%d, w:%d) is not 1x1\n", configs[1].h_cnt, configs[1].w_cnt);
        }
}

unsigned int file_exists(char* filename){
	struct stat buf;
	return (stat(filename, &buf) == 0);	
}

void select_model(int mode, int dev, unsigned int w_cnt, unsigned int h_cnt, char* model){
	printf("%s %s: under implementation...\n", __FILE__, __func__);
/*dev == 0 : gpu | dev == 1 : trt | dev == 2: tpu*/
	if(mode == -1){ return;  // implies that setting 2 is not used, simply return and do nothing	
	}else if(mode == 0){ // full mode
		if(w_cnt == 1 && h_cnt == 1){
			if(dev == 0){       model = "./model/toy_pose3d_full_model_float.tflite";
			}else if(dev == 1){ model = "./model/toy_pose3d_full_model_float.onnx";
			}else{     	    model = "./model/toy_pose3d_full_model_edgetpu.tflite"; }
		}else{ printf("%s %s: [ERROR] full mode is used but w_cnt = %d, h_cnt = %d\n", __FILE__, __func__, w_cnt , h_cnt); exit(0); }		
	}else if(mode == 1){ // blk mode
		char* w_cnt_char[10]; // upto 10 digits
		char* h_cnt_char[10];
		sprintf(w_cnt_char, "%ld", w_cnt);
		sprintf(h_cnt_char, "%ld", h_cnt);
		model = "./model/toy_pose3d_sp_model_blk_";
		sprintf(model, "%s%s", model, h_cnt_char);
		sprintf(model, "x");
		sprintf(model, "%s%s", model, w_cnt_char);
		if(dev == 0){ 	    sprintf(model, "%s%s", model, "_float.tflite"); 
		}else if(dev == 1){ sprintf(model, "%s%s", model, "_float.onnx");
	       	}else{              sprintf(model, "%s%s", model, "_edgetpu.tflite"); }		
	}else if(mode == 2){ // mix mode
		printf("mix mode is under construction..., exit\n"); exit(0);
	}else{
		printf("%s %s: [ERROR] undefine mode used: %d\n", __FILE__, __func__, mode); exit(0);
	}
	if(!file_exists(model)){
		printf("%d %s: [ERROR] file %s doesn't exist, exit\n", __FILE__, __func__, model); exit(0);
	}
}


void configure_blk(struct _CONFIG config){
	config.s_blk_pemeter.in_dims.w_cnt    = config.w_cnt;
	config.s_blk_pemeter.in_dims.h_cnt    = config.h_cnt;
	config.s_blk_pemeter.in_dims.w_size   = (448 / config.s_blk_pemeter.in_dims.w_cnt);
	config.s_blk_pemeter.in_dims.h_size   = (256 / config.s_blk_pemeter.in_dims.h_cnt);
	config.s_blk_pemeter.in_dims.blk_size = config.s_blk_pemeter.in_dims.w_size * config.s_blk_pemeter.in_dims.h_size;
	config.s_blk_pemeter.in_dims.blk_cnt  = config.s_blk_pemeter.in_dims.w_cnt  * config.s_blk_pemeter.in_dims.h_cnt;
	
	config.s_blk_pemeter.out_dims.w_cnt    = config.s_blk_pemeter.in_dims.w_cnt / 8;
	config.s_blk_pemeter.out_dims.h_cnt    = config.s_blk_pemeter.in_dims.h_cnt / 8;
	config.s_blk_pemeter.out_dims.w_size   = (56 / config.s_blk_pemeter.out_dims.w_cnt);
	config.s_blk_pemeter.out_dims.h_size   = (32 / config.s_blk_pemeter.out_dims.h_cnt);
	config.s_blk_pemeter.out_dims.blk_size = config.s_blk_pemeter.out_dims.w_size * config.s_blk_pemeter.out_dims.h_size;
        config.s_blk_pemeter.out_dims.blk_cnt  = config.s_blk_pemeter.out_dims.w_cnt  * config.s_blk_pemeter.out_dims.h_cnt;	
}

