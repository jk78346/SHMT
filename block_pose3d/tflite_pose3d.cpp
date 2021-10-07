/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "util_tflite.h"
#include "util_trt.h"
#include "tflite_pose3d.h"
#include "unistd.h"
#include <float.h>
#include "util_pmeter.h"
#include "time.h"

//#define POSENET_MODEL_PATH            "./model/human_pose_estimation_3d_0001_256x448_float.tflite"
//#define POSENET_MODEL_PATH          "./model/human_pose_estimation_3d_0001_256x448_float16_quant.tflite"
//#define POSENET_EDGETPU_MODEL_PATH  "./model/human_pose_estimation_3d_0001_256x448_integer_quant.tflite"
//#define POSENET_EDGETPU_MODEL_PATH  "./model/human_pose_estimation_3d_0001_256x448_weight_quant.tflite"
#define POSENET_MODEL_PATH          "./model/toy_pose3d_full_model_float.tflite"
#define POSENET_FULL_MODEL_PATH     "./model/toy_pose3d_full_model_float.tflite"
//#define POSENET_BLOCK_MODEL_PATH     "./model/toy_pose3d_full_model_float.tflite"
//#define POSENET_BLOCK_MODEL_PATH    "./model/toy_pose3d_sp_model_blk_8_float.tflite"
#define POSENET_BLOCK_MODEL_PATH    "./model/toy_pose3d_half_model_float.tflite"

//#define POSENET_EDGETPU_MODEL_PATH  "./model/human_pose_estimation_3d_0001_256x448_full_integer_quant_edgetpu.tflite"
#define POSENET_EDGETPU_MODEL_PATH  "./model/toy_pose3d_full_model_float.tflite"

#define UFF_MODEL_PATH  "./model/toy_pose3d_full_model_float.onnx"
#define PLAN_MODEL_PATH "./model/temp.plan"

static tflite_interpreter_t s_interpreter;
static tflite_tensor_t      s_tensor_input;
static tflite_tensor_t      s_tensor_heatmap;
static tflite_tensor_t      s_tensor_offsets;

static int     s_img_w = 0;
static int     s_img_h = 0;
static int     s_hmp_w = 0;
static int     s_hmp_h = 0;

#if defined (USE_TRT)
static IExecutionContext   *s_trt_context;
static trt_tensor_t         s_trt_tensor_input;
static trt_tensor_t         s_trt_tensor_heatmap;
static trt_tensor_t         s_trt_tensor_offsets;
static trt_tensor_t         s_trt_tensor_pafs;

static std::vector<void *>  s_gpu_buffers;
#endif

static _result_quality s_result_quality;
#if defined (USE_BLK)
static _blk_pemeter    s_blk_pemeter;
#endif

#if defined (USE_BLK)
void feed_blk_bufs(unsigned char* buf_ui8, float** blk_buf_fp32, int dst_h, int dst_w){
    int w_cnt    = s_blk_pemeter.w_cnt;
    int h_cnt    = s_blk_pemeter.h_cnt;
    int w_size   = s_blk_pemeter.w_size;
    int h_size   = s_blk_pemeter.h_size;
    int blk_cnt  = s_blk_pemeter.blk_cnt;
    int blk_size = s_blk_pemeter.blk_size;
    
//    if((dst_h != (h_cnt * w_size)) || (dst_w != (w_cnt * h_size))){
//       fprintf(stderr, "%s line %d: dst_w: %d, dst_h: %d, w_cnt: %d, h_cnt: %d, w_size: %d, h_size: %d\n", __func__, __LINE__, dst_w, dst_h, w_cnt, h_cnt, w_size, h_size);
//    }   
    float mean =   0.0f;
    float std  = 255.0f;
    for(int i = 0 ; i < h_cnt ; i++){            // y-axis
      for(int j = 0 ; j < w_cnt ; j++){        // x-axis
          int blk_serial_idx = i*w_cnt +j;
          for(int y = 0 ; y < h_size ; y++){     // y-axis
              for(int x = 0 ; x < w_size ; x++){ // x-axis
//                    printf("blk index: (%d, %d), index within blk: (%d, %d), linear index: %d and the next 3, max boundary: %d\n", j, i, x, y, i*blk_cnt_w+j, dst_h*dst_w*4);
                  int global_x = j*w_size+x;
                  int global_y = i*h_size+y;
		  int r = buf_ui8[(global_y*dst_w+global_x)*4];
                  int g = buf_ui8[(global_y*dst_w+global_x)*4+1];
                  int b = buf_ui8[(global_y*dst_w+global_x)*4+2];
//                  if(i*blk_cnt_w+j >= blk_cnt)
//                      printf("the index of blk_ptrs is: %d: blk_cnt: %d\n", i*blk_cnt_w+j, blk_cnt);
                  blk_buf_fp32[blk_serial_idx][(y*w_size+x)*3] = (float)(r - mean) / std;
                  blk_buf_fp32[blk_serial_idx][(y*w_size+x)*3+1] = (float)(g - mean) / std;
                  blk_buf_fp32[blk_serial_idx][(y*w_size+x)*3+2] = (float)(b - mean) / std;
              }
          }   
      }
    }
    return;
}
#endif

/* -------------------------------------------------- *
 *  Create TensorFlow Lite Interpreter
 * -------------------------------------------------- */
int
init_tflite_pose3d (int use_quantized_tflite, pose3d_config_t *config)
{
#if defined (USE_BLK)
    int blk_size = 8; // The blk_size must be aligned with the block_model
    s_blk_pemeter.w_size   = 224;//blk_size;//224;//blk_size;
    s_blk_pemeter.h_size   = 256;//blk_size;//256;//blk_size;
    s_blk_pemeter.blk_size = s_blk_pemeter.w_size * s_blk_pemeter.h_size;
    s_blk_pemeter.w_cnt    = (448 / s_blk_pemeter.w_size);
    s_blk_pemeter.h_cnt    = (256 / s_blk_pemeter.h_size);
    s_blk_pemeter.blk_cnt  = s_blk_pemeter.w_cnt * s_blk_pemeter.h_cnt;

    s_blk_pemeter.w_size_out   = 28;//blk_size/8; //28;
    s_blk_pemeter.h_size_out   = 32;//blk_size/8; // 32;
    s_blk_pemeter.blk_size_out = s_blk_pemeter.w_size_out * s_blk_pemeter.h_size_out;
    s_blk_pemeter.w_cnt_out    = (56 / s_blk_pemeter.w_size_out);
    s_blk_pemeter.h_cnt_out    = (32 / s_blk_pemeter.h_size_out);
    s_blk_pemeter.blk_cnt_out  = s_blk_pemeter.w_cnt_out * s_blk_pemeter.h_cnt_out;

    tflite_create_interpreter_from_file (&s_interpreter, POSENET_FULL_MODEL_PATH, POSENET_BLOCK_MODEL_PATH, s_blk_pemeter.blk_cnt);

    tflite_get_tensor_by_name_blk (&s_interpreter, 0, "data",     &s_tensor_input, &s_blk_pemeter);   /* (1, 256, 448, 3) */
    tflite_get_tensor_by_name_blk (&s_interpreter, 1, "Identity/Conv2D",   &s_tensor_offsets, &s_blk_pemeter); /* (1,  32,  56, 57) */
    tflite_get_tensor_by_name_blk (&s_interpreter, 1, "Identity_1/Conv2D", &s_tensor_heatmap, &s_blk_pemeter); /* (1,  32,  56, 19) */
#else
    const char *posenet_model;

    if(use_quantized_tflite == 0){
      posenet_model = POSENET_MODEL_PATH;
    }else{
      posenet_model = POSENET_EDGETPU_MODEL_PATH;
    }
    tflite_create_interpreter_from_file (&s_interpreter, posenet_model);

    tflite_get_tensor_by_name (&s_interpreter, 0, "data",       &s_tensor_input);   /* (1, 256, 448, 3) */
    tflite_get_tensor_by_name (&s_interpreter, 1, "Identity/Conv2D",   &s_tensor_offsets); /* (1,  32,  56, 57) */
    tflite_get_tensor_by_name (&s_interpreter, 1, "Identity_1/Conv2D", &s_tensor_heatmap); /* (1,  32,  56, 19) */

#endif

    /* input image dimention */
    s_img_w = s_tensor_input.dims[2];
    s_img_h = s_tensor_input.dims[1];
    fprintf (stderr, "input image size: (%d, %d)\n", s_img_w, s_img_h);

    /* heatmap dimention */
    s_hmp_w = s_tensor_heatmap.dims[2];
    s_hmp_h = s_tensor_heatmap.dims[1];
    fprintf (stderr, "heatmap size: (%d, %d)\n", s_hmp_w, s_hmp_h);

    return 0;
}

#if defined (USE_TRT)
/* -------------------------------------------------- *
 *  create cuda engine
 * -------------------------------------------------- */
static int
convert_onnx_to_plan (const std::string &plan_file_name, const std::string &uff_file_name)
{
    ICudaEngine *engine;
    engine = trt_create_engine_from_onnx (uff_file_name);
    if (!engine)
    {
        fprintf (stderr, "ERR:%s(%d): Failed to load graph from file.\n", __FILE__, __LINE__);
        return -1;
    }

    trt_emit_plan_file (engine, plan_file_name);

    engine->destroy();

    return 0;
}

/* -------------------------------------------------- *
 *  Create TensorRT Interpreter
 * -------------------------------------------------- */
int
init_trt_pose3d (pose3d_config_t *config/*, char* model_name*/)
{
    ICudaEngine *engine = NULL;

    trt_initialize ();

    /* Try to load Prebuilt TensorRT Engine */
    fprintf (stderr, "loading prebuilt TensorRT engine...\n");
    engine = trt_load_plan_file (PLAN_MODEL_PATH);

    /* Build TensorRT Engine */
    if (engine == NULL)
    {
        convert_onnx_to_plan (PLAN_MODEL_PATH, /*model_name*/UFF_MODEL_PATH);

        engine = trt_load_plan_file (PLAN_MODEL_PATH);
        if (engine == NULL)
        {
            fprintf (stderr, "%s(%d)\n", __FILE__, __LINE__);
            return -1;
        }
    }

    s_trt_context = engine->createExecutionContext();


    /* Allocate IO tensors */
    trt_get_tensor_by_name (engine, "data",       &s_trt_tensor_input);   /* (1, 256, 448,  3) */
    trt_get_tensor_by_name (engine, "Identity/Conv2D",   &s_trt_tensor_offsets); /* (1,  32,  56, 57) */
    trt_get_tensor_by_name (engine, "Identity_1/Conv2D", &s_trt_tensor_heatmap); /* (1,  32,  56  19) */
//    trt_get_tensor_by_name (engine, "Identity_2:0", &s_trt_tensor_pafs);    /* (1,  32,  56, 38) */

    int num_bindings = engine->getNbBindings();
    s_gpu_buffers.resize (num_bindings);
    s_gpu_buffers[s_trt_tensor_input  .bind_idx] = s_trt_tensor_input  .gpu_mem;
    s_gpu_buffers[s_trt_tensor_heatmap.bind_idx] = s_trt_tensor_heatmap.gpu_mem;
    s_gpu_buffers[s_trt_tensor_offsets.bind_idx] = s_trt_tensor_offsets.gpu_mem;
//    s_gpu_buffers[s_trt_tensor_pafs   .bind_idx] = s_trt_tensor_pafs   .gpu_mem;

    config->score_thresh = 0.3f;
    config->iou_thresh   = 0.3f;

    /* input image dimention */
    s_img_w = s_trt_tensor_input.dims.d[2];
    s_img_h = s_trt_tensor_input.dims.d[1];
    fprintf (stderr, "input image size: (%d, %d)\n", s_img_w, s_img_h);

    /* heatmap dimention */
    s_hmp_w = s_trt_tensor_heatmap.dims.d[2];
    s_hmp_h = s_trt_tensor_heatmap.dims.d[1];
    fprintf (stderr, "heatmap size: (%d, %d)\n", s_hmp_w, s_hmp_h);

    return 0;
}
#endif

void *
get_pose3d_input_buf (int *w, int *h)
{
#if defined (USE_TRT)
    *w = s_trt_tensor_input.dims.d[2];
    *h = s_trt_tensor_input.dims.d[1];
    return s_trt_tensor_input.cpu_mem;
#else
    *w = s_tensor_input.dims[2];
    *h = s_tensor_input.dims[1];
    return s_tensor_input.ptr;
#endif
}

#if defined (USE_BGT)
void *
get_pose3d_input_buf_tpu()
{
    return s_tensor_input.tpu_ptr;
}
#endif

#if defined (USE_BLK)
void **
get_pose3d_input_buf_blk()
{
    return s_tensor_input.blk_ptrs;
}
#endif

/* -------------------------------------------------- *
 * Invoke TensorFlow Lite
 * -------------------------------------------------- */
static float
get_heatmap_score (int idx_y, int idx_x, int key_id)
{
    int idx = (idx_y * s_hmp_w * kPoseKeyNum) + (idx_x * kPoseKeyNum) + key_id;
#if defined (USE_TRT)
    float *heatmap_ptr = (float *)s_trt_tensor_heatmap.cpu_mem;
#else
    float *heatmap_ptr = (float *)s_tensor_heatmap.ptr;
#endif
    return heatmap_ptr[idx];
}

#if defined (USE_BGT)
static float
get_heatmap_score_tpu (int idx_y, int idx_x, int key_id)
{
    int idx = (idx_y * s_hmp_w * kPoseKeyNum) + (idx_x * kPoseKeyNum) + key_id;
    float *heatmap_ptr = (float *)s_tensor_heatmap.tpu_ptr;
    return heatmap_ptr[idx];
}
#endif

#if defined (USE_BLK)
static float
get_heatmap_score_blk (int idx_y, int idx_x, int key_id, int blk_id)
{
//	printf("idx_x: %d, idx_y: %d, key_id: %d, blk_id: %d\n", idx_x, idx_y, key_id, blk_id);
// TODO: need to calculate output shape
    int idx = (idx_y * s_blk_pemeter.w_size_out/*s_hmp_w*/ * kPoseKeyNum) + (idx_x * kPoseKeyNum) + key_id;
    float *heatmap_ptr = (float *)s_tensor_heatmap.blk_ptrs[blk_id];
    return heatmap_ptr[idx];
}
#endif

static void
get_offset_vector (float *ofst_x, float *ofst_y, float *ofst_z, int idx_y, int idx_x, int pose_id_)
{
    int map_id_to_panoptic[] = {1, 0,  9, 10, 11, 3, 4, 5, 12, 13, 14, 6, 7, 8, 15, 16, 17, 18, 2};
    int pose_id = map_id_to_panoptic[pose_id_];
    int idx0 = (idx_y * s_hmp_w * kPoseKeyNum*3) + (idx_x * kPoseKeyNum*3) + (3 * pose_id + 0);
    int idx1 = (idx_y * s_hmp_w * kPoseKeyNum*3) + (idx_x * kPoseKeyNum*3) + (3 * pose_id + 1);
    int idx2 = (idx_y * s_hmp_w * kPoseKeyNum*3) + (idx_x * kPoseKeyNum*3) + (3 * pose_id + 2);
#if defined (USE_TRT)
    float *offsets_ptr = (float *)s_trt_tensor_offsets.cpu_mem;
#else
    float *offsets_ptr = (float *)s_tensor_offsets.ptr;
#endif
    *ofst_x = offsets_ptr[idx0];
    *ofst_y = offsets_ptr[idx1];
    *ofst_z = offsets_ptr[idx2];
}

#if defined (USE_BGT)
static void
get_offset_vector_tpu (float *ofst_x, float *ofst_y, float *ofst_z, int idx_y, int idx_x, int pose_id_)
{
    int map_id_to_panoptic[] = {1, 0,  9, 10, 11, 3, 4, 5, 12, 13, 14, 6, 7, 8, 15, 16, 17, 18, 2};
    int pose_id = map_id_to_panoptic[pose_id_];

    int idx0 = (idx_y * s_hmp_w * kPoseKeyNum*3) + (idx_x * kPoseKeyNum*3) + (3 * pose_id + 0);
    int idx1 = (idx_y * s_hmp_w * kPoseKeyNum*3) + (idx_x * kPoseKeyNum*3) + (3 * pose_id + 1);
    int idx2 = (idx_y * s_hmp_w * kPoseKeyNum*3) + (idx_x * kPoseKeyNum*3) + (3 * pose_id + 2);

    float *offsets_ptr = (float *)s_tensor_offsets.tpu_ptr;

    *ofst_x = offsets_ptr[idx0];
    *ofst_y = offsets_ptr[idx1];
    *ofst_z = offsets_ptr[idx2];
}
#endif

#if defined (USE_BLK)
static void
get_offset_vector_blk (float *ofst_x, float *ofst_y, float *ofst_z, int idx_y, int idx_x, int pose_id_)
{
    int map_id_to_panoptic[] = {1, 0,  9, 10, 11, 3, 4, 5, 12, 13, 14, 6, 7, 8, 15, 16, 17, 18, 2};
    int pose_id = map_id_to_panoptic[pose_id_];

// use global idx_x, idx_y to calculate local idx_x, idx_y on blk with id: blk_id
    int blk_id = -1;
    int local_idx_x = idx_x % s_blk_pemeter.w_size_out; 
    int local_idx_y = idx_y % s_blk_pemeter.h_size_out;

    int blk_id_x = idx_x / s_blk_pemeter.w_size_out;
    int blk_id_y = idx_y / s_blk_pemeter.h_size_out;
    blk_id = blk_id_y * s_blk_pemeter.h_cnt_out + blk_id_x;
// end calculation
//    printf("w_size_out: %d, h_size_out: %d\n", s_blk_pemeter.w_size_out, s_blk_pemeter.h_size_out);
//    printf("global: x: %d, y: %d\t local: x: %d, y: %d, blk_id_x: %d, blk_id_y: %d, blk_id: %d\n", idx_x, idx_y, local_idx_x, local_idx_y, blk_id_x, blk_id_y, blk_id);
    int idx0 = (local_idx_y * s_blk_pemeter.w_size_out/*s_hmp_w*/ * kPoseKeyNum*3) + (local_idx_x * kPoseKeyNum*3) + (3 * pose_id + 0);
    int idx1 = (local_idx_y * s_blk_pemeter.w_size_out/*s_hmp_w*/ * kPoseKeyNum*3) + (local_idx_x * kPoseKeyNum*3) + (3 * pose_id + 1);
    int idx2 = (local_idx_y * s_blk_pemeter.w_size_out/*s_hmp_w*/ * kPoseKeyNum*3) + (local_idx_x * kPoseKeyNum*3) + (3 * pose_id + 2);

    float *offsets_ptr = (float *)s_tensor_offsets.blk_ptrs[blk_id];

    *ofst_x = offsets_ptr[idx0];
    *ofst_y = offsets_ptr[idx1];
    *ofst_z = offsets_ptr[idx2];
}
#endif

static void
get_index_to_pos (int idx_x, int idx_y, int key_id, fvec2 *pos2d, fvec3 *pos3d, int device, int blocking)
{
    float ofst_x, ofst_y, ofst_z;
    if(blocking == 0){
#if defined (USE_BGT)
        if(device == 0){
    	    get_offset_vector (&ofst_x, &ofst_y, &ofst_z, idx_y, idx_x, key_id);
        }else{
            get_offset_vector_tpu (&ofst_x, &ofst_y, &ofst_z, idx_y, idx_x, key_id);
        }
#else
        get_offset_vector (&ofst_x, &ofst_y, &ofst_z, idx_y, idx_x, key_id);
#endif    
    }else{
#if defined (USE_BLK)
        get_offset_vector_blk (&ofst_x, &ofst_y, &ofst_z, idx_y, idx_x, key_id);
#else
	printf("USE_BLK is not defined but blk API is used. ERROR\n");
	exit(0);
#endif
    }


    /* pos 2D */
    pos2d->x = (float)idx_x / (float)(s_hmp_w -1);
    pos2d->y = (float)idx_y / (float)(s_hmp_h -1);

    /* pos 3D */
    pos3d->x = ofst_x;
    pos3d->y = ofst_y;
    pos3d->z = ofst_z;
}

//float x_max = FLT_MIN, x_min = FLT_MAX, y_max = FLT_MIN, y_min = FLT_MAX, z_max = FLT_MIN, z_min = FLT_MAX;

void mpjpe(pose_t target, pose_t pred)
{
    float sum = 0;
    for(int i = 0 ; i < kPoseKeyNum ; i++)
    {
        sum += sqrt(pow(pred.key3d[i].x - target.key3d[i].x, 2.0) + pow(pred.key3d[i].y - target.key3d[i].y, 2.0) + pow(pred.key3d[i].z - target.key3d[i].z, 2.0));
        //printf("[target]: (%f, %f, %f)\n", target.key3d[i].x, target.key3d[i].y, target.key3d[i].z);
	//if( target.key3d[i].x > x_max ){ x_max = target.key3d[i].x; }
	//if( target.key3d[i].x < x_min ){ x_min = target.key3d[i].x; }
	//if( target.key3d[i].y > y_max ){ y_max = target.key3d[i].y; }
	//if( target.key3d[i].y < y_min ){ y_min = target.key3d[i].y; }
	//if( target.key3d[i].z > z_max ){ z_max = target.key3d[i].z; }
	//if( target.key3d[i].z < z_min ){ z_min = target.key3d[i].z; }
        //printf("[target]: (%f~%f, %f~%f, %f~%f)\n", x_max, x_min, y_max, y_min, z_max, z_min);
    }

    float mpjpe     = s_result_quality.curr_mpjpe = sum/kPoseKeyNum;
    int cnt         = s_result_quality.cnt++;
    float avg_mpjpe = s_result_quality.avg_mpjpe  = (s_result_quality.avg_mpjpe * cnt + mpjpe) / (cnt + 1);    

    printf("curr mpjpe: %f, avg mpjpe: %f\n", mpjpe, avg_mpjpe);
}


static void
decode_multiple_poses (posenet_result_t *pose_result)
{
    memset (pose_result, 0, sizeof (posenet_result_t));
}

static void
decode_single_pose (posenet_result_t *pose_result, int device)
{
    int   max_block_idx[kPoseKeyNum][2] = {0};
    float max_block_cnf[kPoseKeyNum]    = {0};

    /* find the highest heatmap block for each key */
    for (int i = 0; i < kPoseKeyNum; i ++)
    {
        float max_confidence = -FLT_MAX;
	for (int y = 0; y < s_hmp_h; y ++)
        {
            for (int x = 0; x < s_hmp_w; x ++)
            {
#if defined (USE_BGT)
    		    float confidence = (device == 0)?get_heatmap_score (y, x, i):get_heatmap_score_tpu (y, x, i);
#else
    		    float confidence = get_heatmap_score (y, x, i);
#endif
    		    if (confidence > max_confidence)
                {
                    max_confidence = confidence;
                    max_block_cnf[i] = confidence;
                    max_block_idx[i][0] = x;
                    max_block_idx[i][1] = y;
                }
            }
        }
    }
//    for(int i = 0 ; i < kPoseKeyNum ; i++){
//    	printf("max_block_cnf[%2d]   : %f\t", i, max_block_cnf[i]);
//    	printf("max_block_idx[%2d][0]: %d\t", i, max_block_idx[i][0]);
//    	printf("max_block_idx[%2d][1]: %d\n", i, max_block_idx[i][1]);
//    }
#if 0
    for (int i = 0; i < kPoseKeyNum; i ++)
    {
        fprintf (stderr, "---------[%d] --------\n", i);
        for (int y = 0; y < s_hmp_h; y ++)
        {
            fprintf (stderr, "[%d] ", y);
            for (int x = 0; x < s_hmp_w; x ++)
            {
                float confidence = get_heatmap_score (y, x, i);
                fprintf (stderr, "%6.3f ", confidence);

                if (x == max_block_idx[i][0] && y == max_block_idx[i][1])
                    fprintf (stderr, "#");
                else
                    fprintf (stderr, " ");
            }
            fprintf (stderr, "\n");
        }
    }
#endif

    /* find the offset vector and calculate the keypoint coordinates. */
    for (int i = 0; i < kPoseKeyNum;i ++ )
    {
        int idx_x = max_block_idx[i][0];
        int idx_y = max_block_idx[i][1];
        fvec2 pos2d;
        fvec3 pos3d;
	
	get_index_to_pos (idx_x, idx_y, i, &pos2d, &pos3d, device, 0/*not blk algorithm*/);

        pose_result->pose[0].key[i].x     = pos2d.x;
        pose_result->pose[0].key[i].y     = pos2d.y;
        pose_result->pose[0].key[i].score = max_block_cnf[i];

//	printf("[%2d] x: %f, y: %f, score: %f\n", i, pos2d.x, pos2d.y, max_block_cnf[i]);

        pose_result->pose[0].key3d[i].x   = pos3d.x;
        pose_result->pose[0].key3d[i].y   = pos3d.y;
        pose_result->pose[0].key3d[i].z   = pos3d.z;
        pose_result->pose[0].key3d[i].score = max_block_cnf[i];
    }
    pose_result->num = 1;
    pose_result->pose[0].pose_score = 1.0f;

    /* eval pose result. */
    //mpjpe(pose_result->pose[0], pose_result->tpu_pose[0]);

}

int   blk_max_block_idx[kPoseKeyNum][2] = {0};
float blk_max_block_cnf[kPoseKeyNum]    = {0};

#if defined (USE_BLK)
static void
decode_single_pose_blk (posenet_result_t *pose_result, int device, int blk_id)
{
    int g_w_id, g_h_id;
    get_blk_coordinates(s_blk_pemeter.w_cnt, s_blk_pemeter.h_cnt, s_blk_pemeter.w_size, blk_id, g_w_id, g_h_id); 
    if(blk_id == 0){ // first block is responsible for resetting thevalue
        for(int i = 0 ; i < kPoseKeyNum ; i++){
            blk_max_block_idx[i][0] = 0;
            blk_max_block_idx[i][1] = 0;
	    blk_max_block_cnf[i]    = 0;
	}
    }
    /* find the highest heatmap block for each key */
    for (int i = 0; i < kPoseKeyNum; i ++)
    {
        float max_confidence = -FLT_MAX;
	for (int y = 0; y < s_blk_pemeter.h_size_out/*s_hmp_h*/; y ++)
        {
            for (int x = 0; x < s_blk_pemeter.w_size_out/*s_hmp_w*/; x ++)
            {
#if defined (USE_BGT)
    		    float confidence = (device == 0)?get_heatmap_score (y, x, i):get_heatmap_score_tpu (y, x, i);
#elif defined (USE_BLK)
    		    float confidence = get_heatmap_score_blk (y, x, i, blk_id);
#else
    		    float confidence = get_heatmap_score (y, x, i);
#endif
    		    if (confidence > max_confidence)
                {
                    max_confidence = confidence;
                    blk_max_block_cnf[i] = confidence;
                    /*printf("g_w_id: %d, w_size: %d, x: %d | g_h_id: %d, h_size: %d, y: %d\n", g_w_id,
				    					  		      s_blk_pemeter.w_size,
											      x,
											      g_h_id, 
											      s_blk_pemeter.h_size,
											      y);
		    */
		    blk_max_block_idx[i][0] = g_w_id*s_blk_pemeter.w_size_out+x;
		    blk_max_block_idx[i][1] = g_h_id*s_blk_pemeter.h_size_out+y;
                }
            }
        }
    }
//    for(int i = 0 ; i < kPoseKeyNum ; i++){
//    	printf("max_block_cnf[%2d]   : %f\t", i, blk_max_block_cnf[i]);
//    	printf("max_block_idx[%2d][0]: %d\t", i, blk_max_block_idx[i][0]);
//    	printf("max_block_idx[%2d][1]: %d\n", i, blk_max_block_idx[i][1]);
//    }
//    printf("\n");

    if(blk_id == s_blk_pemeter.blk_cnt - 1){ // the last block is responsible for summrize the output
    /* find the offset vector and calculate the keypoint coordinates. */
   	 for (int i = 0; i < kPoseKeyNum;i ++ )
   	 {
       	 	int idx_x = blk_max_block_idx[i][0];
     	 	int idx_y = blk_max_block_idx[i][1];
  		fvec2 pos2d;
        	fvec3 pos3d;

		get_index_to_pos (idx_x, idx_y, i, &pos2d, &pos3d, device, 1/*is blk algo.*/);

        	pose_result->pose[0].key[i].x     = pos2d.x;
        	pose_result->pose[0].key[i].y     = pos2d.y;
        	pose_result->pose[0].key[i].score = blk_max_block_cnf[i];

//		printf("[%2d] x: %f, y: %f, score: %f\n", i, pos2d.x, pos2d.y, max_block_cnf[i]);
        	
		pose_result->pose[0].key3d[i].x   = pos3d.x;
        	pose_result->pose[0].key3d[i].y   = pos3d.y;
        	pose_result->pose[0].key3d[i].z   = pos3d.z;
        	pose_result->pose[0].key3d[i].score = blk_max_block_cnf[i];
   	 }
	 pose_result->num = 1;
     	 pose_result->pose[0].pose_score = 1.0f;
    }
}
#endif
/* -------------------------------------------------- *
 * Invoke TensorFlow Lite
 * -------------------------------------------------- */
double ttime[10] = {0};
double baseline_invoke_ms = 0;
double blk_invoke_ms = 0;
double baseline_invoke_avg_ms = 0;
double blk_invoke_avg_ms = 0;
int baseline_cnt = 0;
int blk_cnt = 0;
int
invoke_pose3d (posenet_result_t *pose_result)
{
#if defined (USE_TRT)
    /* copy to CUDA buffer */
    trt_copy_tensor_to_gpu (s_trt_tensor_input);

    /* invoke inference */
    int batchSize = 1;
    s_trt_context->execute (batchSize, &s_gpu_buffers[0]);

    /* copy from CUDA buffer */
    trt_copy_tensor_from_gpu (s_trt_tensor_heatmap);
    trt_copy_tensor_from_gpu (s_trt_tensor_offsets);
//    trt_copy_tensor_from_gpu (s_trt_tensor_pafs);

    if (0)
        decode_multiple_poses (pose_result);
    else
        decode_single_pose (pose_result, 0);

    pose_result->pose[0].heatmap = s_trt_tensor_heatmap.cpu_mem;
#else
    ttime[0] = pmeter_get_time_ms();
    if (s_interpreter.interpreter->Invoke() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }
    ttime[1] = pmeter_get_time_ms();
    baseline_invoke_ms = ttime[1] - ttime[0];
    baseline_invoke_avg_ms = (baseline_invoke_avg_ms * baseline_cnt + baseline_invoke_ms) / (baseline_cnt + 1);
    baseline_cnt++;
    if(baseline_cnt >= 100){
    	printf("baseline invoke avg ms: %f\n", baseline_invoke_avg_ms);
    }
    if (0)
        decode_multiple_poses (pose_result);
    else
        decode_single_pose (pose_result, 0);

    pose_result->pose[0].heatmap = s_tensor_heatmap.ptr;
#endif
    pose_result->pose[0].heatmap_dims[0] = s_hmp_w;
    pose_result->pose[0].heatmap_dims[1] = s_hmp_h;
    return 0;
}


#if defined (USE_BGT)
int
invoke_pose3d_tpu (posenet_result_t *pose_result)
{
    if (s_interpreter.tpu_interpreter->Invoke() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }
    if (0)
        decode_multiple_poses (pose_result);
    else
        decode_single_pose (pose_result, 1);

    pose_result->pose[0].heatmap = s_tensor_heatmap.tpu_ptr;
    pose_result->pose[0].heatmap_dims[0] = s_hmp_w;
    pose_result->pose[0].heatmap_dims[1] = s_hmp_h;
    return 0;
}
#endif

#if defined (USE_BLK)
int
invoke_pose3d_blk (posenet_result_t *pose_result)
{
//    printf("%s: blk_cnt: h:%d x w:%d = %d\n", __func__, s_blk_pemeter.h_cnt, s_blk_pemeter.w_cnt, s_blk_pemeter.blk_cnt);
    blk_invoke_ms = 0;
    for(int i = 0 ; i < s_blk_pemeter.blk_cnt ; i++){
    	ttime[0] = pmeter_get_time_ms();
	if (s_interpreter.blk_interpreters[i]->Invoke() != kTfLiteOk)
        {
            fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
            return -1;
        }
    	ttime[1] = pmeter_get_time_ms();
        blk_invoke_ms += ttime[1] - ttime[0];
	if (0)
	    decode_multiple_poses (pose_result);
	else
	    decode_single_pose_blk (pose_result, 1, i);
    }
    blk_invoke_avg_ms = (blk_invoke_avg_ms * blk_cnt + blk_invoke_ms) / (blk_cnt + 1);
    blk_cnt++;
    if(blk_cnt >= 100){
    	printf("blk invoke avg ms: %f\n", blk_invoke_avg_ms);
    }

    pose_result->pose[0].heatmap = s_tensor_heatmap.ptr;
    pose_result->pose[0].heatmap_dims[0] = s_hmp_w;
    pose_result->pose[0].heatmap_dims[1] = s_hmp_h;
    return 0;
}
#endif

