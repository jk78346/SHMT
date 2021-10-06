/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "util_tflite.h"
#include "tflite_pose3d.h"
#include <float.h>

//#define POSENET_MODEL_PATH            "./model/human_pose_estimation_3d_0001_256x448_float.tflite"
//#define POSENET_MODEL_PATH          "./model/human_pose_estimation_3d_0001_256x448_float16_quant.tflite"
//#define POSENET_EDGETPU_MODEL_PATH  "./model/human_pose_estimation_3d_0001_256x448_integer_quant.tflite"
//#define POSENET_EDGETPU_MODEL_PATH  "./model/human_pose_estimation_3d_0001_256x448_weight_quant.tflite"
#define POSENET_MODEL_PATH          "./model/toy_pose3d_full_model_float.tflite"
#define POSENET_FULL_MODEL_PATH     "./model/toy_pose3d_full_model_float.tflite"
#define POSENET_BLOCK_MODEL_PATH    "./model/toy_pose3d_sp_model_blk_64_float.tflite"
//#define POSENET_BLOCK_MODEL_PATH    "./model/toy_pose3d_half_model_float.tflite"
//#define POSENET_EDGETPU_MODEL_PATH  "./model/human_pose_estimation_3d_0001_256x448_full_integer_quant_edgetpu.tflite"
#define POSENET_EDGETPU_MODEL_PATH  "./model/toy_pose3d_full_model_float.tflite"

static tflite_interpreter_t s_interpreter;
static tflite_tensor_t      s_tensor_input;
static tflite_tensor_t      s_tensor_heatmap;
static tflite_tensor_t      s_tensor_offsets;

static int     s_img_w = 0;
static int     s_img_h = 0;
static int     s_hmp_w = 0;
static int     s_hmp_h = 0;

static _result_quality s_result_quality;
static _blk_pemeter    s_blk_pemeter;

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
    int blk_size = 64; // The blk_size must be aligned with the block_model
    s_blk_pemeter.w_size   = blk_size; //224;//blk_size;
    s_blk_pemeter.h_size   = blk_size; //256;//blk_size;
    s_blk_pemeter.blk_size = s_blk_pemeter.w_size * s_blk_pemeter.h_size;
    s_blk_pemeter.w_cnt    = (448 / s_blk_pemeter.w_size);
    s_blk_pemeter.h_cnt    = (256 / s_blk_pemeter.h_size);
    s_blk_pemeter.blk_cnt  = s_blk_pemeter.w_cnt * s_blk_pemeter.h_cnt;
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

void *
get_pose3d_input_buf (int *w, int *h)
{
    *w = s_tensor_input.dims[2];
    *h = s_tensor_input.dims[1];
    return s_tensor_input.ptr;
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
    float *heatmap_ptr = (float *)s_tensor_heatmap.ptr;
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
    int idx = (idx_y * s_blk_pemeter.w_size/*s_hmp_w*/ * kPoseKeyNum) + (idx_x * kPoseKeyNum) + key_id;
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

    float *offsets_ptr = (float *)s_tensor_offsets.ptr;

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
    int local_idx_x = idx_x % s_blk_pemeter.w_size; 
    int local_idx_y = idx_y % s_blk_pemeter.h_size;

    int blk_id_x = idx_x / s_blk_pemeter.w_size;
    int blk_id_y = idx_y / s_blk_pemeter.h_size;
    blk_id = blk_id_y * s_blk_pemeter.h_cnt + blk_id_x;
// end calculation
    //printf("global: x: %d, y: %d\t local: x: %d, y: %d, blk_id_x: %d, blk_id_y: %d, blk_id: %d\n", idx_x, idx_y, local_idx_x, local_idx_y, blk_id_x, blk_id_y, blk_id);
    int idx0 = (local_idx_y * s_blk_pemeter.w_size/*s_hmp_w*/ * kPoseKeyNum*3) + (local_idx_x * kPoseKeyNum*3) + (3 * pose_id + 0);
    int idx1 = (local_idx_y * s_blk_pemeter.w_size/*s_hmp_w*/ * kPoseKeyNum*3) + (local_idx_x * kPoseKeyNum*3) + (3 * pose_id + 1);
    int idx2 = (local_idx_y * s_blk_pemeter.w_size/*s_hmp_w*/ * kPoseKeyNum*3) + (local_idx_x * kPoseKeyNum*3) + (3 * pose_id + 2);

    float *offsets_ptr = (float *)s_tensor_offsets.blk_ptrs[blk_id];

    *ofst_x = offsets_ptr[idx0];
    *ofst_y = offsets_ptr[idx1];
    *ofst_z = offsets_ptr[idx2];
}
#endif

static void
get_index_to_pos (int idx_x, int idx_y, int key_id, fvec2 *pos2d, fvec3 *pos3d, int device)
{
    float ofst_x, ofst_y, ofst_z;
#if defined (USE_BGT)
    if(device == 0){
    	get_offset_vector (&ofst_x, &ofst_y, &ofst_z, idx_y, idx_x, key_id);
    }else{
        get_offset_vector_tpu (&ofst_x, &ofst_y, &ofst_z, idx_y, idx_x, key_id);
    }
#elif defined (USE_BLK)
    get_offset_vector_blk (&ofst_x, &ofst_y, &ofst_z, idx_y, idx_x, key_id);
#else
    get_offset_vector (&ofst_x, &ofst_y, &ofst_z, idx_y, idx_x, key_id);
#endif
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

	get_index_to_pos (idx_x, idx_y, i, &pos2d, &pos3d, device);

        pose_result->pose[0].key[i].x     = pos2d.x;
        pose_result->pose[0].key[i].y     = pos2d.y;
        pose_result->pose[0].key[i].score = max_block_cnf[i];

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

int   max_block_idx[kPoseKeyNum][2] = {0};
float max_block_cnf[kPoseKeyNum]    = {0};

static void
decode_single_pose_blk (posenet_result_t *pose_result, int device, int blk_id)
{
    int g_w_id, g_h_id;
    get_blk_coordinates(s_blk_pemeter.w_cnt, s_blk_pemeter.h_cnt, s_blk_pemeter.w_size, blk_id, g_w_id, g_h_id); 
    if(blk_id == 0){ // first block is responsible for resetting thevalue
        for(int i = 0 ; i < kPoseKeyNum ; i++){
            max_block_idx[i][0] = 0;
            max_block_idx[i][1] = 0;
	    max_block_cnf[i]    = 0;
	}
    }
    /* find the highest heatmap block for each key */
    for (int i = 0; i < kPoseKeyNum; i ++)
    {
        float max_confidence = -FLT_MAX;
	for (int y = 0; y < s_blk_pemeter.h_size/*s_hmp_h*/; y ++)
        {
            for (int x = 0; x < s_blk_pemeter.w_size/*s_hmp_w*/; x ++)
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
                    max_block_cnf[i] = confidence;
                    max_block_idx[i][0] = s_blk_pemeter.w_cnt*s_blk_pemeter.w_size+x;
                    max_block_idx[i][1] = s_blk_pemeter.h_cnt*s_blk_pemeter.h_size+y;
                }
            }
        }
    }

    if(blk_id == s_blk_pemeter.blk_cnt - 1){ // the last block is responsible for summrize the output
    /* find the offset vector and calculate the keypoint coordinates. */
   	 for (int i = 0; i < kPoseKeyNum;i ++ )
   	 {
       	 	int idx_x = max_block_idx[i][0];
     	 	int idx_y = max_block_idx[i][1];
  		fvec2 pos2d;
        	fvec3 pos3d;

		get_index_to_pos (idx_x, idx_y, i, &pos2d, &pos3d, device);

        	pose_result->pose[0].key[i].x     = pos2d.x;
        	pose_result->pose[0].key[i].y     = pos2d.y;
        	pose_result->pose[0].key[i].score = max_block_cnf[i];

        	pose_result->pose[0].key3d[i].x   = pos3d.x;
        	pose_result->pose[0].key3d[i].y   = pos3d.y;
        	pose_result->pose[0].key3d[i].z   = pos3d.z;
        	pose_result->pose[0].key3d[i].score = max_block_cnf[i];
   	 }
    	pose_result->num = 1;
    	pose_result->pose[0].pose_score = 1.0f;
    }
}

/* -------------------------------------------------- *
 * Invoke TensorFlow Lite
 * -------------------------------------------------- */
int
invoke_pose3d (posenet_result_t *pose_result)
{
    if (s_interpreter.interpreter->Invoke() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }
    if (0)
        decode_multiple_poses (pose_result);
    else
        decode_single_pose (pose_result, 0);

    pose_result->pose[0].heatmap = s_tensor_heatmap.ptr;
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
    for(int i = 0 ; i < s_blk_pemeter.blk_cnt ; i++){
        if (s_interpreter.blk_interpreters[i]->Invoke() != kTfLiteOk)
        {
            fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
            return -1;
        }
	if (0)
	    decode_multiple_poses (pose_result);
	else
	    decode_single_pose_blk (pose_result, 1, i);
    }

    pose_result->pose[0].heatmap = s_tensor_heatmap.ptr;
    pose_result->pose[0].heatmap_dims[0] = s_hmp_w;
    pose_result->pose[0].heatmap_dims[1] = s_hmp_h;
    return 0;
}
#endif
