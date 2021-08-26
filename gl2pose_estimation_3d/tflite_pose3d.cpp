/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "util_tflite.h"
#include "tflite_pose3d.h"
#include <float.h>

#define POSENET_MODEL_PATH          "./model/human_pose_estimation_3d_0001_256x448_float.tflite"
#define POSENET_EDGETPU_MODEL_PATH  "./model/human_pose_estimation_3d_0001_256x448_full_integer_quant_edgetpu.tflite"

static tflite_interpreter_t s_interpreter;
static tflite_tensor_t      s_tensor_input;
static tflite_tensor_t      s_tensor_heatmap;
static tflite_tensor_t      s_tensor_offsets;

static int     s_img_w = 0;
static int     s_img_h = 0;
static int     s_hmp_w = 0;
static int     s_hmp_h = 0;

static _result_quality s_result_quality;



/* -------------------------------------------------- *
 *  Create TensorFlow Lite Interpreter
 * -------------------------------------------------- */
int
init_tflite_pose3d (int use_quantized_tflite, pose3d_config_t *config)
{
#if defined (USE_BGT)
    tflite_create_interpreter_from_file (&s_interpreter, POSENET_MODEL_PATH, POSENET_EDGETPU_MODEL_PATH);
#else
    const char *posenet_model;

    if(use_quantized_tflite == 0){
      posenet_model = POSENET_MODEL_PATH;
    }else{
      posenet_model = POSENET_EDGETPU_MODEL_PATH;
    }
    tflite_create_interpreter_from_file (&s_interpreter, posenet_model);
#endif
    tflite_get_tensor_by_name (&s_interpreter, 0, "data",       &s_tensor_input);   /* (1, 256, 448, 3) */
    tflite_get_tensor_by_name (&s_interpreter, 1, "Identity",   &s_tensor_offsets); /* (1,  32,  56, 57) */
    tflite_get_tensor_by_name (&s_interpreter, 1, "Identity_1", &s_tensor_heatmap); /* (1,  32,  56, 19) */

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




static void
get_index_to_pos (int idx_x, int idx_y, int key_id, fvec2 *pos2d, fvec3 *pos3d)
{
    float ofst_x, ofst_y, ofst_z;
    get_offset_vector (&ofst_x, &ofst_y, &ofst_z, idx_y, idx_x, key_id);

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
decode_single_pose (posenet_result_t *pose_result)
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
                float confidence = get_heatmap_score (y, x, i);
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
        get_index_to_pos (idx_x, idx_y, i, &pos2d, &pos3d);

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
        decode_single_pose (pose_result);

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
        decode_single_pose (pose_result);

    pose_result->pose[0].heatmap = s_tensor_heatmap.ptr;
    pose_result->pose[0].heatmap_dims[0] = s_hmp_w;
    pose_result->pose[0].heatmap_dims[1] = s_hmp_h;
    return 0;
}
#endif
