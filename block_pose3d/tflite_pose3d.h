/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TFLITE_POSE3D_H_
#define TFLITE_POSE3D_H_


#ifdef __cplusplus
extern "C" {
#endif

#define MAX_POSE_NUM  10

enum pose_key_id {
    kNose = 0,          //  0
    kNeck,              //  1

    kRightShoulder,     //  2
    kRightElbow,        //  3
    kRightWrist,        //  4
    
    kLeftShoulder,      //  5
    kLeftElbow,         //  6
    kLeftWrist,         //  7

    kRightHip,          //  8
    kRightKnee,         //  9
    kRightAnkle,        // 10

    kLeftHip,           // 11
    kLeftKnee,          // 12
    kLeftAnkle,         // 13

    kLeftEye,           // 14
    kRightEye,          // 15
    kLeftEar,           // 16
    kRightEar,          // 17

    kPad,               // 18

    kPoseKeyNum
};

typedef struct fvec2
{
    float x, y;
} fvec2;

typedef struct fvec3
{
    float x, y, z;
} fvec3;


typedef struct _pose_key_t
{
    float x;
    float y;
    float z;
    float score;
} pose_key_t;

typedef struct _pose_t
{
    pose_key_t key  [kPoseKeyNum];
    pose_key_t key3d[kPoseKeyNum];
    float pose_score;

    void *heatmap;
    int   heatmap_dims[2];  /* heatmap resolution. (9x9) */
} pose_t;

typedef struct _posenet_result_t
{
    int num;
    pose_t pose[MAX_POSE_NUM];
} posenet_result_t;



typedef struct _pose3d_config_t
{
    float score_thresh;
    float iou_thresh;
} pose3d_config_t;

typedef struct _result_quality
{
    float curr_mpjpe;
    float avg_mpjpe; // average mpjpe across all frames from 0 to curr
    int   cnt; // frame count from 0 to curr
} result_quality;


int  init_tflite_pose3d (int use_quantized_tflite, pose3d_config_t *config, char* blk_arg);
int  init_trt_pose3d (pose3d_config_t *config/*, char* model_name*/);
void *get_pose3d_input_buf (int *w, int *h);
//void *get_pose3d_input_buf_trt (int *w, int *h);
void *get_pose3d_input_buf_tpu ();
void **get_pose3d_input_buf_blk ();
int invoke_pose3d (posenet_result_t *pose_result);
int invoke_trt_pose3d (posenet_result_t *pose_result);

#if defined (USE_BGT)
int invoke_pose3d_tpu (posenet_result_t *pose_result);
#endif
#if defined (USE_BLK) || defined (USE_BLK_TRT)
int invoke_pose3d_blk (posenet_result_t *pose_result);
void feed_blk_bufs(unsigned char* buf_ui8, float** blk_buf_fp32, int dst_h, int dst_w);
#endif


#ifdef __cplusplus
}
#endif

#endif /* TFLITE_POSE3D_H_ */
