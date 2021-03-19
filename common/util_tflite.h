/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef _UTIL_TFLITE_H_
#define _UTIL_TFLITE_H_

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#if defined (USE_GL_DELEGATE)
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"
#endif
#if defined (USE_GPU_DELEGATEV2) || defined (USE_BGT)
#include "tensorflow/lite/delegates/gpu/delegate.h"
#endif

#if defined (USE_NNAPI_DELEGATE)
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#endif

#if defined (USE_HEXAGON_DELEGATE)
#include "tensorflow/lite/experimental/delegates/hexagon/hexagon_delegate.h"
#endif

#if defined (USE_XNNPACK_DELEGATE)
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#endif

#if defined (USE_EDGETPU) || defined (USE_BGT)
#include "edgetpu.h"
//#include "tensorflow/lite/interpreter.h"
#endif

typedef struct tflite_interpreter_t
{
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter>     interpreter;
    tflite::ops::builtin::BuiltinOpResolver  resolver;
#if defined (USE_BGT)
    // second set of interpreter for edgeTPU besides GPU
    std::unique_ptr<tflite::FlatBufferModel> tpu_model;
    std::unique_ptr<tflite::Interpreter>     tpu_interpreter;
    tflite::ops::builtin::BuiltinOpResolver  tpu_resolver;
#endif
} tflite_interpreter_t;

typedef struct tflite_createopt_t
{
    int gpubuffer;
} tflite_createopt_t;

typedef struct tflite_tensor_t
{
    int         idx;        /* whole  tensor index */
    int         io;         /* [0] input_tensor, [1] output_tensor */
    int         io_idx;     /* in/out tensor index */
    TfLiteType  type;       /* [1] kTfLiteFloat32, [2] kTfLiteInt32, [3] kTfLiteUInt8 */
    void        *ptr;
#if defined (USE_BGT)
    void        *tpu_ptr;
#endif
    int         dims[4];
    float       quant_scale;
    int         quant_zerop;
} tflite_tensor_t;


#ifdef __cplusplus
extern "C" {
#endif

int tflite_create_interpreter (tflite_interpreter_t *p, const char *model_buf, size_t model_size);
int tflite_get_tensor_by_name (tflite_interpreter_t *p, int io, const char *name, tflite_tensor_t *ptensor);

#if defined (USE_BGT)
int tflite_create_interpreter_from_file (tflite_interpreter_t *p, const char *model_path, const char *tpu_model_path);
#else
int tflite_create_interpreter_from_file (tflite_interpreter_t *p, const char *model_path);
#endif
int tflite_create_interpreter_ex_from_file (tflite_interpreter_t *p, const char *model_path, tflite_createopt_t *opt);



#ifdef __cplusplus
}
#endif

#endif /* _UTIL_TFLITE_H_ */

