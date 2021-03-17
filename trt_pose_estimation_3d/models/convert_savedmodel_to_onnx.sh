#/bin/sh
set -e
set -x

python3 -m tf2onnx.convert \
    --saved-model saved_model \
#    --saved-model ~/PINTO_model_zoo/029_human-pose-estimation-3d-0001/10_tftrt/tensorrt_saved_model_float32/ \
    --output human-pose-estimation-3d.onnx\
    --inputs data:0[1,256,448,3]


    
