import os
assert 'IS_GPGTPU_CONTAINER' in os.environ, \
           f" Kernel model generating script is not running within GPGTPU container. "
import time
import keras
import random
import argparse
import cv2 as cv
import subprocess
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.stats import binom
from utils.params_base import TrainParams
from utils.utils import (
        get_imgs_count, 
        get_img_paths_list,
        Quality,    
)
from keras.callbacks import (
        ModelCheckpoint, 
        EarlyStopping
)
from keras.models import Sequential
from tensorflow.keras import (
        layers, 
        callbacks
)
from tensorflow.keras.optimizers import Adam
from utils.params_base import TrainParamsBase
from kernels.kernel_models import KernelModels
from kernels.ground_truth_functions import Applications

class MyDataGen():
    """ A customized data generator """
    def __init__(self, params, target_func):
        self.batch_size         = params.batch_size
        self.in_shape           = params.in_shape
        self.out_shape          = params.out_shape
        self.num_samples        = params.num_train
        self.num_representative = params.num_representative
        self.model_name         = params.model_name
        # The ground truth function callable
        self.func               = target_func
        self.input_img_paths    = get_img_paths_list("/mnt/Data/Sobel_2048/in_npy/train/ILSVRC2014_train_0000/", '.npy')[:self.num_samples] 
        self.num_imgs           = len(self.input_img_paths)

    def random_input_gen(self):
        """ This function generates random samples for training input. """
        x = np.zeros((self.num_samples,) + self.in_shape + (1,) , dtype="float32")
        y = np.zeros((self.num_samples,) + self.out_shape + (1,), dtype="float32")
        
        for j in range(self.num_samples):
            if self.model_name == 'histogram256':
                x_slice = np.full(self.in_shape[0], binom.pmf(list(range(self.in_shape[0])), 255, random.random()))    
                x_slice = (x_slice / max(x_slice)) * 255
                x_slice = x_slice.astype("uint8")
            else:
                np.random.seed()
                x_slice = np.random.randint(255, size=self.in_shape, dtype="uint8")
            
            y_slice = self.func(x_slice)
            x_slice = np.expand_dims(x_slice, axis=-1)
            y_slice = np.expand_dims(y_slice, axis=-1)
            x[j] = (x_slice.astype('float32') / 255.) 
            y[j] = (y_slice.astype('float32') / 255.)
        return x, y
    
    def representative_gen(self):
        """ representative dataset generator """
        for j in range(self.num_representative):
#            x_slice = np.load(self.input_img_paths[j])
#            x_slice = np.expand_dims(x_slice, axis=0)
            random.seed(time.time())
            x_slice = np.random.randint(255, size=(1,) + self.in_shape, dtype="uint8")
            x_slice = np.expand_dims(x_slice, axis=-1)
            x = x_slice.astype('float32')
            yield [x]

def gpu_setup():
    """ GPU setup """
    physical_devices = tf.config.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)
    assert tf.test.is_built_with_cuda()

def get_funcs(model_name):
    """ Get target function as ground truth generator and kernel model in Keras for training."""
    my_kernel_model = KernelModels(model_name)
    my_application  = Applications(model_name)
    model = my_kernel_model.get_model()
    app = my_application.get_func()
    return app, model

def model_lr(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.995 

def train(params, kernel_model, random_input_gen):
    """ The main training script """
    gpu_setup()
    model = kernel_model(params.in_shape, params.out_shape)
    model.summary()

    X_train, Y_train = random_input_gen()

    model.compile(optimizer=params.optimizer, 
              loss=params.loss, 
              metrics=params.metrics)
    
    early = EarlyStopping(min_delta=params.min_delta, 
                          patience=params.patience, 
                          verbose=params.verbose, 
                          mode=params.mode)

    cp_callback = keras.callbacks.ModelCheckpoint(filepath=params.checkpoint_path, 
                                              save_weights_only=params.save_weights_only, 
                                              save_best_only=params.save_best_only,
                                              verbose=params.verbose)

    lr_scheduler = keras.callbacks.LearningRateScheduler(model_lr)

    print("model.fit starting...")
    hist = model.fit(X_train,
                     Y_train,
                 epochs=params.epochs, 
                 batch_size=params.batch_size,
                 validation_split=0.1,
                 max_queue_size=params.max_queue_size,
                 use_multiprocessing=params.use_multiprocessing,
                 workers=params.workers,
                 verbose=params.verbose,
                 callbacks=[early, cp_callback, lr_scheduler])

    tf.keras.models.save_model(model, params.saved_model_dir)
    print(f"model \"{args.model}\" saved at {params.saved_model_dir}.")

def calc_metrics(Y_ground_truth, Y_predict):
    quality = Quality(Y_ground_truth, Y_predict)
    print("error_rate      : ", quality.error_rate(), " %")
    print("error_percentage: ", quality.error_percentage(), " %")
    print("RMSE_percentage : ", quality.rmse_percentage(), " %")
    print("SSIM            : ", quality.ssim())
    print("PNSR            : ", quality.pnsr(), " dB")

def pre_quantize_test(params, target_func):
# Now only image type of applications are supported now.
    image = Image.open(params.lenna_path)
    image = image.resize(params.in_shape)

    X_test = np.asarray(image).astype('uint8') 
    print("X.shape: ", X_test.shape, ", dtype: ", X_test.dtype, ", max: ", X_test.max(), ", min: ", X_test.min())
    print(X_test)
    # get ground truth
    Y_ground_truth = target_func(np.asarray(image).astype('uint8'))

    np.set_printoptions(edgeitems=5, precision=3, linewidth=120)

    model = tf.keras.models.load_model(params.saved_model_dir)

    print(model.get_weights())

    X_test = np.expand_dims(X_test, axis=(0, 3))

    print("start to evaluate...")
    # get model prediction
    X_test = (X_test.astype('float32') / 255.) 
    model.summary()
    print("X_test.shape: ", X_test.shape)
    Y_predict = model.predict(X_test, batch_size=1)
    Y_predict = np.asarray(Y_predict)
    Y_predict = (Y_predict)  * 255.
    Y_predict = np.squeeze(Y_predict, axis=0)
    Y_predict = np.squeeze(Y_predict, axis=-1)
    print("Y_predict.shape: ", Y_predict.shape, ", dtype: ", Y_predict.dtype, ", max: ", Y_predict.max(), ", min: ", Y_predict.min())
    print(Y_predict)

    print("Y_ground_truth.shape: ", Y_ground_truth.shape, ", dtype: ", Y_ground_truth.dtype, ", max: ", Y_ground_truth.max(), ", min: ", Y_ground_truth.min())
    print(Y_ground_truth)

    calc_metrics(Y_ground_truth, Y_predict) 

    cv.imwrite("./ground_truth.png", Y_ground_truth)
    cv.imwrite("./predict.png", Y_predict)

def convert_to_tflite(params, representative_gen):
    """ This function converts saved tf model to edgeTPU-compatible tflite model. """
    print("start converting to tflite setting...")
    converter = tf.lite.TFLiteConverter.from_saved_model(params.saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    print("converter starts converting...")
    tflite_model = converter.convert()
    with open(params.tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print("edgetpu_compiler compiling...")
    os.system("edgetpu_compiler -s -m 13 "+params.tflite_model_path+" -o "+params.saved_model_dir)

def main(args):
    """ The main script """
    params = TrainParams(args.model)
    assert (args.model in dir(KernelModels) and args.model in dir(Applications)), \
        f" Given model name \"{args.model}\" is not supported. Check for available kernel and application implementations."
    for k, v in vars(params).items():
        print(k, ": ", v)
    target_func, kernel_model = get_funcs(args.model)
    my_data_gen = MyDataGen(params, target_func)
    if args.skip_train == False:
        train(params, kernel_model, my_data_gen.random_input_gen)
    if args.skip_pre_test == False:
        pre_quantize_test(params, target_func)
    if args.skip_tflite == False:
        convert_to_tflite(params, my_data_gen.representative_gen)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This Python script generates NN-based tflite model that simulates target function kernel.')
    parser.add_argument('--model', action='store', type=str, help='name of the kernel model for training')
    parser.add_argument('--skip_train', dest='skip_train', action='store_true', help='To skip kernel model training if saved model already exists.')
    parser.add_argument('--skip_pre_test', dest='skip_pre_test', action='store_true', help='To skip testing on trained fp32 model (before quantization).')
    parser.add_argument('--skip_tflite', dest='skip_tflite', action='store_true', help='To skip tflite converting.')
    parser.set_defaults(skip_train=False)
    parser.set_defaults(skip_tflite=False)
    args = parser.parse_args()
    main(args)
