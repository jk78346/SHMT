import os
assert 'IS_GPGTPU_CONTAINER' in os.environ, \
           f" Kernel model training script is not running within GPGTPU container. "
import keras
import tflearn
import argparse
import cv2 as cv
import subprocess
import numpy as np
import tensorflow as tf
from utils.utils import (
        get_gittop, 
        get_imgs_count, 
        get_img_paths_list
)
from keras.callbacks import (
        ModelCheckpoint, 
        EarlyStopping
)
from keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from utils.params_base import TrainParamsBase
from kernels.kernel_models import KernelModels
from kernels.ground_truth_functions import Applications
from tensorflow.keras.preprocessing.image import load_img

class MyDataGen(keras.utils.Sequence):
    """ A customized data generator """
    def __init__(self, batch_size, shape, num_samples, func):
        self.batch_size  = batch_size
        self.shape       = shape
        self.num_samples = num_samples
        # The ground truth function callable
        self.func        = func

    def __len__(self):
        """ Return number of batchs of this dataset. """
        return int(np.floor(self.num_samples) / self.batch_size)

    def __getitem__(self, idx):
        """ This function returns tuple (input, target) correspond to batch #idx. """
        x = np.zeros((self.batch_size,) + self.shape + (1,), dtype="uint8")
        y = np.zeros((self.batch_size,) + self.shape + (1,), dtype="uint8")
        for j in range(self.batch_size):
            x_slice = np.random.randint(255, size=self.shape, dtype="uint8")
            y_slice = self.func(x_slice)
            x_slice = np.expand_dims(x_slice, axis=-1)
            y_slice = np.expand_dims(y_slice, axis=-1)
            x[j] = x_slice
            y[j] = y_slice
        return x, y

class TrainParams(TrainParamsBase):
    """ training parameters setup. Specify any parameter other than default here. """
    def __init__(self, model_name):
        """ Give default paths for output artifacts. """
        TrainParamsBase.__init__(self, model_name)
        assert (model_name in dir(KernelModels) and model_name in dir(Applications)), \
            f" Given model name: {model_name} is not supported. Check for available kernel and application implementations. "

        # callbacks - checkpoint params
        model_path_base = get_gittop() + "/models/" + model_name + "/"
        os.system("mkdir -p " + model_path_base)
        self.checkpoint_path = model_path_base + "/" + model_name + "_checkpoint/" + model_name + ".ckpt"
        self.save_weights_only = False
        self.save_best_only = True

        # output path params
        self.saved_model_path = model_path_base + "/" + model_name + "_model"

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

def main(args):
    """ The main training script """
    gpu_setup()
    params = TrainParams(args.model)
    target_func, kernel_model = get_funcs(args.model)
    model = kernel_model(params.shape)
    model.summary()

    params.print_num_samples()
    
    train_gen = MyDataGen(params.batch_size, params.shape, params.num_train, target_func)
    val_gen   = MyDataGen(params.batch_size, params.shape, params.num_val, target_func)
    test_gen  = MyDataGen(params.batch_size, params.shape, params.num_test, target_func)

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

    print("model.fit starting...")
    hist = model.fit(train_gen, 
                 epochs=params.epochs, 
                 validation_data=val_gen, 
                 validation_steps=params.validation_steps,
                 max_queue_size=params.max_queue_size,
                 use_multiprocessing=params.use_multiprocessing,
                 workers=params.workers,
                 verbose=params.verbose,
                 callbacks=[early, cp_callback])

    tf.saved_model.save(model, params.saved_model_path)
    print(f"model {args.model} saved at {params.saved_model_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This Python script trains NN-based TF model that simulates target function kernel.')
    parser.add_argument('--model', action='store', type=str, help='name of the kernel model for training')
    args = parser.parse_args()
    main(args)
