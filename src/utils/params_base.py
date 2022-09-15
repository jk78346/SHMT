import os
import keras
from .utils import get_gittop

class TrainParamsBase():
    """ All default parameter setting. """
    
    def __init__(self, 
                 model_name: str,
                 size = 2048,
                 in_shape = (2048, 2048),
                 out_shape = (2048, 2048),
                 batch_size = 4,
                 epochs = 1000,
                 num_train = 400,
                 num_representative = 2,
                 optimizer = keras.optimizers.Adam(learning_rate=0.01),
                 loss = keras.losses.MeanSquaredError(),
                 metrics = [keras.metrics.RootMeanSquaredError()],
                 max_queue_size = 16,
                 use_multiprocessing = True,
                 validation_steps = 1,
                 workers = 2,
                 verbose = 1,
                 min_delta = 0,
                 patience = 4,
                 mode = 'auto'
                 ):
        # general model params
        self.model_name         = model_name
        self.size               = size
        self.in_shape           = in_shape
        self.out_shape          = out_shape
        self.batch_size         = batch_size
        self.epochs             = epochs
        self.num_train          = num_train
        self.num_representative = num_representative
        assert self.num_train >= self.batch_size, \
                f" num_train: {self.num_train} is smaller than batch_size: {self.batch_size}"

        # model.compile params
        self.optimizer = optimizer
        self.loss      = loss #keras.losses.sparse_categorical_crossentropy 
        self.metrics   = metrics # ['accuracy']

        # model.fit params
        self.max_queue_size      = max_queue_size
        self.use_multiprocessing = use_multiprocessing
        self.validation_steps    = validation_steps
        self.workers             = workers
        self.verbose             = verbose

        # callbacks - early stopping params
        self.min_delta = min_delta
        self.patience  = patience
        self.mode      = mode
    
class TrainParams(TrainParamsBase):
    """ training parameters setup based on given model name. """
    def __init__(self, model_name):
        if model_name == 'sobel_2d':
            TrainParamsBase.__init__(self, 
                                     model_name, 
                                     optimizer=keras.optimizers.Adam(learning_rate=0.02), 
                                     patience=5)
        elif model_name == 'mean_2d':
            TrainParamsBase.__init__(self, 
                                     model_name, 
                                     optimizer=keras.optimizers.Adam(learning_rate=0.02), 
                                     #loss=keras.losses.BinaryCrossentropy(from_logits=True),
                                     patience=5)
        elif model_name == 'laplacian_2d':
            TrainParamsBase.__init__(self, 
                                     model_name, 
                                     optimizer=keras.optimizers.Adam(learning_rate=0.02), 
                                     patience=5)
        elif model_name == 'histogram256':
            TrainParamsBase.__init__(self, 
                                     model_name, 
                                     in_shape=(2048,), 
                                     out_shape=(256*4,), 
                                     num_train=10000, 
                                     min_delta=0.00001)
        else: # use default params
            TrainParamsBase.__init__(self, model_name)

        # callbacks - checkpoint params
        model_path_base = get_gittop() + "/models/" + model_name + "_" + 'x'.join([str(i) for i in self.in_shape])
        os.system("mkdir -p " + model_path_base)
        self.checkpoint_path = model_path_base + "/" + model_name + "_checkpoint/" + model_name + ".ckpt"
        self.save_weights_only = False
        self.save_best_only = True
 
        # saved tf model dir
        self.saved_model_dir = model_path_base
        self.saved_model_path = self.saved_model_dir
 
        # expected saved tflite model path
        self.tflite_model_path = model_path_base + "/" + model_name + ".tflite"

        # single test Lenna image path
        self.lenna_path = get_gittop() + "/data/lena_gray_2Kx2K.bmp"


