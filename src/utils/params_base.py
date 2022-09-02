import os
import keras

class TrainParamsBase():
    """ All default parameter setting. """
    
    def __init__(self, model_name: str):
        # general model params
        self.size               = 2048
        self.shape              = (self.size, self.size)
        self.batch_size         = 4
        self.epochs             = 100
        self.num_train          = 400
        self.num_representative = 2
        assert self.num_train >= self.batch_size, \
                f" num_train: {self.num_train} is smaller than batch_size: {self.batch_size}"

        # model.compile params
        self.optimizer = "adam"
        self.loss      = "mse" #keras.losses.sparse_categorical_crossentropy 
        self.metrics   = [] # ['accuracy']

        # model.fit params
        self.max_queue_size      = 16
        self.use_multiprocessing = True
        self.validation_steps    = 1
        self.workers             = 2
        self.verbose             = 1

        # callbacks - early stopping params
        self.min_delta = 0
        self.patience  = 4
        self.mode      = 'auto'


