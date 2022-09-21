import keras
from tensorflow.keras import layers, backend

class KernelModels:
    """ This class contains NN-based Keras models that modeling/simulating a series of computational kernels. """
    def __init__(self, model_name):
        self.model_name = model_name

    def get_model(self):
        """ Return corresponding model function object """
        func = getattr(self, self.model_name)
        assert callable(func), \
            f" Kernel model name: {self.model_name} not found. "
        return func

    @staticmethod
    def minimum_2d(in_shape, out_shape):
        """ This function return s minimum kernel model for latency testing purpose only. """
        inputs = keras.Input(shape=in_shape+(1,))
        x = layers.Conv2D(filters=1, kernel_size=1, padding='same', use_bias=False)(inputs)
        outputs = x
        return keras.Model(inputs, outputs)
        

    @staticmethod
    def sobel_2d(in_shape, out_shape):
        """ This function returns a NN-based Sobel model that simulates Sobel edge detection behavior. """
        encoded_dim = 16
        init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=2022)
        inputs = keras.Input(shape=in_shape+(1,))
        x = layers.Conv2D(filters=2, kernel_size=3, padding='same', activation='relu', use_bias=False, kernel_initializer=init)(inputs)
        x = layers.Conv2D(filters=encoded_dim, kernel_size=3, padding='same', activation='relu', use_bias=False, kernel_initializer=init)(x)
        x = layers.Conv2D(filters=1, kernel_size=3, padding='same', activation='relu', use_bias=False, kernel_initializer=init)(x)
        outputs = x
        return keras.Model(inputs, outputs)
    
    @staticmethod
    def mean_2d(in_shape, out_shape):
        """ This functions returns a NN-based mean filter model. """
        inputs = keras.Input(shape=in_shape+(1,))
        x = layers.Conv2D(filters=1, kernel_size=(3,3), padding='same', use_bias=False)(inputs)
        outputs = x
        return keras.Model(inputs, outputs)

    @staticmethod
    def laplacian_2d(in_shape, out_shape):
        """ This function returns a NN-based Laplacian filter model."""
        encoded_dim = 16
        inputs = keras.Input(shape=in_shape+(1,))
        x = layers.Conv2D(filters=2, kernel_size=5, padding='same', activation='relu')(inputs)
        x = layers.Conv2D(filters=encoded_dim, kernel_size=5, padding='same', activation='relu')(x)
        x = layers.Conv2D(filters=1, kernel_size=5, padding='same', activation='relu')(x)
        outputs = x
        return keras.Model(inputs, outputs)

    @staticmethod
    def histogram256(in_shape, out_shape):
        """ This function returns a NN-based hist256 model. """
        inputs = keras.Input(shape=in_shape+(1,))
#        x = layers.Conv1D(filters=4, kernel_size=1, activation='relu')(x)
#        x = layers.Conv1D(filters=4, kernel_size=1, activation='relu')(x)
        x = layers.Conv1D(filters=4, kernel_size=1, activation='relu')(inputs)
        x = layers.Flatten()(x)
        x = layers.Dense(256*4)(x)
        x = layers.Dense(256*4)(x)
#        x = layers.Dense(512*4)(x)
#        x = layers.Lambda(lambda x: backend.sum(x, axis=1))(x)


        outputs = x
        return keras.Model(inputs, outputs)
