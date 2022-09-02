import keras
from tensorflow.keras import layers

class KernelModels:
    """ This class contains NN-based Keras models that modeling/simulating a series of computation kernels. """
    def __init__(self, model_name):
        self.model_name = model_name

    def get_model(self):
        """ Return corresponding model function object """
        func = getattr(self, self.model_name)
        assert callable(func), \
            f" Kernel model name: {model_name} not found. "
        return func

    @staticmethod
    def sobel_2d(img_size):
        """ This function returns a NN-based Sobel model that simulates Sobel edge detection behavior. """
        encoded_dim = 16
        inputs = keras.Input(shape=img_size+(1,))
        x = layers.Conv2D(filters=encoded_dim, kernel_size=3, padding='same', activation='relu')(inputs)
        outputs = layers.Conv2D(filters=1, kernel_size=3, padding='same', activation='relu')(x)
        return keras.Model(inputs, outputs)

