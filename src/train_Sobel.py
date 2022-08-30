import keras,os
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import tflearn
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
assert tf.test.is_built_with_cuda()

size = 2048
img_size=(size, size)
num_classes=1
batch_size = 4
epochs=1

path_base = "/mnt/Data/Sobel_"+str(size)+"/"
train_input_img_paths  = path_base + "in/train/ILSVRC2014_train_0000/"
train_target_img_paths = path_base + "out/train/ILSVRC2014_train_0000/"
val_input_img_paths  = path_base + "in/val/"
val_target_img_paths = path_base + "out/val/"
test_input_img_paths  = path_base + "in/test/"
test_target_img_paths = path_base + "out/test/"

def get_imgs_num(path):
    """ This function returns number of images under this path """
    return len([
        os.path.join(path, fname)
        for fname in os.listdir(path)
        if fname.endswith(".JPEG")
    ])

def get_imgs(path):
    """ This function returns list of img paths """
    return sorted(
            [
                os.path.join(path, fname)
                for fname in os.listdir(path)
                if fname.endswith(".JPEG")
            ]
            )

class MyDataGen(keras.utils.Sequence):
    """ A customized data generator """
    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size       = batch_size
        self.img_size         = img_size
        self.input_img_paths  = get_imgs(input_img_paths)
        self.target_img_paths = get_imgs(target_img_paths)

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """ This function returns tuple (input, target) correspond to btach #idx.  """
        i = idx * self.batch_size;
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            img = np.expand_dims(img, axis=2) 
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            img = np.expand_dims(img, axis=2) 
            y[j] = img
        return x, y

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size+(1,))
    x = layers.Conv2D(16, 1, padding="same")(inputs)
    outputs = layers.Conv2D(num_classes, 1, activation="softmax", padding="same")(x)

    model = keras.Model(inputs, outputs)
    return model

model = get_model(img_size, num_classes)
model.summary()

early = EarlyStopping(min_delta=0, patience=20, verbose=1, mode='auto')

checkpoint_path = "train/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
                                              save_weights_only=True, 
                                              verbose=1)

train_gen = MyDataGen(batch_size, img_size, train_input_img_paths, train_target_img_paths)
val_gen   = MyDataGen(batch_size, img_size, val_input_img_paths,   val_target_img_paths)
test_gen  = MyDataGen(batch_size, img_size, test_input_img_paths,  test_target_img_paths)

print("number of train samples: ", get_imgs_num(train_input_img_paths))
print("number of val   samples: ", get_imgs_num(val_input_img_paths))
print("number of test  samples: ", get_imgs_num(test_input_img_paths))

model.compile(optimizer="rmsprop", 
              loss=keras.losses.sparse_categorical_crossentropy, 
              metrics=['accuracy'])

print("model.fit starting...")
hist = model.fit(train_gen, 
                 epochs=epochs, 
                 validation_data=val_gen, 
                 callbacks=[early, cp_callback])

tf.saved_model.save(model, './Sobel_model')

model.load_weights(checkpoint_path)

layer_array = ["conv2d", "conv2d_1", "dense", "dense_1", "dense_2"]

def get_tf_weights(model, layer_array):
    data_file = [] # contains weights and bias
    
    for layer in layer_array:
        weights = model.get_layer(layer).get_weights()[0]
        biases  = model.get_layer(layer).get_weights()[1]
        layer_pair = [weights, biases]
        data_file.append(layer_pair)

    np.save('vgg2_weight.npy', data_file)

get_tf_weights(model, layer_array)


