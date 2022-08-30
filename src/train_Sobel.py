import keras,os
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import tflearn
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import configparser
import subprocess

physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
assert tf.test.is_built_with_cuda()

size = 2048
img_size=(size, size)
num_classes=1
batch_size = 16
epochs=100

config = configparser.ConfigParser()

def get_gittop():
    """"""
    return subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], \
                            stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')

conf_file = get_gittop()+"/configure.cfg"
config.readfp(open(conf_file))

path_base = config.get('global_path', 'DATASET_MOUNT')+"/Data/Sobel_"+str(size)+"/"
train_input_img_paths  = path_base + "in_npy/train/ILSVRC2014_train_0000/"
train_target_img_paths = path_base + "out_npy/train/ILSVRC2014_train_0000/"
val_input_img_paths  = path_base + "in_npy/val/"
val_target_img_paths = path_base + "out_npy/val/"
test_input_img_paths  = path_base + "in_npy/test/"
test_target_img_paths = path_base + "out_npy/test/"

sobel_model_base = get_gittop()+"/models/Sobel/"
os.system("mkdir -p "+sobel_model_base)

checkpoint_path  = sobel_model_base+"/Sobel_checkpoint/cp.ckpt"
saved_model_path = sobel_model_base+"/Sobel_model"

def get_imgs_num(path):
    """ This function returns number of images under this path """
    return len([
        os.path.join(path, fname)
        for fname in os.listdir(path)
        if fname.endswith(".npy")
    ])

def get_imgs(path):
    """ This function returns list of img paths """
    return sorted(
            [
                os.path.join(path, fname)
                for fname in os.listdir(path)
                if fname.endswith(".npy")
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
        """ This function returns tuple (input, target) correspond to btach #idx. """
        i = idx * self.batch_size;
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_input_img_paths):
            img = np.load(path)
            img = np.expand_dims(img, axis=2) 
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = np.load(path)
            img = np.expand_dims(img, axis=2) 
            y[j] = img
        return x, y

def get_model(img_size, num_classes):
    """ This function returns a simple design of Sobel model """
    encoded_dim = 4
    inputs = keras.Input(shape=img_size+(1,))
    x = layers.Resizing(128, 128)(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(encoded_dim * encoded_dim, activation="sigmoid")(x)
    x = layers.Reshape((encoded_dim, encoded_dim, 1))(x)
    outputs = layers.UpSampling2D((img_size[i]/encoded_dim for i in range(len(img_size))), interpolation='bilinear')(x)
    model = keras.Model(inputs, outputs)
    return model

model = get_model(img_size, num_classes)
model.summary()

early = EarlyStopping(min_delta=0, patience=10, verbose=1, mode='auto')

checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
                                              save_weights_only=False, 
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
                 max_queue_size=8,
                 use_multiprocessing=True,
                 workers=4,
                 callbacks=[early, cp_callback])

tf.saved_model.save(model, saved_model_path)
print("Sobel model saved.")


