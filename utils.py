from keras.preprocessing import image
import numpy as np
import bcolz
import itertools
from keras.utils.np_utils import to_categorical

gen_t1 = image.ImageDataGenerator(rotation_range=15, height_shift_range=0.05,
                shear_range=0.1, channel_shift_range=20, width_shift_range=0.1)
#gen_t1 loss=0.6993, accuracy: 79.3299%, 5 epochs,
#gen_t1 loss=0.7558, accuracy: 80.3646%, 20 epochs,
#gen_t1 loss=0.8266, accuracy: 79.7241%, 20 epochs,

gen_t2 = image.ImageDataGenerator(rescale=1./255,featurewise_center=True,rotation_range=15,featurewise_std_normalization=True,
                height_shift_range=0.05,width_shift_range=.1,
                shear_range=0.1, channel_shift_range=20)

#gen_t2 results in very bad accuracy, and it needs more than 20 epochs to achieve 50%

def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True, batch_size=4, class_mode='categorical',
                target_size=(224,224)):
    return gen.flow_from_directory(dirname, target_size=target_size,
            class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)


def get_classes(path):
    train_batches = get_batches(path+'keras_train_batch', shuffle=False, batch_size=1)
    val_batches = get_batches(path+'keras_valid_batch', shuffle=False, batch_size=1)
    test_batches = get_batches(path+'test', shuffle=False, batch_size=1)
    return (val_batches.classes, train_batches.classes, onehot(val_batches.classes), onehot(train_batches.classes),
        val_batches.filenames, train_batches.filenames, test_batches.filenames)

def get_data(path, target_size=(224,224)):
    batches = get_batches(path, shuffle=False, batch_size=1, class_mode=None, target_size=target_size)
    steps_per_epoch=len(train_batches )
    return np.concatenate([batches.next() for i in range(steps_per_epoch)])

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]

def onehot(x):
    return to_categorical(x)
