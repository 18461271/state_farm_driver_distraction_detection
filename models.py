import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.layers import Dense, Activation, Flatten,Dropout, GlobalAveragePooling2D,Convolution2D,MaxPooling2D
from keras.models import Model
from keras import optimizers
from keras.models import Sequential
from keras.layers import merge, Input, BatchNormalization
from keras.models import model_from_json
from keras.utils import np_utils
# other imports
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob,sys
import h5py
import os
import json
import datetime
import time

from keras.preprocessing.image import img_to_array
# other imports


with open('conf/conf.json') as f:
  config = json.load(f)
weights   = config["weights"]


def vgg_feature():    #2.07,, 49.81%
    #base_model = VGG16(weights=weights, include_top=False,input_shape=(224,224,3))
    base_model = VGG16(weights=weights)
    model = Model(input = base_model.input, output=base_model.get_layer('block5_conv3').output)
    #model.compile(optimizers.Adam(lr=1e-4),loss='categorical_crossentropy', metrics=['accuracy'])
    return model



def vgg_predict(p):
    base_model = VGG16(weights=weights)
    model=Sequential([
        MaxPooling2D(input_shape=base_model.get_layer('block5_conv3').output_shape[1:]),
        Flatten(),
        Dropout(p),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(10, activation='softmax')
        ])
    model.compile(optimizers.Adam(lr=1e-4),loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def test_model_2(): #3.23, 37%
    model=Sequential([
    BatchNormalization(axis=1, input_shape=(224,224,3)),
     Flatten(),
     Dense(100, activation='relu'),
     BatchNormalization(),
     Dense(10, activation='softmax')
     ])
    model.compile(optimizers.Adam(lr=1e-4),loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def test_model_3(): # relu  1.52, 56.5%   elu: 2.27  46.711
    model = Sequential([
        BatchNormalization(axis=1, input_shape=(224,224,3)),
        Convolution2D(32,(3,3), activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D(),
        Convolution2D(64,(3,3), activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D(),
        Convolution2D(128,(3,3), activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D(),
        Flatten(),
        Dense(200, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(200, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizers.Adam(lr=1e-4),loss='categorical_crossentropy', metrics=['accuracy'])
    return model

p=0.5
def vgg_tuned():#epoch=8, loss=0.766, acc=78.07
    base_model=VGG16(weights=weights, include_top=False,input_shape=(224,224,3))
    for layer in base_model.layers:
        layer.trainable = False
    y = base_model.output
    y = Flatten()(y)
    y = Dropout(p)(y)
    y = Dense(256, activation="relu")(y) #1024
    y = BatchNormalization()(y)
    y = Dropout(p)(y)
    y = Dense(256, activation="relu")(y) #1024
    y = BatchNormalization()(y)
    y = Dropout(p)(y)
    predictions = Dense(10, activation="softmax")(y)
    model = Model(input = base_model.input, output = predictions)
    model.compile(optimizers.Adam(lr=1e-4),loss='categorical_crossentropy', metrics=['accuracy'])
    return model



def vgg_sgd():
    base_model=VGG16(weights=weights, include_top=False,input_shape=(224,224,3))
    for layer in base_model.layers:
        layer.trainable = False
    y = base_model.output
    y = Flatten()(y)
    y = Dropout(p)(y)
    y = Dense(256, activation="relu")(y) #1024
    y = BatchNormalization()(y)
    y = Dropout(p)(y)
    y = Dense(256, activation="relu")(y) #1024
    y = BatchNormalization()(y)
    y = Dropout(p)(y)
    predictions = Dense(10, activation="softmax")(y)
    model = Model(input = base_model.input, output = predictions)
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
