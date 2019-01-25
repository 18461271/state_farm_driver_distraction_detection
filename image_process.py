# -*- coding: utf-8 -*-
import numpy as np,sys
import os
import glob
import cv2
import math
import pickle
import datetime, random
import pandas as pd
from utils import *

np.random.seed(1)
img_rows=224
img_cols=224

df = pd.read_csv('driver_imgs_list.csv') # supplied by  kaggle
by_drivers = df.groupby('subject')
unique_drivers = list(by_drivers.groups.keys())

# Set validation set percentage with regards to training set
val_pct = 0.2
random.shuffle(unique_drivers)

# These are the drivers we will be entirely moving to the validation set
to_val_drivers = unique_drivers[:int(len(unique_drivers) * val_pct)]

#split the kaggle train images according to driver so that each driver is never repeated in the train and valid dataset.
df1 = pd.read_csv('train_driver_imgs_list.csv') # exported from local database
df2 = pd.read_csv('val_driver_imgs_list.csv') # exported from local database
train_imgs_list= list(df1['img'])  #list(map(lambda x: x[:-4].strip(), df1['img']) )  # get train img list
val_imgs_list= list(df2['img'] )#list(map(lambda x: x[:-4].strip(), df2['img'])) #

test_folder = "dataset/test"

target_train_folder = "dataset/keras_train_batch2/"
target_valid_folder ="dataset/keras_valid_batch2/"

def vgg_image(x):
	image = load_img(x, target_size=(224, 224))
	image = img_to_array(image)
	image = image.reshape(( image.shape[0], image.shape[1], image.shape[2]))
	image = preprocess_input(image)
	#print(image.shape)
	#print(type(image))
	return image

def get_im(path, img_rows, img_cols, color_type=1):
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows))
                                  #resized = resized.astype(np.float32, copy=False)
    return resized
    #return img


def load_train(img_rows, img_cols, color_type=1):
    x_train = []
    x_val = []
    train={}
    val={}
    y_train=[]
    y_val=[]
    val_driver_id = []
    train_driver_id = []
    driver_data = get_driver_data()
    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('dataset', 'train',
                            'c' + str(j), '*.jpg')
        files = glob.glob(path)
        d='c'+str(j)
        os.mkdir(target_train_folder +d)
        os.mkdir(target_valid_folder +d)
        for fl in files:
            flbase = os.path.basename(fl)
            if flbase in train_imgs_list:
                img = get_im(fl, img_rows, img_cols, color_type)
                #img = vgg_image(fl)
                #sys.exit()
                cv2.imwrite(os.path.join(target_train_folder + 'c' + str(j),  flbase), img)
                #cv2.imwrite(os.path.join(train_folder , flbase), img)
                #train[j]=img
                #x_train.append(img)
                #y_train.append(j)
                #train_driver_id.append(driver_data[flbase])
            elif flbase in val_imgs_list:
                img = get_im(fl, img_rows, img_cols, color_type)
                #img = vgg_image(fl)
                cv2.imwrite(os.path.join(target_valid_folder + 'c' + str(j),  flbase), img)
                #cv2.imwrite(os.path.join(val_folder , flbase), img)
                #val[j]=img
                #x_val.append(img)
                #y_val.append(j)
                #val_driver_id.append(driver_data[flbase])
            else:
                print("image name is not in none of the following list: train and the val")
                sys.exit()
    #saveFile("vgg_y_val.csv",y_val)
    #saveFile("vgg_y_train.csv",y_train)
    #saveFile( "vgg_x_val.csv",x_val)
    #saveFile( "vgg_x_train.csv",x_train)
    #saveFile( "vgg_train_driver_id.csv",train_driver_id)
    return 1
    #return x_train, y_train #, driver_id #, unique_drivers

def load_test(img_rows, img_cols, color_type=1):
    print('Read test images')
    path = os.path.join('dataset', 'test', '*.jpg')
    files = glob.glob(path)
    x_test_normalized = []
    x_test_id_normalized = []
    total = 0
    thr = math.floor(len(files)/10)
    for fl in files:
        #print(fl,type(fl))
        #sys.exit()
        flbase = os.path.basename(fl)
        #img = vgg_image(fl, img_rows, img_cols, color_type)
        img = vgg_image(fl)
            #cv2.imwrite(os.path.join(test_folder , flbase), img)
        #img = cv2.imread(fl)
        x_test_normalized.append(img)
        x_test_id_normalized.append(flbase)
        total += 1
        if total % thr == 0:
            print('Read {} images from {}'.format(total, len(files)))
    #save_array("x_test.csv",np.array(x_test))
    #saveFile("x_test.csv",x_test)
    save_array("x_test_id_vgg6.dat",np.array(x_test_id_normalized))
    save_array("x_test_vgg6.dat",np.array(x_test_normalized))
    return 1#x_test_id



#load_train(img_rows, img_cols, color_type=3)
#load_test(img_rows, img_cols, color_type=3)
#print("image processing finished and saved")
