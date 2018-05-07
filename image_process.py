# -*- coding: utf-8 -*-
import numpy as np,sys
import os
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd
from general_functions import saveFile
from general_functions import loadFile
from utils import *
# other imports
np.random.seed(1)
# color type: 1 - grey, 3 - rgb
color_type_global = 3
# color_type = 1 - gray
# color_type = 3 - RGB
img_rows=224
img_cols=224
df1 = pd.read_csv('train_driver_imgs_list.csv') # exported from local database
df2 = pd.read_csv('val_driver_imgs_list.csv') # exported from local database
train_imgs_list= list(df1['img'])  #list(map(lambda x: x[:-4].strip(), df1['img']) )  # get train img list
val_imgs_list= list(df2['img'] )#list(map(lambda x: x[:-4].strip(), df2['img'])) #


train_folder = "dataset/kaggle_train_clean"
val_folder = "dataset/kaggle_valid_clean"
test_folder = "dataset/test"

vgg_mean = np.array([103.939, 116.799, 123.68], dtype=np.float32).reshape((1,1,3))

def vgg_preprocess(x):
    x= x - vgg_mean
    return x[:, ::-1] # reverse axis rgb->bgr

def get_im_vgg(path, img_rows, img_cols, color_type=1):
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows))
    resized = resized.astype(np.float32, copy=False)
    resized=vgg_preprocess(resized)
    #print(type(resized),resized.shape)
    return resized

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

def get_driver_data():
    dr = dict()
    path = os.path.join('dataset', 'driver_imgs_list.csv')
    print('Read drivers data')
    f = open(path, 'r')
    line = f.readline()
    while (1):
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        dr[arr[2]] = arr[0]
    f.close()
    return dr

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
        os.mkdir('dataset/keras_train_batch/'+d)
        os.mkdir('dataset/keras_valid_batch/'+d)
        for fl in files:
            flbase = os.path.basename(fl)
            if flbase in train_imgs_list:
                img = get_im(fl, img_rows, img_cols, color_type)
                cv2.imwrite(os.path.join('dataset', 'keras_train_batch','c' + str(j),  flbase), img)
                #cv2.imwrite(os.path.join(train_folder , flbase), img)
                #train[j]=img
                x_train.append(img)
                y_train.append(j)
                #train_driver_id.append(driver_data[flbase])
            elif flbase in val_imgs_list:
                img = get_im(fl, img_rows, img_cols, color_type)
                cv2.imwrite(os.path.join('dataset', 'keras_valid_batch','c' + str(j),  flbase), img)

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

    return x_train, y_train #, driver_id #, unique_drivers

def load_test(img_rows, img_cols, color_type=1):
    print('Read test images')
    path = os.path.join('dataset', 'kaggle_test_clean', '*.jpg')
    files = glob.glob(path)
    x_test = []
    x_test_id = []
    total = 0
    thr = math.floor(len(files)/10)
    for fl in files:
        #print(fl,type(fl))
        #sys.exit()
        flbase = os.path.basename(fl)
            #img = get_im(fl, img_rows, img_cols, color_type)
            #cv2.imwrite(os.path.join(test_folder , flbase), img)
        img = cv2.imread(fl)
        x_test.append(img)
        x_test_id.append(flbase)
        total += 1
        if total % thr == 0:
            print('Read {} images from {}'.format(total, len(files)))
    #save_array("x_test.csv",np.array(x_test))
    #saveFile("x_test.csv",x_test)
    #save_array("x_test_id.dat",np.array(x_test_id))
    #save_array("x_test.dat",np.array(x_test))
    #return 1#x_test_id
    return x_test, x_test_id


#load_train(img_rows, img_cols, color_type=3)
#load_test(img_rows, img_cols, color_type=3)
#x_train=a[0]
#y_train=a[1]
#print("image processing finished and saved")
