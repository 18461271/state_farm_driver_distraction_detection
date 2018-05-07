from keras.models import model_from_json
from image_process import *
from utils import *
from models import *
import sys,time
import os,json
import numpy as np


img_rows=224
img_cols=224
batch_size=32
print("[INFO] Loaded models ")
#path = os.path.join('dataset', 'output',  '*.json')
with open('conf/conf.json') as f:
  config = json.load(f)
output_path =  config["output_path"]

#model_file = glob.glob(path)
json_file = open('dataset/output/model/batch_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("dataset/output/model/batch_model.h5")

#(val_classes, trn_classes, val_labels, trn_labels, val_filenames, train_filenames, test_filenames) = get_classes(path)

print("[INFO] loading test data ...")

test=load_array('x_test.dat')
test_ID = list(load_array('x_test_id.dat'))
#print(test_ID[:10])
print("[INFO] prediction started...")


test_preds = model.predict(test,verbose=1)
labels=['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
submission = pd.DataFrame(test_preds, columns=labels)
submission.insert(0, 'image', test_ID)
submission.head()

file_name = 'summit.csv'
submit_file = os.path.join(output_path, file_name)

submission.to_csv(submit_file, index=False)
print("[INFO] Saved submission csv file.")
