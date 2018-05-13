from utils import *
from image_process import *
from models import *
import sys,time
from keras.models import model_from_json

with open('conf/conf.json') as f:
  config = json.load(f)
model_path   = config["model_path"]

batch_size=32
epochs=19
path="dataset/"

train_folder = "dataset/keras_train_batch"
valid_folder = "dataset/keras_valid_batch"
test_folder = "dataset/kaggle_test_clean"



train_batches = get_batches(train_folder, gen_t1,  batch_size=batch_size)
#train_batches = get_batches(train_folder, batch_size=batch_size)
val_batches = get_batches(valid_folder, batch_size=batch_size, shuffle=False)

(val_classes, trn_classes, val_labels, trn_labels, val_filenames, train_filenames, test_filenames) = get_classes(path)

#trn = get_data(path+'keras_train_batch')
#val = get_data(path+'keras_valid_batch')
t=time.time()
model= vgg_tuned()
model.fit_generator(train_batches,steps_per_epoch=len(train_batches ), epochs=epochs, validation_data=val_batches,
                 validation_steps=len( val_batches),verbose=1)
print('Training time: %s s' % ( time.time()  -t ))
(loss, accuracy)=model.evaluate_generator(val_batches,steps=len(val_batches ), verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

model_json = model.to_json()
with open(model_path +  "batch_model.json", "w") as json_file:
 json_file.write(model_json)

# save weights
model.save_weights(model_path + "batch_model.h5")
print("[STATUS] saved model and weights to disk..")


print("----------------------------------------------------------")
