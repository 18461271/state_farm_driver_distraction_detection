# 
[state farm distracted driver detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection)

This is my first kaggle project,  I really enjoyed it.

### Public score: [0.79740](https://github.com/18461271/state_farm_driver_distraction_detection/blob/master/state_farm.JPG). Top 30%


The general steps are as follows:

1. Data: Split the kaggle train dataset into two parts, 80% is training data and 20% is validation data, making sure each driver is either in the train folder or in the validation folder. (This was done by importing driver_imgs_list into local database and randomly select driver ID and then exported the two dataset. )


* ``` SELECT * FROM `driver_imgs_list` WHERE subject IN ('p066', 'p056', 'p050', 'p021', 'p016'); ```
* ```SELECT * FROM `driver_imgs_list` WHERE NOT subject='p066' AND NOT subject='p056' AND NOT subject='p050' AND NOT subject='p021'AND NOT subject='p016'; ```

2. Images processing methods:  image_process.py



* (1)function: get_im() in image_process.py , it is used for resizing image to (224,224,3)
* (2)function: vgg_image() in image_process.pyvgg_process, it is used for vgg16 model, but it is very easy to go to overfitting,
* (3)gen_t1 in util.py,  image augmentation, this works like a charm.

3. Models:  models.py



* vgg_feature(): get the images features from vgg16 pretrained models.
* vgg_predict(p): load the images features and final tuning the model.

* test_model_2 : very basic cnn structure, has loss 3.23, accuracy 37%
* test_model_3(): adding more layers, activation function = 'relu', has loss  1.52, accuracy 56.5% ,
                                      activation function ='elu',   has loss 2.27, accuracy 46.711%  
* vgg_tuned(): vgg16 pretrained model without top layers and adding customized Dense layers. This model is mainly used for training.
* vgg_sgd(): the structure is the same as the previous one, only optimizer method differs, it takes more time, and the result is worse.


4. Model training: keras_batch_train.py
* using vgg16 pre-trained weights to train method_(2):vgg_processed data, save weights and continue training method_(1) processed data, save weights and train method(1)(3) processed data and then method(2)(3) processed data, the model is trained back and forth, the accuracy of validation data can achieve up to 100%, it is definitely overfitted.
* batchsize: 32, in general, the bigger the batchsize, the better accuracy and less loss, but it also requires more ram. My personal magic number is 2048 .
* epochs: 5, 10, 30. My model got converged after 10 epochs.

5. Test: keras_batch_test.py

 * test images are  processed by method(1) yields kaggle score: 0.80121, yields kaggle score 4.3 by using method(3).

Looking further: more advanced techniques like hand picked features ( head and radio area) are interesting, KNN is also worth trying, python package [annoy](https://github.com/spotify/annoy) is a good choice, it's fast and accurate, it is also implemented by Spotify. Howerver, my laptop is running out of memmory during saving features.

My device information is as follows:
 * Intel Core i7-77000HQ CPU@ 2.8GHz, 16GB RAM
 * NVIDIA GeForce GTX 1060 GDDR5@ 6,0GB (192-bit)


Winning Methods
1st by [jacobkie](https://www.kaggle.com/c/state-farm-distracted-driver-detection/discussion/22906#131467)
