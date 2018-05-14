# 
[state farm distracted driver detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection)

This is my first kaggle project,  I really enjoyed it.



### Public score: [0.80121](https://github.com/18461271/state_farm_driver_distraction_detection/blob/master/state_farm.JPG)



The general steps are as follows:

1. Data: Split the kaggle train dataset into two parts, 80% is training data and 20% is validation data, making sure each driver is either in the train folder or in the validation folder. (This is done by import driver_imgs_list into local database and randomly select driver ID and then export the two dataset. )


*``` SELECT * FROM `driver_imgs_list` WHERE subject IN ('p066', 'p056', 'p050', 'p021', 'p016')```
* ```SELECT * FROM `driver_imgs_list` WHERE NOT subject='p066' AND NOT subject='p056' AND NOT subject='p050' AND NOT subject='p021'AND NOT subject='p016'; ```

2. Images processing: 
* (1)resize to (224,224,3)
* (2) vgg_process 
* (3)image augmentation.

3. Models: vgg16 pretrained models without using the full connnected layers and using customized layers.

4. Train: using vgg16 pre-trained weights to train method_(2):vgg_processed data, save weights and continue training method_(1) processed data, save weights and train method(1)(3) processed data and then method(2)(3) data.

5. Test: test images are  processed by method(1).


Looking further: more advanced techniques like hand picked features ( head and radio area) are interesting, KNN is also worth trying, but my laptop is running out of memmory during saving features.

My device information:
Intel Core i7-77000HQ CPU@2.8GHz, 16GB RAM
NVIDIA GeForce GTX 1060 GDDR5@6,0GB (192-bit)


Winning Methods
1st by [jacobkie](https://www.kaggle.com/c/state-farm-distracted-driver-detection/discussion/22906#131467)
