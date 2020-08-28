# Colormask-detection
1.Detection using coloured mask dataset.

Libraries to install - cv2, os, numpy ,np_utils from keras.utils , load_model from keras.model.

Dataset - used 1995 with_mask images and 1918 without_mask images.

Steps for implementation:

Step 1) Importing the required libraries.

Step 2) Adding datasets to the data path . Datasets contains images of with mask and without mask. Creating empty list of data and target, after looping through the categories, images are stored in data and images of with and without mask are appended to target.

Step 3) Converting the images into grayscale and resizing them.

Step 4) Sometimes images may not be available and face some error then it is handled by exception. Normalize the image by dividing it by 255 which converts pixel range into 0 and 1. Reshape the images into 4D because neural network takes 4D input.

Step 5) Building CNN model Two Conv2D layers, two max pooling layer, one flatten layer and one dense layer(categorical)).

Step 6) Spliting the dataset into train and test set. Have trained the model for 20 epochs with 0.1 test_size and 0.2 validation_split and got an trained accuaracy of 99.5%, validation accuracy of 93.15%.

Step 7) We have made use of ModelCheckpoint to save the best epoch.

Step 8) For face detection we have used haarcascade_frontalface_alt.xml file.

step 9) Using VideoCapture(0) for webcam. Detecting the face. Converting into grayscale. Normalize the image by dividing it by 255 which converts pixel range into 0 and 1. Reshape the images into 4D because neural network takes 4D input. Creating a bounding box around the face and detecting whether the person is wearing a mask or no mask.

step 10) In cv2. imshow we use colored frames captured from the webcam.

Detecting for colored mask using only white mask dataset.
When we have only white mask dataset, it will detect only when we wear white mask so if we want the model to detect even when we wear a colored mask the output from the webcam should be in grayscale.

Dataset - https://github.com/balajisrinivas/Face-Mask-Detection/tree/master/dataset
Dataset for grayscale - https://github.com/prajnasb/face_detector/tree/master/dataset

The libraries and the step to process is all same as done for colored mask.

© 2020 GitHub, Inc.
