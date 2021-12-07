"""
@author: Alberto Pazzaia
Dataset: https://www.kaggle.com/wardaddy24/marble-surface-anomaly-detection-2 LAST CONSULTATION 08/10/2021
"""
#%% Libraries used
import cv2
import os # To select elements in folders

# Data processing and mathematics
import numpy as np
from sklearn.model_selection import train_test_split

#%% MARBLE DATASET
# Fix Random Seed for all the code (to reproduce exacly we should fix the environment seed https://stackoverflow.com/questions/50659482/why-cant-i-get-reproducible-results-in-keras-even-though-i-set-the-random-seeds)
# random seeds must be set before importing keras & tensorflow
my_seed = 42
np.random.seed(my_seed)
import random 
random.seed(my_seed)
import tensorflow as tf
tf.random.set_seed(my_seed)

# Path where the images are stored
img_path = '../Kaggle dataset/Marble dataset/train/'

# I save images, labels and names
images = []
label = []
images_names = []

for dir_name in os.listdir(img_path):
    for filename in os.listdir(os.path.join(img_path, dir_name)):
        img = cv2.imread(os.path.join(img_path, os.path.join(dir_name, filename)))
        # Resize images --> ResNet50 takes 224,224 and EfficientNetB2 takes 260,260
        img = cv2.resize(img, (224, 224)) # ResNet50 takes 224,224
        # No color conversion, because is already made in both the pre-processing
        if img is not None:
            images.append(img)
            label.append(str(dir_name))
            images_names.append(str(filename))

# Conversion into array
images_array = np.array(images)

images_train, images_test, label_train, label_test, validation_images_train, validation_images_test = train_test_split(images_array, label, images_names, 
                                                                                                                       test_size = 0.25)

# Dataset saving: train images, label of the images, validation of the images, label of the images
np.savez_compressed('../Kaggle dataset/FINAL_Marble_Dataset_Validation_224-224', a = images_train, b = label_train, c = images_test, d = label_test, e = validation_images_train, f = validation_images_test)
# For EfficientNetB2
#np.savez_compressed('../Kaggle dataset/FINAL_Marble_Dataset_Validation_260-260', a = images_train, b = label_train, c = images_test, d = label_test, e = validation_images_train, f = validation_images_test)

#%% Test dataset download
# Fix Random Seed for all the code (to reproduce exacly we should fix the environment seed https://stackoverflow.com/questions/50659482/why-cant-i-get-reproducible-results-in-keras-even-though-i-set-the-random-seeds)
# random seeds must be set before importing keras & tensorflow
my_seed = 42
np.random.seed(my_seed)
import random 
random.seed(my_seed)
import tensorflow as tf
tf.random.set_seed(my_seed)

# Folders with the images
img_path_test = '../Kaggle dataset/Marble dataset/test/'

# I load and modify the images for the classification report (in the test folder)
images_test = []
label_test = []
images_names_test = []
img = []
for dir_name in os.listdir(img_path_test):
    for filename in os.listdir(os.path.join(img_path_test, dir_name)):
        img = cv2.imread(os.path.join(img_path_test, os.path.join(dir_name, filename)))
        # Resize images --> ResNet50 and EffNetB0 take 224,224 and EfficientNetB2 takes 260,260
        img = cv2.resize(img, (224, 224)) # ResNet50 takes 224,224
        # No color conversion, because is already made in boththe pre-processing
        if img is not None:
            images_test.append(img)
            label_test.append(str(dir_name))
            images_names_test.append(str(filename))

# Conversion into array
images_test = np.array(images_test)

# Dataset saving: train images, label of the images, validation of the images, label of the images
np.savez_compressed('../Kaggle dataset/FINAL_Marble_Dataset_Test_224-224', a = images_test, b = label_test, c = images_names_test)
# For EfficientNetB2
#np.savez_compressed('../Kaggle dataset/FINAL_Marble_Dataset_Test_260-260', a = images_test, b = label_test, c = images_names_test)
