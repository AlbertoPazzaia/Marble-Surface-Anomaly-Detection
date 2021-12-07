# -*- coding: utf-8 -*-
"""
@author: Alberto Pazzaia
Dataset:  https://www.kaggle.com/wardaddy24/marble-surface-anomaly-detection-2 LAST CONSULTATION 08/10/2021
"""

#%% Libraries
import tensorflow as tf
# To load and visualize data and images
import cv2
import numpy as np
from IPython.display import Image, display
import matplotlib.cm as cm

# For the classification report and the confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns # For the figure
import matplotlib.pyplot as plt  # For the figure

# For Marble dataset
import os # To select elements in folders
from sklearn.preprocessing import LabelEncoder # To encode the label values

# Fix Random Seed for all the code (to reproduce exacly we should fix the environment seed https://stackoverflow.com/questions/50659482/why-cant-i-get-reproducible-results-in-keras-even-though-i-set-the-random-seeds)
# random seeds must be set before importing keras & tensorflow
my_seed = 42
np.random.seed(my_seed)
import random 
random.seed(my_seed)
#import tensorflow as tf
tf.random.set_seed(my_seed)

#%% Functions
# Model to load the saved weights
def return_trained_model_ResNet50(fold=0):

    base_model = tf.keras.applications.ResNet50(include_top = False, weights = "imagenet", input_shape = (224, 224, 3), pooling = max) 
    # To define the graph
    inputs = base_model.input
    # Freeze the pretrained weights
    for layer in base_model.layers[:143]:
      layer.trainable = False

    x = tf.keras.layers.Flatten()(base_model.output)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    
    # Compile
    model = tf.keras.Model(inputs, outputs, name="ResNet_MOD")
    
    return model

def return_trained_model_EfficientNetB2(fold=0):

    base_model = tf.keras.applications.EfficientNetB2(include_top = False, weights = "imagenet", input_shape = (260, 260, 3), drop_connect_rate = 0.4) 
    # To define the graph
    inputs = base_model.input
    # Freeze the pretrained weights
    base_model.trainable = False
    for layer in base_model.layers[-20:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
            
    x = tf.keras.layers.Flatten()(base_model.output)
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
    
    # Compile
    model = tf.keras.Model(inputs, outputs, name="EffNetB2_MOD")
    
    return model

def return_trained_model_EfficientNetB0(fold=0):

    base_model = tf.keras.applications.EfficientNetB0(include_top = False, 
                                                      weights = "imagenet", 
                                                      input_shape = (224, 224, 3), 
                                                      drop_connect_rate = 0.4) 
    # To define the graph
    inputs = base_model.input
    # Freeze the pretrained weights
    base_model.trainable = False
    for layer in base_model.layers[-20:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
            
    x = tf.keras.layers.Flatten()(base_model.output)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    
    # Compile
    model = tf.keras.Model(inputs, outputs, name="EffNetB0_MOD")
    
    return model

def preprocess_data(X, Y):
    #X_p = tf.keras.applications.resnet50.preprocess_input(X)
    X_p = tf.keras.applications.efficientnet.preprocess_input(X) # Inutile! (controllare documentazione)
    Y_p = tf.keras.utils.to_categorical(Y, 4)
    return X_p, Y_p

def preprocess_data_ID(X, Y):
    #X_p = tf.keras.applications.resnet50.preprocess_input(X)
    X_p = tf.keras.applications.efficientnet.preprocess_input(X) # Inutile! (controllare documentazione)
    Y_p = tf.keras.utils.to_categorical(Y, 2)
    return X_p, Y_p

# GradCam
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# To plot the cam with the heatmap
def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=4):
    # Load the original image
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))

#%% Load and GradCam of the Marble Dataset
model = return_trained_model_EfficientNetB2()
model.summary()

# Carico i pesi del modello
model.load_weights('../Kaggle dataset/my_model.h5')

# Load the test dataset
img_path_test = '../Kaggle dataset/Marble dataset/test/'
images_for_pred = []
label_for_pred = []
images_names_for_pred = []
img = []
for dir_name in os.listdir(img_path_test):
    for filename in os.listdir(os.path.join(img_path_test, dir_name)):
        img = cv2.imread(os.path.join(img_path_test, os.path.join(dir_name, filename)))
        # Resize images --> ResNet50 takes 224,224 and EfficientNetB2 takes 260,260
        img = cv2.resize(img, (260, 260)) # ResNet50 takes 224,224
        # No color conversion, because is already made in boththe pre-processing
        if img is not None:
            images_for_pred.append(img)
            label_for_pred.append(str(dir_name))
            images_names_for_pred.append(os.path.join(img_path_test, os.path.join(dir_name, filename)))

# Conversion into array
images_for_pred = np.array(images_for_pred)

# Label Encoding of the new data: I use the same previous encoding (va bene?)
# Assigning a numerical value to the labels
le = LabelEncoder() # Create an instance of labelencoder
le.fit(label_for_pred)
labels_for_pred_encoded = le.transform(label_for_pred)

x_for_pred, y_for_pred = preprocess_data(images_for_pred, labels_for_pred_encoded)

# Test the accuracy with the data I load
base_learning_rate = 0.00002
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
          optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate), metrics=['accuracy'])
loss, acc = model.evaluate(x_for_pred, y_for_pred, verbose = 2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    
# With only the validation set
rounded_predictions_probabilities = model.predict(x_for_pred)
rounded_predictions = np.argmax(rounded_predictions_probabilities, axis = 1)
rounded_labels = np.argmax(y_for_pred, axis=1)
    
# Confusion matrix for test data and predictions
confusion_mat = confusion_matrix(rounded_labels, rounded_predictions)
# Plot
fig = plt.figure(figsize = (16, 7))
sns.set(font_scale = 1.4)
sns.heatmap(confusion_mat , annot = True, annot_kws = {'size':26}, cmap = plt.cm.Greens, linewidths = 1)
class_names = ['crack', 'dot', 'good', 'joint']
tick_marks = np.arange(len(class_names)) + 0.5
plt.xticks(tick_marks, class_names, rotation = 25)
plt.yticks(tick_marks, class_names, rotation = 0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix of marble dataset.') # modify for quality
fig.show()

#%% Heatmap of the test
last_conv_layer_name = 'top_activation'
# Name of the image text
font = cv2.FONT_ITALIC
fontScale = 0.3
fontColor = (255, 255, 255)
lineType = 0
org = (5, 10)

# To shuffle the position of the images
from sklearn.utils import shuffle
test_names = []
x_for_pred, y_for_pred, test_names = shuffle(x_for_pred, y_for_pred, images_names_for_pred, random_state = 0)

for i in range(len(images_names_for_pred)):
    if images_names_for_pred[i] != 'Augmented':
        # Load and process the image for the heatmap test
        orig = cv2.imread(test_names[i]) #load the original image from disk (in OpenCV format) and then resize the image to its target dimensions
        orig = cv2.resize(orig, (260, 260)) # Resize
        cv2.imwrite('orig.png', orig)
        
        # Processing of the image
        image = np.array(orig)
        image = np.expand_dims(image, axis=0)
        image = tf.keras.applications.efficientnet.preprocess_input(image)

        # Print what the top predicted class is
        preds = model.predict(image)
        
        # Remove last layer's softmax
        model.layers[-1].activation = None

        # Generate class activation heatmap
        heatmap = make_gradcam_heatmap(image, model, last_conv_layer_name)
        
        # Visualization of the image with cv2
        name_test = 'orig.png'
        save_and_display_gradcam(name_test, heatmap)
        final_heatmap = cv2.imread('cam.jpg')

        # Image names and prediction
        predicted_index = np.argmax(preds[0])
        print('Predicted class: ' + str(predicted_index))
        # Real index
        true_class = np.argmax(y_for_pred[i])
        print('True class: ' + str(true_class))
        print('###########################')
                    
        # To see the map and the image side by side
        final_heatmap = cv2.putText(final_heatmap, list(le.classes_)[predicted_index], org, font, fontScale, fontColor, lineType)
        image = cv2.putText(image[0,], list(le.classes_)[true_class], org, font, fontScale, (0, 0, 0), lineType)
        comparison = np.concatenate((image, final_heatmap), axis=1)
        
        # To visualize the image
        cv2.namedWindow('Oculus Images', cv2.WINDOW_NORMAL)
        cv2.imshow('Oculus Images', comparison)
        key = cv2.waitKey(0)        
        if key == 27: # if ESC is pressed, exit loop (why with 27 I go out of the loop?)
            print("ESC pressed, ending loop")
            cv2.destroyAllWindows() # Destroys all of the HighGUI windows
            break
