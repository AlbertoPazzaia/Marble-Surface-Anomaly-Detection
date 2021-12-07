"""
@author: Alberto Pazzaia
Dataset:  https://www.kaggle.com/wardaddy24/marble-surface-anomaly-detection-2 LAST CONSULTATION 08/10/2021
"""

#%% Import Libraries
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Data processing and mathematics
#import numpy as np
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
# For the normal NN
import keras.utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
 
# For the classification report and the confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns # For the figure
import matplotlib.pyplot as plt  # For the figure
#from sklearn.metrics import auc

# Fix Random Seed for all the code (to reproduce exacly we should fix the environment seed https://stackoverflow.com/questions/50659482/why-cant-i-get-reproducible-results-in-keras-even-though-i-set-the-random-seeds)
# random seeds must be set before importing keras & tensorflow
my_seed = 42
np.random.seed(my_seed)
import random 
random.seed(my_seed)
#import tensorflow as tf
tf.random.set_seed(my_seed)

#%% I recover the dataset from the savings
data = np.load('FINAL_Marble_Dataset_Validation_260-260.npz')

images_train = data['a']
label_train = data['b']
images_test = data['c']
label_test = data['d']
validation_images_train = data['e'].tolist()
validation_images_test = data['f'].tolist()

# Label Encoding: convert text/categorical data into numerical data for the label set
le = LabelEncoder() # Create an instance of labelencoder

# Faccio l'encoding nel programma, prima di caricare i file
# Assigning a numerical value to the labels (e.g. 100.0 becomes 0 and 50.0 becomes 1)
le.fit(label_train)
train_labels_encoded = le.transform(label_train)
le.fit(label_test)
test_labels_encoded = le.transform(label_test)

#%% Functions used in the code
def ResNet50_preprocess_data(X, Y):
    X_p = tf.keras.applications.resnet50.preprocess_input(X)
    Y_p = tf.keras.utils.to_categorical(Y, 4)
    return X_p, Y_p

def ResNet50_create_model():
    base_model = tf.keras.applications.ResNet50(include_top = False, weights = "imagenet", input_shape = (224, 224, 3), pooling = max)

    # To define the graph
    inputs = base_model.input
    
    # NO FINE-TUNING
    base_model.trainable = False # Freeze the pretrained weights
    # FINE TUNING:
    #for layer in base_model.layers[:143]: # Freeze up to the layer 143
    # layer.trainable = False
    
    x = tf.keras.layers.Flatten()(base_model.output)
    outputs = tf.keras.layers.Dense(4, activation = 'softmax')(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name = "ResNet50")

    base_learning_rate = 0.00002

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  metrics=['accuracy'])

    return model

def EffNetB0_preprocess_data(X, Y):
    X_p = tf.keras.applications.efficientnet.preprocess_input(X) # Useless: placeholder! (check documentation)
    Y_p = tf.keras.utils.to_categorical(Y, 4)
    return X_p, Y_p

def EffNetB0_create_model():
    base_model = tf.keras.applications.EfficientNetB0(include_top = False, weights = "imagenet", input_shape = (224, 224, 3), drop_connect_rate = 0.4)
    
    # To define the graph
    inputs = base_model.input
    base_model.trainable = False # No Fine-Tuning. Freeze the pretrained weights
    # FINE TUNING:
    #for layer in base_model.layers[-20:]: # Unfreeze for fine-tuning
    #    if not isinstance(layer, tf.keras.layers.BatchNormalization):
    #        layer.trainable = True

    x = tf.keras.layers.Flatten()(base_model.output)
    outputs = tf.keras.layers.Dense(4, activation = 'softmax')(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name = "EffNetB0")

    base_learning_rate = 0.00002

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  metrics=['accuracy'])

    return model

def EffNetB2_preprocess_data(X, Y):
    X_p = tf.keras.applications.efficientnet.preprocess_input(X) # Useless: placeholder! (check documentation)
    Y_p = tf.keras.utils.to_categorical(Y, 4)
    return X_p, Y_p

def EffNetB2_create_model():
    base_model = tf.keras.applications.EfficientNetB2(include_top = False, weights = "imagenet", input_shape = (260, 260, 3), drop_connect_rate = 0.4)

    # To define the graph
    inputs = base_model.input
    # Freeze the pretrained weights
    base_model.trainable = False
    # FINE TUNING
    for layer in base_model.layers[-20:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    x = tf.keras.layers.Flatten()(base_model.output)
    outputs = tf.keras.layers.Dense(4, activation = 'softmax')(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name = "EffNetB2")

    base_learning_rate = 0.0002

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  metrics=['accuracy'])

    return model

def Classic_Deep_Learning_preprocess_data(images_train, images_test, train_labels_encoded, test_labels_encoded):
    # IMAGE PREPROCESSING
    # normalize pixel values
    images_train = images_train.astype('float32') / 255
    images_test = images_test.astype('float32') / 255
    # TEST
    pixels1 = np.asarray(images_train[0])
    # confirm pixel range is 0-255
    print('Data Type: %s' % pixels1.dtype)
    print('Min: %.3f, Max: %.3f' % (pixels1.min(), pixels1.max()))
    # calculate global mean
    mean_images_train = []
    for i in range(len(images_train)):
        mean_images_train.append(images_train[i].mean())
    mean_images_test = []
    for i in range(len(images_test)):
        mean_images_test.append(images_test[i].mean())

    pixels2 = np.asarray(images_train[0]) # Test
    mean = pixels2.mean() # Test
    print('Mean: %.3f' % mean) # Test
    print('Min: %.3f, Max: %.3f' % (pixels2.min(), pixels2.max())) # Test
    pixels2 = pixels2 - mean # Test
    mean = pixels2.mean() # Test
    print('Mean: %.3f' % mean) # Test
    print('Min: %.3f, Max: %.3f' % (pixels2.min(), pixels2.max())) # Test

    # global centering of pixels
    for i in range(len(images_train)):
        images_train[i] = images_train[i] - mean_images_train[i]
    for i in range(len(images_test)):
        images_test[i] = images_test[i] - mean_images_test[i]

    # Label
    Y_train = keras.utils.to_categorical(train_labels_encoded, 4) # To categorical take the encoded version
    Y_test = keras.utils.to_categorical(test_labels_encoded, 4)
    return images_train, images_test, Y_train, Y_test

def Classic_Deep_Learning_create_model1():
    model = Sequential()
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(8, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(4, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # the model so far outputs 3D feature maps (height, width, features)
    model.add(Flatten()) # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(4))
    model.add(Activation('softmax'))
    # COMPILE # E il learning rate?
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def Classic_Deep_Learning_create_model2():
    model = Sequential()    
    model.add(Conv2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(8, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # the model so far outputs 3D feature maps (height, width, features)
    model.add(Flatten()) # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(4))
    model.add(Activation('softmax'))
    # COMPILE 
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#%% Mlflow load data

# Data preprocessing for CNN
#x_train, y_train = EffNetB2_preprocess_data(images_train, train_labels_encoded)
#x_test, y_test = EffNetB2_preprocess_data(images_test, test_labels_encoded)
# Data preprocessing for Classic Deep Learning
x_train, x_test, y_train, y_test = Classic_Deep_Learning_preprocess_data(images_train, images_test, train_labels_encoded, test_labels_encoded)

# to store the data in MLflow
import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient

# Names of the run
experiment_name = 'FINAL Marble Tests'
artifact_repository = './mlflow-run/FINAL Marble Tests'
run_name = 'Baseline: Conv2D (16, 8) Dense(8, 4). adam, categorical_crossentropy, 20 epoch.'

# Provide uri and connect to your tracking server
mlflow.set_tracking_uri('http://127.0.0.1:5000/')
mlflow.get_tracking_uri()
# Initialize Mlflow client!!!
client = MlflowClient()

# If experiment doesn't exist then it will create new
# else it will take the experiment id and will use to run the experiments
try:
    #create experiment
    experiment_id = client.create_experiment(experiment_name, artifact_location= artifact_repository)
except:
    #set the experiment id if it already exists
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    print(experiment_id)

# To autolog the details of the run if I use tensorflow
mlflow.tensorflow.autolog()
#mlflow.tensorflow.autolog(every_n_iter = 1)

# We run the model and save the parameters in MLflow
with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
    # Create and train a new model instance.
    # For CNN
    #model = EffNetB2_create_model()
    # For NN
    model = Classic_Deep_Learning_create_model2()
    
    # Get run id
    run_id = run.info.run_uuid
    
    # Provide notes about the run
    MlflowClient().set_tag(run_id, "mlflow.note.content",
                           "I use Classical Deep Learning with Conv2D (16, 8) and Dense(8, 4) for the Marble slabs. Categorical_crossentropy, adam, batch size = 64. Random Seed: 42")
    
    # Define custom tag
    #tags = {"Application": "Payment Monitoring Platform", "release.candidate": "PMP", 
            #"release.version": "2.2.0"}
    
    # Set Tag
    #mlflow.set_tags(tags)
    
    model.fit(x_train, y_train, batch_size=64, epochs=20, verbose=1, validation_data=(x_test, y_test))
    
    # With only the VALIDATION SET!!
    rounded_predictions_probabilities = model.predict(x_test)
    rounded_predictions = np.argmax(rounded_predictions_probabilities, axis = 1)
    rounded_labels = np.argmax(y_test, axis=1)
    
    # Artifacts creation
    # CONFUSION MATRIX
    # View confusion matrix for test data and predictions
    confusion_mat = confusion_matrix(rounded_labels, rounded_predictions)
    # Plot
    fig = plt.figure(figsize = (16, 7))
    sns.set(font_scale = 1.4)
    sns.heatmap(confusion_mat, annot = True, annot_kws = {'size':26}, cmap = plt.cm.Greens, linewidths = 1)
    class_names = ['crack', 'dot', 'good', 'joint']
    tick_marks = np.arange(len(class_names)) + 0.5
    plt.xticks(tick_marks, class_names, rotation = 25)
    plt.yticks(tick_marks, class_names, rotation = 0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix of the Validation set of the Marble Dataset')
    fig.show()
    fig.savefig('Confusion Matrix.png')
    
    # CLASSIFICATION REPORT
    report = classification_report(rounded_labels, rounded_predictions, output_dict=True)
    classification_report_test = pd.DataFrame(report).transpose()
    # Dataframe creation
    classification_report_test.to_excel(r'../Kaggle dataset/classification_report.xlsx', index = True)
    
    # Model save
    model.save('FINAL_CNN_model.h5') # Per il marble dataset
    
    # Log dataframe details as artifacts
    mlflow.log_artifact('../Kaggle dataset/Confusion Matrix.png')
    mlflow.log_artifact('../Kaggle dataset/Classification_report.xlsx')
    mlflow.log_artifact('../Kaggle dataset/FINAL_CNN_model.h5')
    
    mlflow.end_run()

#%% TEST SET
# I recover the dataset from the savings
data = np.load('FINAL_Marble_Dataset_Test_224-224.npz')
images_test = data['a']
label_test = data['b']
images_names_test = data['c']

# Label Encoding: convert text/categorical data into numerical data for the label set
le = LabelEncoder() # Create an instance of labelencoder

# Encoding: Assigning a numerical value to the labels
le.fit(label_test)
test_labels_encoded = le.transform(label_test)

# Data preprocessing CNN
#x_test, y_test = EffNetB2_preprocess_data(images_test, test_labels_encoded)
# Data preprocessing NN
x_test, x_test, y_test, y_test = Classic_Deep_Learning_preprocess_data(images_test, images_test, test_labels_encoded, test_labels_encoded)

# I upload the model to test the data
#model = EffNetB2_create_model()
model = Classic_Deep_Learning_create_model1() # Model creation

#model.summary()
# Save in the folder the right weigths from the savings!!!!!
# Weights upload
model.load_weights('../Kaggle dataset/FINAL_CNN_model.h5')
model.fit(x_test, y_test, batch_size=64, epochs=20, verbose=1)

# Test the accuracy with the data I have loaded
loss, acc = model.evaluate(x_test, y_test, verbose = 2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

rounded_predictions_probabilities = model.predict(x_test)
rounded_predictions = np.argmax(rounded_predictions_probabilities, axis = 1)
rounded_labels = np.argmax(y_test, axis=1)

# Artifacts creation
# CONFUSION MATRIX
# View confusion matrix for test data and predictions
confusion_mat = confusion_matrix(rounded_labels, rounded_predictions)
# Plot
fig = plt.figure(figsize = (16, 7))
sns.set(font_scale = 1.4)
sns.heatmap(confusion_mat, annot = True, annot_kws = {'size':26}, cmap = plt.cm.Greens, linewidths = 1)
class_names = ['crack', 'dot', 'good', 'joint']
tick_marks = np.arange(len(class_names)) + 0.5
plt.xticks(tick_marks, class_names, rotation = 25)
plt.yticks(tick_marks, class_names, rotation = 0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix of the Test set of the Marble Dataset')
fig.show()
fig.savefig('Confusion Matrix.png')

# CLASSIFICATION REPORT
report = classification_report(rounded_labels, rounded_predictions, output_dict=True)
classification_report_test = pd.DataFrame(report).transpose()
# Dataframe creation
classification_report_test.to_excel(r'../Kaggle dataset/classification_report.xlsx', index = True)

#%% SVM as classifier of the CNN output vector with the TEST SET
data = np.load('FINAL_Marble_Dataset_Test_224-224.npz')

images_test = data['a']
label_test = data['b']
images_names_test = data['c']

data = np.load('FINAL_Marble_Dataset_Validation_224-224.npz')

images_train = data['a']
label_train = data['b']
validation_images_train = data['e'].tolist()

# Label Encoding: convert text/categorical data into numerical data for the label set
le = LabelEncoder() # Create an instance of labelencoder

# Faccio l'encoding nel programma, prima di caricare i file
# Assigning a numerical value to the labels (e.g. 100.0 becomes 0 and 50.0 becomes 1)
le.fit(label_test)
test_labels_encoded = le.transform(label_test)
le.fit(label_train)
train_labels_encoded = le.transform(label_train)

# Data preprocessing train data (named Validation)
x_train, y_train = ResNet50_preprocess_data(images_train, train_labels_encoded)

# I upload the model to test the data
model = base_model = tf.keras.applications.ResNet50(include_top = False, # I do not use the classification layers
                                                    weights = "imagenet", 
                                                    input_shape = (224, 224, 3), 
                                                    pooling = max) # Model creation

# I create a new Resnet model with freezed weights
model.trainable = False
#model.summary()

# Feature extraction from train images
import datetime
start = datetime.datetime.now()
# Fatto con tutte le immagini e non solo il training set. Va bene anche con più label SVM?
# Feature extraction with transfer learning
feature_extractor = model.predict(x_train)
end = datetime.datetime.now()
print("Total features extraction time is: ", end-start)

# Reshape of the features (the same as flatten in the CNN)
features = feature_extractor.reshape(feature_extractor.shape[0], -1)

# Preprocess of test images
x_test, y_test = ResNet50_preprocess_data(images_test, test_labels_encoded)

# Send test data to the same feature extractor process
import datetime
start = datetime.datetime.now()
test_feature = model.predict(x_test)
end = datetime.datetime.now()
print("Total test features extraction time is: ", end-start)

x_test_feature = test_feature.reshape(test_feature.shape[0], -1)

### SVM
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import datetime
start = datetime.datetime.now()
# It is possible to insert different combination of parametrs in param_grid and test them simultaneously
param_grid={'C':[1],'gamma':[0.0001],'kernel':['linear']}
svc=svm.SVC(probability=True)
SVM_model=GridSearchCV(svc,param_grid)
# SVM construction
SVM_model.fit(features, train_labels_encoded)
end = datetime.datetime.now()
print("Total SVM model construction Public Dataset time is: ", end-start)

print('The Model is trained well with the given images')
SVM_model.best_params_ # Contains the best parameters obtained from GridSearchCV

# Test of the SVM with the test iages 
y_pred=SVM_model.predict(x_test_feature)
print("The predicted Data is :")
print(y_pred)
print("The actual data is:")
print(np.array(test_labels_encoded))

from sklearn import metrics
print('Accuracy = ', metrics.accuracy_score(test_labels_encoded, y_pred))

### Diagnostic tools for the interpretation:
from sklearn.metrics import confusion_matrix, classification_report  
import seaborn as sns # For the figure
import matplotlib.pyplot as plt  # For the figure  
# CONFUSION MATRIX
# View confusion matrix for test data and predictions
confusion_mat = confusion_matrix(test_labels_encoded, y_pred)
# Plot
fig = plt.figure(figsize = (16, 7))
sns.set(font_scale = 1.4)
sns.heatmap(confusion_mat, annot = True, annot_kws = {'size':26}, cmap = plt.cm.Greens, linewidths = 1)
class_names = ['crack', 'dot', 'good', 'joint']
tick_marks = np.arange(len(class_names)) + 0.5
plt.xticks(tick_marks, class_names, rotation = 25)
plt.yticks(tick_marks, class_names, rotation = 0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix of the Test set of the Marble Dataset with SVM')
fig.show()
fig.savefig('Confusion Matrix.png')

# CLASSIFICATION REPORT
report = classification_report(test_labels_encoded, y_pred, output_dict=True)
classification_report_test = pd.DataFrame(report).transpose()
# Dataframe creation
classification_report_test.to_excel(r'../Kaggle dataset/classification_report.xlsx', index = True)

# PRECISION-RECALL CURVE for MULTI-LABEL CLASSIFICATION
# ...Missing

#%% Random Forest
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators=30)

# Train the model on training data
import datetime
start = datetime.datetime.now()
RF_model.fit(features, train_labels_encoded)
end = datetime.datetime.now()
print("Total Random Forest Public Dataset model constrution time is: ", end-start)

# predict_RF = RF_model.predict_proba(x_test_feature) # sostituito
predict_RF = RF_model.predict(x_test_feature)

# Print the overall accuracy
from sklearn import metrics
print('Accuracy = ', metrics.accuracy_score(test_labels_encoded, predict_RF))

# CONFUSION MATRIX
# View confusion matrix for test data and predictions
confusion_mat = confusion_matrix(test_labels_encoded, predict_RF)
# Plot
fig = plt.figure(figsize = (16, 7))
sns.set(font_scale = 1.4)
sns.heatmap(confusion_mat, annot = True, annot_kws = {'size':26}, cmap = plt.cm.Greens, linewidths = 1)
class_names = ['crack', 'dot', 'good', 'joint']
tick_marks = np.arange(len(class_names)) + 0.5
plt.xticks(tick_marks, class_names, rotation = 25)
plt.yticks(tick_marks, class_names, rotation = 0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix of the Test set of the Marble Dataset with SVM')
fig.show()
fig.savefig('Confusion Matrix.png')

# CLASSIFICATION REPORT
report = classification_report(test_labels_encoded, predict_RF, output_dict=True)
classification_report_test = pd.DataFrame(report).transpose()
# Dataframe creation
classification_report_test.to_excel(r'../Kaggle dataset/classification_report.xlsx', index = True)

#%% PCA reduction of the features
# I try to make again the SVM reducing the number of feature extracted from each image
# Feature compression libriries
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

pca = PCA(n_components=680) # n_components == min(n_samples, n_features)
pca.fit(features)
fig = plt.figure(figsize = (16, 7))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Cum variance")
fig.show()

import datetime
start = datetime.datetime.now()
PCA_features = pca.fit_transform(features)
test_PCA_features = pca.transform(x_test_feature) # Test features
end = datetime.datetime.now()
print("Total PCA compression to 680 features Public Dataset time is: ", end-start)

#%% SVM with PCA
from sklearn import svm
from sklearn.model_selection import GridSearchCV
# I canuse different parameters
param_grid={'C':[1],'gamma':[0.0001],'kernel':['linear']}
svc=svm.SVC(probability=True)
model=GridSearchCV(svc,param_grid)

import datetime
start = datetime.datetime.now()
#Fit the model. Do not forget to use on-hot-encoded Y values.
model.fit(PCA_features, train_labels_encoded)
end = datetime.datetime.now()
print("Total execution time of SVM model construction classifier with PCA is: ", end-start)

print('The Model is trained well with the given images')
model.best_params_ # Contains the best parameters obtained from GridSearchCV

# model testing
y_pred=model.predict(test_PCA_features)

#print("The predicted Data is :")
#print(y_pred)
#print("The actual data is:")
#print(np.array(test_labels_encoded))
#from sklearn import metrics
#print('Accuracy = ', metrics.accuracy_score(test_labels_encoded, y_pred))

### Diagnostic tools for the interpretation:
# Confusion Matrix
confusion_mat = confusion_matrix(test_labels_encoded, y_pred)
# Plot
fig = plt.figure(figsize = (16, 7))
sns.set(font_scale = 1.4)
sns.heatmap(confusion_mat, annot = True, annot_kws = {'size':26}, cmap = plt.cm.Greens, linewidths = 1)
class_names = ['crack', 'dot', 'good', 'joint']
tick_marks = np.arange(len(class_names)) + 0.5
plt.xticks(tick_marks, class_names, rotation = 25)
plt.yticks(tick_marks, class_names, rotation = 0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix of the Test set of the Marble Dataset with SVM')
fig.show()
fig.savefig('Confusion Matrix.png')

# CLASSIFICATION REPORT
report = classification_report(test_labels_encoded, y_pred, output_dict=True)
classification_report_test = pd.DataFrame(report).transpose()
# Dataframe creation
classification_report_test.to_excel(r'../Kaggle dataset/classification_report.xlsx', index = True)