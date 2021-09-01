
import tensorflow as tf

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten,\
 Conv2D, MaxPooling2D,BatchNormalization,LayerNormalization
from tensorflow import keras
from keras.preprocessing import image 
import json
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing import image 
import os
import json
import matplotlib.pyplot as plt

import seaborn as sns

from keras.preprocessing import image

DATA_PATH = "data.json"
# SAVED_MODEL_PATH = "model3.h5"
SAVED_MODEL_PATH = "model030.h5"

EPOCHS = 10
BATCH_SIZE = 32
PATIENCE = 5
LEARNING_RATE = 0.0001
import matplotlib.pyplot as plt

def plot_history(history):

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="accuracy")
    axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")

    # create loss subplot
    axs[1].plot(history.history["loss"], label="loss")
    axs[1].plot(history.history['val_loss'], label="val_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evaluation")

    plt.show()
def load_data(data_path):
    """Loads training dataset from json file.
    :param data_path (str): Path to json file containing data
    :return X (ndarray): Inputs
    :return y (ndarray): Targets
    """
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    print("Training sets loaded!")
    return X, y


def prepare_dataset(data_path, test_size=0.2, validation_size=0.2):
    """Creates train, validation and test sets.
    :param data_path (str): Path to json file containing data
    :param test_size (flaot): Percentage of dataset used for testing
    :param validation_size (float): Percentage of train set used for cross-validation
    :return X_train (ndarray): Inputs for the train set
    :return y_train (ndarray): Targets for the train set
    :return X_validation (ndarray): Inputs for the validation set
    :return y_validation (ndarray): Targets for the validation set
    :return X_test (ndarray): Inputs for the test set
    :return X_test (ndarray): Targets for the test set
    """

    # load dataset
    X, y = load_data(data_path)

    # create train, validation, test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # add an axis to nd array
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]

    return X_train, y_train, X_validation, y_validation, X_test, y_test

def build_model( loss="sparse_categorical_crossentropy", learning_rate=0.0001):

    model = Sequential()
    
    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(227,227,3), kernel_size=(11,11),\
     strides=(4,4), padding='valid'))
    model.add(Activation('relu'))
    # Pooling 
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())
    
    
    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    
    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    
    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    
    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu',name="my_intermediate_layer"))
    # Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    
    # Passing it to a dense layer
    model.add(Flatten())
    
    
    # 1st Dense Layer
    model.add(Dense(4096, input_shape=(224*224*3,)))
    model.add(Activation('relu'))
    
    
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.5))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    
    # 2nd Dense Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    
    # Add Dropout
    model.add(Dropout(0.5))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    #  output Layer 
    model.add(Dense(7))
    
    model.add(Activation('softmax'))
    optimiser = tf.optimizers.Adam(learning_rate=learning_rate)

    # compile model
    model.compile(optimizer=optimiser,
                  loss=loss,
                  metrics=["accuracy"])

    model.summary()
    return model
def plot_history(history):

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="accuracy")
    axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")

    # create loss subplot
    axs[1].plot(history.history["loss"], label="loss")
    axs[1].plot(history.history['val_loss'], label="val_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evaluation")

    plt.show()

def train(model, epochs, batch_size, patience, X_train, y_train, X_validation, y_validation):

    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor="accuracy", min_delta=0.001, patience=patience)

    # train model
    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_validation, y_validation),
                        callbacks=[earlystop_callback])
    return history
# feature_extractor = keras.Model(
#     inputs=model.inputs,
#     outputs=model.get_layer(name="my_intermediate_layer").output,
# )
# Call feature extractor on test input.

# print(x)
# x = preprocess_input(x) 
def main():
    import pickle
    commands = np.array(tf.io.gfile.listdir(str("/home/akram/Alphabet1/output1/train")))
    print('Commands:', commands)
    pickle_in = open("xTrain.pickle","rb")
    XTrain = pickle.load(pickle_in)
    
    pickle_in = open("yTrain.pickle","rb")
    yTrain = pickle.load(pickle_in)
    
    
    pickle_in = open("XTest.pickle","rb")
    XTest = pickle.load(pickle_in)
    pickle_in = open("yTest.pickle","rb")
    yTest = pickle.load(pickle_in)
    
    pickle_in = open("xVal.pickle","rb")
    XVal= pickle.load(pickle_in)
    pickle_in = open("yVal.pickle","rb")
    yVal = pickle.load(pickle_in)
    

    print(XTrain.shape)
    input_shape = (XTrain.shape[1], XTrain.shape[2], 3)
    print(input_shape)
    XTrain = np.array(XTrain)
    yTrain= np.array(yTrain)
    XTest = np.array(XTest)
    yTest= np.array(yTest)
    XVal = np.array(XVal)
    yVal= np.array(yVal)
    # XTrain = XTrain/255.0
    # XTest = XTest/255.0
    # XVal = XVal/255.0
    input_shape = (XTrain.shape[1], XTrain.shape[2], 3)
    print(input_shape)
    model = build_model( learning_rate=LEARNING_RATE)
    history = train(model, EPOCHS, BATCH_SIZE, PATIENCE,
                    XTrain, yTrain, XVal, yVal)
    
    plot_history(history)
    # evaluate network on test set
    test_loss, test_acc = model.evaluate(XTest, yTest)
    print("\nTest loss: {}, test accuracy: {}".format(test_loss, 100*test_acc))
    y_pred = np.argmax(model.predict(XTest), axis=1)
    y_true = yTest
    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'Test set accuracy: {test_acc:.0%}')
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred) 
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, xticklabels=commands, yticklabels=commands, 
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()

    model.save(SAVED_MODEL_PATH)

if __name__ == "__main__":
    main()