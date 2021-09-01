
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
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten,\
 Conv2D, MaxPooling2D,BatchNormalization,LayerNormalization
from tensorflow import keras
from keras.preprocessing import image 
import json
from sklearn.model_selection import train_test_split
import pickle

import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten,\
    Conv2D, MaxPooling2D, BatchNormalization, LayerNormalization
from tensorflow import keras
from tensorflow.keras.models import Sequential
import seaborn as sns

from keras.preprocessing import image
import json
import joblib

DATASET_PATH ="/Alphabet1/dadaset0"

DATA_PATH = "data2.json"
val ="val.json"
train="train.json"
test="test.json"
SAVED_MODEL_PATH = "model02.h5"
EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 5
LEARNING_RATE = 0.001


def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    print("Training sets loaded!")
    return X, y


def prepare_dataset(data_path, test_size=0.2, validation_size=0.2):


    # load dataset
    X, y = load_data(data_path)
    # X_test, y_test = load_data("test.json")
    # X_validation,y_validation=load_data("val.json")


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=validation_size)

    # add an axis to nd array
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]

    return X_train, y_train, X_validation, y_validation, X_test, y_test


def build_model(input_shape, loss="sparse_categorical_crossentropy", learning_rate=0.0001):


    model = tf.keras.models.Sequential()

    # 1st conv layer
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape,
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'))

    # 2nd conv layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                      kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'))

    # 3rd conv layer
    model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu',name="my_intermediate_layer",
                                      kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(
        (2, 2), strides=(2, 2), padding='same'))

    # flatten 
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    tf.keras.layers.Dropout(0.8)

    # softmax 
    model.add(tf.keras.layers.Dense(28, activation='softmax'))

    optimiser = tf.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimiser,
                  loss=loss,
                  metrics=["accuracy"])

    model.summary()

    return model


def train(model, epochs, batch_size, patience, X_train, y_train, X_validation, y_validation):

    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor="accuracy", min_delta=0.001, patience=patience)

    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_validation, y_validation),
                        callbacks=[earlystop_callback])
    return history


def plot_history(history):

    fig, axs = plt.subplots(2)

    #  accuracy subplot
    axs[0].plot(history.history["accuracy"], label="accuracy")
    axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")

    #  loss subplot
    axs[1].plot(history.history["loss"], label="loss")
    axs[1].plot(history.history['val_loss'], label="val_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evaluation")

    plt.show()


def main():
    commands = np.array(tf.io.gfile.listdir(str(DATASET_PATH)))
    print('Commands:', commands)

    X_train, y_train, X_validation, y_validation, X_test, y_test = prepare_dataset(
        DATA_PATH)
    print(X_train.shape)
    print(X_test.shape)
    print(X_validation.shape)

    X_train, X_test = X_train / 255.0, X_test / 255.0
    X_validation=X_validation/255.0
    # pickle_in = open("X.pickle","rb")
    # X = pickle.load(pickle_in)
    
    # pickle_in = open("y.pickle","rb")
    # y = pickle.load(pickle_in)
    
    # # X = X/255.0
    # X = np.array(X)
    # y= np.array(y)
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2)
    # X_train, X_validation, y_train, y_validation = train_test_split(
    # X_train, y_train, test_size=0.2)
    # X_train, X_test = X_train / 255.0, X_test / 255.0
    # create network
    
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    print(input_shape)
    model = build_model(input_shape, learning_rate=LEARNING_RATE)
#     feature_extractor = keras.Model(
#     inputs=model.inputs,
#     outputs=model.get_layer(name="my_intermediate_layer").output,
# )
#     features = feature_extractor(X_train)

#     features = features.reshape(features.shape[0], -1)




    # # train network
    history = train(model, EPOCHS, BATCH_SIZE, PATIENCE,
                   X_train, y_train, X_validation, y_validation)

    # plot accuracy/loss for training/validation set as a function of the epochs
    plot_history(history)

    # evaluate network on test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("\nTest loss: {}, test accuracy: {}".format(test_loss, 100*test_acc))
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = y_test
    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'Test set accuracy: {test_acc:.0%}')
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred) 
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, xticklabels=commands, yticklabels=commands, 
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()
    
    
    # predict probabilities for test set
    yhat_probs = model.predict(X_test, verbose=0)
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(X_test, verbose=0)
    # reduce to 1d array
    # yhat_probs = yhat_probs[:, 0]
    # yhat_classes = yhat_classes[:, 0]
    
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test, yhat_classes, pos_label='positive',
                                           average='micro')
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, yhat_classes, pos_label='positive',
                                           average='micro')
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, yhat_classes, pos_label='positive',
                                           average='micro')
    print('F1 score: %f' % f1)
    
    # kappa
    kappa = cohen_kappa_score(y_test, yhat_classes
                                           )
    print('Cohens kappa: %f' % kappa)
    # ROC AUC
    auc = roc_auc_score(y_test, yhat_probs, multi_class='ovr'
                                         )
    print('ROC AUC: %f' % auc)

    # save model
    model.save(SAVED_MODEL_PATH)
    joblib.dump(lr, 'model.pkl')

if __name__ == "__main__":
    main()
