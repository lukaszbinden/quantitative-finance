# LZ: code from:
# https://github.com/hudson-and-thames/research/blob/master/Chapter3/2019-03-06_JJ_Meta-Labels-MNIST.ipynb

# from IPython.core.display import Image, display
# display(Image('https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Precisionrecall.svg/660px-Precisionrecall.svg.png', width=300, unconfined=True))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import itertools
import seaborn as sns
from subprocess import check_output
from sklearn import metrics as sk_metrics
from sklearn.metrics import confusion_matrix

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# %matplotlib inline


def plot_roc(actual, prediction):
    # Calculate ROC / AUC
    fpr, tpr, thresholds = sk_metrics.roc_curve(actual, prediction, pos_label=1)
    roc_auc = sk_metrics.auc(fpr, tpr)

    # Plot
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Example')
    plt.legend(loc="lower right")
    plt.show()


def test_meta_label(primary_model, secondary_model, x, y, threshold):
    """
    Function outputs the results of the primary model with a threshold of 50%. It then outputs the results of the meta model.
    Ideally you want to see that the meta model out performs the primary model.

    I am busy investigating why meta modeling works. A little tricky since I'm yet to find a solid paper on the technique. Its very briefly mentioned in
    Advances in Financial Machine Learning.

    :param primary_model: model object (First, we build a model that achieves high recall, even if the precision is not particularly high)
    :param secondary_model: model object (the role of the secondary ML algorithm is to determine whether a positive from the primary (exogenous) model
                            is true or false. It is not its purpose to come up with a betting opportunity. Its purpose is to determine whether
                            we should act or pass on the opportunity that has been presented.)
    :param x: Explanatory variables
    :param y: Target variable (One hot encoded)
    :param threshold: The confidence threshold. This is used
    :return: Print the classification report for both the base model and the meta model.
    """
    # Get the actual labels (y) from the encoded y labels
    actual = np.array([i[1] for i in y]) == 1

    # Use primary model to score the data x
    primary_prediction = primary_model.predict(x)
    primary_prediction = np.array([i[1] for i in primary_prediction]).reshape((-1, 1))
    primary_prediction_int = primary_prediction > threshold  # binary labels

    # Print output for base model
    print('Base Model Metrics:')
    print(sk_metrics.classification_report(actual, primary_prediction > 0.50))
    print('Confusion Matrix')
    print(sk_metrics.confusion_matrix(actual, primary_prediction_int))
    accuracy = (actual == primary_prediction_int.flatten()).sum() / actual.shape[0]
    print('Accuracy: ', round(accuracy, 4))
    print('')

    # Secondary model
    new_features = np.concatenate((primary_prediction_int, x), axis=1)

    # Use secondary model to score the new features
    meta_prediction = secondary_model.predict(new_features)
    meta_prediction = np.array([i[1] for i in meta_prediction])
    meta_prediction_int = meta_prediction > 0.5 # binary labels

    # Now combine primary and secondary model in a final prediction
    final_prediction = (meta_prediction_int & primary_prediction_int.flatten())

    # Print output for meta model
    print('Meta Label Metrics: ')
    print(sk_metrics.classification_report(actual, final_prediction))
    print('Confusion Matrix')
    print(sk_metrics.confusion_matrix(actual, final_prediction))
    accuracy = (actual == final_prediction).sum() / actual.shape[0]
    print('Accuracy: ', round(accuracy, 4))


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Plot the distribution of numbers
    showDistr = False
    if showDistr:
        cnt = Counter(y_test)
        print(cnt)
        sns.countplot(y_test)
        plt.title('Distribution of Images')
        plt.xlabel('Images')
        plt.show()

    # Normalising the data
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Change these params if you want to change the numbers selected
    num1 = 3
    num2 = 5

    # Subset on only two numbers
    x_sub_train = x_train[(y_train == num1) | (y_train == num2)]
    y_sub_train = y_train[(y_train == num1) | (y_train == num2)]

    x_sub_test = x_test[(y_test == num1) | (y_test == num2)]
    y_sub_test = y_test[(y_test == num1) | (y_test == num2)]

    print('X values')
    print('x_train', x_sub_train.shape)
    print('x_test', x_sub_test.shape, '\n')
    print('Y values')
    print('y train', y_sub_train.shape)
    print('y test', y_sub_test.shape)

    # Flatten input
    print(type(x_sub_train))
    print(x_sub_train.shape)
    x_train_flat = x_sub_train.flatten().reshape(x_sub_train.shape[0], 28*28)
    x_test_flat = x_sub_test.flatten().reshape(x_sub_test.shape[0], 28*28)

    # One hot encode target variables
    print(y_sub_train[0:3])
    y_sub_train_encoded = to_categorical([1 if value == num1 else 0 for value in y_sub_train])
    print(y_sub_train_encoded[0:3])

    # Test train split
    X_train, X_val, Y_train, Y_val = train_test_split(x_train_flat, y_sub_train_encoded, test_size=0.1, random_state=42)

    # Build primary model
    model = Sequential()
    model.add(Dense(units=2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    hist = model.fit(x=X_train, y=Y_train, validation_data=(X_val, Y_val), epochs=3, batch_size=320)  # batch size is so large so that the model can be poorly fit, Its easy to get 99% accuracy.
    print("history for model fit [acc]....: %s" % str(hist.history['acc']))
    print("history for model fit [va_acc].: %s" % str(hist.history['val_acc']))


    # Plot ROC
    prediction = model.predict(X_train)
    prediction = np.array([i[1] for i in prediction])
    actual = np.array([i[1] for i in Y_train]) == 1

    # plot_roc(actual, prediction)

    # Create a model with high recall, change the threshold until a good recall level is reached
    threshold = .60
    prediction_int = np.array(prediction) > threshold

    # Classification report
    print('Classification report:___________________________________')
    print(sk_metrics.classification_report(actual, prediction_int))

    # Confusion matrix
    cm = sk_metrics.confusion_matrix(actual, prediction_int)
    print('Confusion Matrix:___________________________________')
    print(cm)

    # Get meta labels
    meta_labels = prediction_int & actual
    meta_labels_encoded = to_categorical(meta_labels)

    # Reshape data
    prediction_int = prediction_int.reshape((-1, 1))

    # MNIST data + forecasts_int
    new_features = np.concatenate((prediction_int, X_train), axis=1)

    # Train a new model
    # Build model
    meta_model = Sequential()
    meta_model.add(Dense(units=2, activation='softmax'))

    meta_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
    hist = meta_model.fit(x=new_features, y=meta_labels_encoded, epochs=4, batch_size=32)
    print("history for META_model fit [acc]....: %s" % str(hist.history['acc']))

    print('test_meta_label TRAINING -->')
    test_meta_label(primary_model=model, secondary_model=meta_model, x=X_train, y=Y_train, threshold=threshold)

    print('test_meta_label TEST -->')
    test_meta_label(primary_model=model, secondary_model=meta_model, x=X_val, y=Y_val, threshold=threshold)

    print("evaluate hold out sample -->")

    # Flatten input
    x_test_flat = x_sub_test.flatten().reshape(x_sub_test.shape[0], 28*28)

    # One hot encode target variables
    y_sub_test_encoded = to_categorical([1 if value == num1 else 0 for value in y_sub_test])

    test_meta_label(primary_model=model, secondary_model=meta_model, x=x_test_flat, y=y_sub_test_encoded, threshold=threshold)


