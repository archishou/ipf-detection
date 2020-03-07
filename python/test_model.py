from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint
from datetime import datetime
import os
import librosa
import librosa.display
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics

from python.helpers.wavhelper import WavFileHelper

path = '/Users/Archish/Documents/CodeProjects/Python/IPF/python/saved_models/weights.best.basic_cnn.hdf5'


def create_model():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, input_shape=(40, 174, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(3, activation='softmax'))
    return model

def extract_features(file_name):
    max_pad_len = 174
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return mfccs

def load_trained_model(weights_path):
   model = create_model()
   model.load_weights(weights_path)
   return model


def main():
    model = load_trained_model(path)
    file = input("enter file path: ")
    file = os.path.join('/Users/Archish/Documents/CodeProjects/Python/IPF/datafiles/all_data/', file)
    prediction_feature = extract_features(file)
    prediction_feature = prediction_feature.reshape(1, 40, 174, 1)
    le = LabelEncoder()
    predicted_vector = model.predict_classes(prediction_feature)
    predicted_class = le.inverse_transform(predicted_vector)
    print("The predicted class is:", predicted_class[0], '\n')

    predicted_proba_vector = model.predict_proba(prediction_feature)
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)):
        category = le.inverse_transform(np.array([i]))
        print(category[0], "\t\t : ", format(predicted_proba[i], '.32f'))
if __name__ == '__main__':
    main()