from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import pandas as pd
from keras.callbacks import ModelCheckpoint
from datetime import datetime
import os
import librosa
import librosa.display
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

data_set = "/Users/Archish/Documents/CodeProjects/Python/IPF/datafiles/all_data"
model_path = "/Users/Archish/Documents/CodeProjects/Python/IPF/python/saved_models/weights.best.cnn.hdf5"


def main():
    features = []
    for file in os.listdir(data_set):
        if file.endswith(".wav"):
            # print(file)
            class_label = class_name(file)
            data_file = os.path.join(data_set, file)
            data = extract_features(data_file)
            features.append([data, class_label])

    # Convert into a Panda dataframe
    featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

    print('Finished feature extraction from ', len(featuresdf), ' files')

    x = np.array(featuresdf.feature.tolist())
    y = np.array(featuresdf.class_label.tolist())

    # Encode the classification labels
    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y))

    # split the dataset
    num_rows = 40
    num_columns = 174
    num_channels = 1

    num_labels = yy.shape[1]

    # Construct modelas
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(GlobalAveragePooling2D())

    model.add(Dense(num_labels, activation='softmax'))

    # Display model architecture summary
    model.summary()

    model.load_weights(model_path)

    while True:
        file = input("choose a file: ") + ".wav"

        prediction_feature = extract_features(os.path.join(data_set, file))
        prediction_feature = prediction_feature.reshape(1, num_rows, num_columns, num_channels)

        predicted_vector = model.predict_classes(prediction_feature)
        predicted_class = le.inverse_transform(predicted_vector)
        print("The predicted class is:", predicted_class[0], '\n')

        predicted_proba_vector = model.predict_proba(prediction_feature)
        predicted_proba = predicted_proba_vector[0]
        for i in range(len(predicted_proba)):
            category = le.inverse_transform(np.array([i]))
            print(category[0], "\t\t : ", format(predicted_proba[i], '.32f'))


def class_name(file):
    if file.startswith("ipf"):
        return "ipf"
    if file.startswith("healthy"):
        return "not_ipf"
    if file.startswith("copd"):
        return "not_ipf"


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

if __name__ == '__main__':
    main()
