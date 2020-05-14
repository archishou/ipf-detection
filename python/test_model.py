from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import sys
import librosa
import librosa.display
import numpy as np
import seaborn as sn
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
path = os.path.normpath(os.path.abspath(sys.argv[0]))
path_list = path.split(os.sep)
ipf = os.sep
for item in path_list[0:-2]:
    ipf = os.path.join(ipf, item)
print(ipf)
data_set = os.path.join(ipf, 'datafiles', 'all_data')
model_path = os.path.join(ipf, 'python', 'saved_models', 'weights.final.hdf5')


def main():
    features = []
    for file in os.listdir(data_set):
        if file.endswith(".wav"):
            # print(file)
            class_label = class_name_new(file)
            data_file = os.path.join(data_set, file)
            data = extract_features(data_file)
            features.append([data, class_label])

    # Convert into a Panda dataframe
    featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

    print('Finished feature extraction from ', len(featuresdf), ' files')

    x = np.array(featuresdf.feature.tolist())
    y = np.array(featuresdf.class_label.tolist())

    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y))

    x_train, x_test, y_train, y_test = train_test_split(x, yy, test_size=0.2, random_state=42)

    # split the dataset
    num_rows = 40
    num_columns = 174
    num_channels = 1

    num_labels = yy.shape[1]

    x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
    x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

    # Construct modelas
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(num_labels, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    model.summary()

    model.load_weights(model_path)

    x_all = np.vstack((x_train, x_test))
    y_all = np.vstack((y_train, y_test))

    y_pred_np = np.zeros(shape=(len(y_all)))
    y_train_labels = np.zeros(shape=(len(y_all)))
    ind = 0
    for f in x_all:
        prediction_feature = f.reshape(1, num_rows, num_columns, num_channels)
        predicted_proba_vector = model.predict_classes(prediction_feature)
        y_pred_np[ind] = predicted_proba_vector
        ind = ind + 1
    ind = 0
    for label in y_all:
        if label[0] == 1: y_train_labels[ind] = 0
        elif label[1] == 1: y_train_labels[ind] = 1
        elif label[2] == 1: y_train_labels[ind] = 2
        ind = ind + 1
    cm = confusion_matrix(y_true=y_train_labels, y_pred=y_pred_np)

    df_cm = pd.DataFrame(cm, index=["COPD", "Healthy", "IPF"], columns=["COPD", "Healthy", "IPF"])
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size

    plt.show()


def class_name_new(file):
    if file.startswith("ipf"):
        return "ipf"
    if file.startswith("healthy"):
        return "healthy"
    if file.startswith("copd"):
        return "copd"


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