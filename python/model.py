from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import pandas as pd
from keras.callbacks import ModelCheckpoint
from datetime import datetime
import seaborn as sn
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
import os
import librosa
import json
import librosa.display
import pickle
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

data_set = "/Users/Archish/Documents/CodeProjects/Python/IPF/datafiles/all_data"


def main():
    features = []
    for file in os.listdir(data_set):
        if file.endswith(".wav"):
            class_label = class_name(file)
            data_file = os.path.join(data_set, file)

            audio, sample_rate = load_audio(data_file)

            raw_data = extract_features(audio, sample_rate)

            shift_1 = extract_features(augment_shift(audio, sample_rate, 1, 'both'), sample_rate)
            shift_2 = extract_features(augment_shift(audio, sample_rate, 2, 'both'), sample_rate)
            shift_3 = extract_features(augment_shift(audio, sample_rate, 3, 'both'), sample_rate)
            shift_4 = extract_features(augment_shift(audio, sample_rate, 4, 'both'), sample_rate)
            shift_5 = extract_features(augment_shift(audio, sample_rate, 5, 'both'), sample_rate)
            shift_6 = extract_features(augment_shift(audio, sample_rate, 6, 'both'), sample_rate)

            pitch_1 = extract_features(augment_pitch(audio, sample_rate, 1), sample_rate)
            pitch_2 = extract_features(augment_pitch(audio, sample_rate, 2), sample_rate)
            pitch_3 = extract_features(augment_pitch(audio, sample_rate, 3), sample_rate)
            pitch_4 = extract_features(augment_pitch(audio, sample_rate, 4), sample_rate)
            pitch_5 = extract_features(augment_pitch(audio, sample_rate, 5), sample_rate)
            pitch_6 = extract_features(augment_pitch(audio, sample_rate, 6), sample_rate)
            pitch_7 = extract_features(augment_pitch(audio, sample_rate, -1), sample_rate)
            pitch_8 = extract_features(augment_pitch(audio, sample_rate, -2), sample_rate)
            pitch_9 = extract_features(augment_pitch(audio, sample_rate, -3), sample_rate)
            pitch_10 = extract_features(augment_pitch(audio, sample_rate, -4), sample_rate)
            pitch_11 = extract_features(augment_pitch(audio, sample_rate, -5), sample_rate)
            pitch_12 = extract_features(augment_pitch(audio, sample_rate, -6), sample_rate)

            features = append_features(features, class_label, raw_data,
                                       shift_1, shift_2, shift_3, shift_4, shift_5, shift_6,
                                       pitch_1, pitch_2, pitch_3, pitch_4, pitch_5, pitch_6,
                                       pitch_7, pitch_8, pitch_9, pitch_10, pitch_11, pitch_12)

    featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

    print('Finished feature extraction from ', len(featuresdf), ' files')

    x = np.array(featuresdf.feature.tolist())
    y = np.array(featuresdf.class_label.tolist())

    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y))

    x_train, x_test, y_train, y_test = train_test_split(x, yy, test_size=0.2, random_state=42)
    num_rows = 40
    num_columns = 174
    num_channels = 1

    x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
    x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

    num_labels = yy.shape[1]
    filter_size = 2

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
    
    score = model.evaluate(x_test, y_test, verbose=1)
    
    accuracy = 100 * score[1]

    print("Pre-training accuracy: %.4f%%" % accuracy)

    num_epochs = 72
    num_batch_size = 256

    checkpointer = ModelCheckpoint(filepath='saved_models/weights.final.confusion.hdf5',
                                   verbose=1, save_best_only=True)
    start = datetime.now()

    history = model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test),
            callbacks=[checkpointer], verbose=1)

    hist_df = pd.DataFrame(history.history) 

    # save to json:  
    hist_json_file = 'history-confusion.json'
    with open(hist_json_file, mode='w+') as f:
        hist_df.to_json(f)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)

    score = model.evaluate(x_train, y_train, verbose=0)
    print("Training Accuracy: ", score[1])

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Testing Accuracy: ", score[1])

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




def extract_features(audio, sample_rate):
    max_pad_len = 174
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    pad_width = max_pad_len - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

    return mfccs


def load_audio(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        return audio, sample_rate
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)


def augment_shift(data, sampling_rate, shift_max, shift_direction):
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data


def augment_pitch(data, sampling_rate, pitch_factor):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)


def augment_speed(data, speed_factor):
    return librosa.effects.time_stretch(data, speed_factor)


def append_features(features, label, *augmented_data):
    for d in augmented_data:
        features.append([d, label])
    return features


def class_name(file):
    if file.startswith("ipf"):
        return "ipf"
    if file.startswith("healthy"):
        return "healthy"
    if file.startswith("copd"):
        return "copd"


if __name__ == '__main__':
    main()
