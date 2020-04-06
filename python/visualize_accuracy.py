import json
import matplotlib.pyplot as plt


def load_hist(path):
    with open(path, 'r', encoding='utf-8') as file:
        n = json.loads(file.read())
    return n

if __name__ == '__main__':
    hist_json_file = 'history.json'
    history = load_hist(hist_json_file)
    train_accuracy = history['accuracy']
    test_accuracy = history['val_accuracy']
    epoch = []
    train_accuracy_values = []
    test_accuracy_values = []

    for key, val in train_accuracy.items():
        epoch.append(key)
        train_accuracy_values.append(val)
    epoch = []
    for key, val in test_accuracy.items():
        epoch.append(key)
        test_accuracy_values.append(val)

    plt.plot(epoch, train_accuracy_values)
    plt.plot(epoch, test_accuracy_values)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()