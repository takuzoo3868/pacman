import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout, LSTM
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.metrics import precision_score, recall_score

from sample.utils import data_loader, build_dataset

batch_size = 500
epochs_num = 1
model_dir = "sawada/models"

XSS_TRAIN_FILE = 'dataset/train_level_1.csv'
XSS_TEST_FILE = 'dataset/test_level_1.csv'
NORMAL_TRAIN_FILE = 'dataset/normal.csv'
NORMAL_TEST_FILE = 'dataset/normal.csv'
STOP_WORDS = ['']


def train(train_generator, train_size, input_num, dims_num):
    print("Start Train Job! ")
    start = time.time()
    inputs = InputLayer(input_shape=(input_num, dims_num), batch_size=batch_size)
    layer1 = LSTM(128)
    output = Dense(2, activation="softmax", name="Output")
    optimizer = Adam()
    model = Sequential()
    model.add(inputs)
    model.add(layer1)
    model.add(Dropout(0.5))
    model.add(output)
    call = TensorBoard(write_grads=True, histogram_freq=1)
    model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit_generator(train_generator, steps_per_epoch=train_size // batch_size, epochs=epochs_num, callbacks=[call])
    #    model.fit_generator(train_generator, steps_per_epoch=5, epochs=5, callbacks=[call])
    model.save(model_dir)
    end = time.time()
    print("Over train job in %f s" % (end - start))


def test(model_dir, test_generator, test_size, input_num, dims_num, batch_size):
    model = load_model(model_dir)
    labels_pre = []
    labels_true = []
    batch_num = test_size // batch_size + 1
    steps = 0
    for batch, labels in test_generator:
        if len(labels) == batch_size:
            labels_pre.extend(model.predict_on_batch(batch))
        else:
            batch = np.concatenate((batch, np.zeros((batch_size - len(labels), input_num, dims_num))))
            labels_pre.extend(model.predict_on_batch(batch)[0:len(labels)])
        labels_true.extend(labels)
        steps += 1
        print("%d/%d batch" % (steps, batch_num))
    labels_pre = np.array(labels_pre).round()

    def to_y(labels_data):
        y = []
        for i in range(len(labels_data)):
            if labels_data[i][0] == 1:
                y.append(0)
            else:
                y.append(1)
        return y

    y_true = to_y(labels_true)
    y_pre = to_y(labels_pre)
    precision = precision_score(y_true, y_pre)
    recall = recall_score(y_true, y_pre)
    print("Precision score is :", precision)
    print("Recall score is :", recall)


def run():
    """
    データ作成
    """
    xss_train_data, xss_train_label = data_loader(XSS_TRAIN_FILE, 'xss')
    xss_test_data, xss_test_label = data_loader(XSS_TEST_FILE, 'xss')
    normal_train_data, normal_train_label = data_loader(NORMAL_TRAIN_FILE, 'normal')
    normal_test_data, normal_test_label = data_loader(NORMAL_TEST_FILE, 'normal')

    X_train = xss_train_data + normal_train_data
    y_train = xss_train_label + normal_train_label
    X_test = xss_test_data + normal_test_data
    y_test = xss_test_label + normal_test_label

    """
    前処理
    """


    """
    実行
    """
    train_generator, test_generator, train_size, test_size, input_num, dims_num = build_dataset(batch_size)
    train(train_generator, train_size, input_num, dims_num)
    test(model_dir, test_generator, test_size, input_num, dims_num, batch_size)


if __name__ == '__main__':
    run()
