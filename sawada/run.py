import csv, pickle, time
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models.word2vec import Word2Vec

from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout, LSTM
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.metrics import precision_score, recall_score

from sawada.utils import data_loader, build_dataset
from sawada.pre_data import pre_process
from sawada.utils import gene_seg

BATCH_SIZE = 500
EPOCHS_NUM = 1
MODEL_DIR = "sawada/models"
log_dir = "word2vec.log"
plt_dir = "sawada/data/word2vec.png"
vec_dir = "sawada/data/word2vec.pickle"

XSS_TRAIN_FILE = "dataset/train_level_2.csv"
XSS_TEST_FILE = "dataset/test_level_2.csv"
NORMAL_TRAIN_FILE = "dataset/normal.csv"
NORMAL_TEST_FILE = "dataset/normal.csv"
STOP_WORDS = ['']

learning_rate = 0.1
vocabulary_size = 3000
batch_size = 128
embedding_size = 128
num_skips = 4
skip_window = 5
num_sampled = 64
num_iter = 5
plot_only = 100


def train(train_generator, train_size, input_num, dims_num):
    print("Start Train Job! ")
    start = time.time()
    inputs = InputLayer(input_shape=(input_num, dims_num), batch_size=BATCH_SIZE)
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
    # model.fit_generator(train_generator, steps_per_epoch=train_size // BATCH_SIZE, epochs=EPOCHS_NUM, callbacks=[call])
    model.fit_generator(train_generator, steps_per_epoch=5, epochs=5, callbacks=[call])
    model.save(MODEL_DIR)
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
    前処理 データベクトル化
    """
    start = time.time()
    words = []
    datas = []
    with open("sawada/data/xssed.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, fieldnames=["payload"])
        for row in reader:
            payload = row["payload"]
            word = gene_seg(payload)
            datas.append(word)
            words += word

    def build_dataset(datas, words):
        count = [["UNK", -1]]
        counter = Counter(words)
        count.extend(counter.most_common(vocabulary_size - 1))
        vocabulary = [c[0] for c in count]
        data_set = []
        for data in datas:
            d_set = []
            for word in data:
                if word in vocabulary:
                    d_set.append(word)
                else:
                    d_set.append("UNK")
                    count[0][1] += 1
            data_set.append(d_set)
        return data_set

    data_set = build_dataset(datas, words)

    model = Word2Vec(data_set, size=embedding_size, window=skip_window, negative=num_sampled, iter=num_iter)
    embeddings = model.wv

    def plot_with_labels(low_dim_embs, labels, filename=plt_dir):
        plt.figure(figsize=(10, 10))
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label, xy=(x, y), xytext=(5, 2),
                         textcoords="offset points",
                         ha="right",
                         va="bottom")
            f_text = "vocabulary_size=%d;batch_size=%d;embedding_size=%d;skip_window=%d;num_iter=%d" % (
                vocabulary_size, batch_size, embedding_size, skip_window, num_iter
            )
            plt.figtext(0.03, 0.03, f_text, color="green", fontsize=10)
        plt.show()
        plt.savefig(filename)

    tsne = TSNE(perplexity=30, n_components=2, init="pca", n_iter=5000)
    plot_words = embeddings.index2word[:plot_only]
    plot_embeddings = []
    for word in plot_words:
        plot_embeddings.append(embeddings[word])
    low_dim_embs = tsne.fit_transform(plot_embeddings)
    plot_with_labels(low_dim_embs, plot_words)

    def save(embeddings):
        dictionary = dict([(embeddings.index2word[i], i) for i in range(len(embeddings.index2word))])
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        word2vec = {"dictionary": dictionary, "embeddings": embeddings, "reverse_dictionary": reverse_dictionary}
        with open(vec_dir, "wb") as f:
            pickle.dump(word2vec, f)

    save(embeddings)
    end = time.time()
    print("Over job in ", end - start)
    print("Saved words vec to", vec_dir)

    """
    データ整形
    """
    pre_process()

    """
    学習器生成・テスト
    """
    train_generator, test_generator, train_size, test_size, input_num, dims_num = build_dataset(BATCH_SIZE)
    train(train_generator, train_size, input_num, dims_num)
    test(MODEL_DIR, test_generator, test_size, input_num, dims_num, BATCH_SIZE)


if __name__ == '__main__':
    run()
