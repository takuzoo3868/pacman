import nltk
import re
from urllib.parse import unquote
import csv, pickle, random, json
import numpy as np
import tensorflow as tf

vec_dir = "sawada/data/word2vec.pickle"
pre_datas_train = "sawada/data/pre_datas_train.csv"
pre_datas_test = "sawada/data/pre_datas_test.csv"
process_datas_dir = "sawada/data/process_datas.pickle"


def gene_seg(payload_data):
    payload_data = payload_data.lower()
    payload_data = unquote(unquote(payload_data))
    payload_data, num = re.subn(r'\d+', "0", payload_data)
    payload_data, num = re.subn(r'(http|https)://[a-zA-Z0-9\.@&/#!#\?]+', "http://u", payload_data)
    r = '''
        (?x)[\w\.]+?\(
        |\)
        |"\w+?"
        |'\w+?'
        |http://\w
        |</\w+>
        |<\w+>
        |<\w+
        |\w+=
        |>
        |[\w\.]+
    '''
    return nltk.regexp_tokenize(payload_data, r)


def data_loader(f_name, l_name):
    with open(f_name, mode='r', encoding='utf-8') as f:
        data = list(set(f.readlines()))
        label = [l_name for i in range(len(data))]

        return data, label


def data_generator(data_dir):
    reader = tf.TextLineReader()
    queue = tf.train.string_input_producer([data_dir])
    _, value = reader.read(queue)
    coord = tf.train.Coordinator()
    sess = tf.Session()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    while True:
        v = sess.run(value)
        [data, label] = v.split(b"|")
        data = np.array(json.loads(data.decode("utf-8")))
        label = np.array(json.loads(label.decode("utf-8")))
        yield (data, label)
    coord.request_stop()
    coord.join(threads)
    sess.close()


def batch_generator(datas_dir, datas_size, batch_size, embeddings, reverse_dictionary, train=True):
    batch_data = []
    batch_label = []
    generator = data_generator(datas_dir)
    n = 0
    while True:
        for i in range(batch_size):
            data, label = next(generator)
            data_embed = []
            for d in data:
                if d != -1:
                    data_embed.append(embeddings[reverse_dictionary[d]])
                else:
                    data_embed.append([0.0] * len(embeddings[0]))
            batch_data.append(data_embed)
            batch_label.append(label)
            n += 1
            if not train and n == datas_size:
                break
        if not train and n == datas_size:
            yield (np.array(batch_data), np.array(batch_label))
            break
        else:
            yield (np.array(batch_data), np.array(batch_label))
            batch_data = []
            batch_label = []


def build_dataset(batch_size):
    with open(vec_dir, "rb") as f:
        word2vec = pickle.load(f)
    embeddings = word2vec["embeddings"]
    reverse_dictionary = word2vec["reverse_dictionary"]
    train_size = word2vec["train_size"]
    test_size = word2vec["test_size"]
    dims_num = word2vec["dims_num"]
    input_num = word2vec["input_num"]
    train_generator = batch_generator(pre_datas_train, train_size, batch_size, embeddings, reverse_dictionary)
    test_generator = batch_generator(pre_datas_test, test_size, batch_size, embeddings, reverse_dictionary, train=False)
    return train_generator, test_generator, train_size, test_size, input_num, dims_num
