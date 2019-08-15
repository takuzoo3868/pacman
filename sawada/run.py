import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# dataset files
XSS_TRAIN_FILE = 'dataset/train_level_1.csv'
XSS_TEST_FILE = 'dataset/test_level_1.csv'
NORMAL_TRAIN_FILE = 'dataset/normal.csv'
NORMAL_TEST_FILE = 'dataset/normal.csv'

XSS_TRAIN_FILE_2 = 'dataset/train_level_2.csv'
XSS_TEST_FILE_2 = 'dataset/test_level_2.csv'

NORMAL_TRAIN_FILE_4 = 'dataset/train_level_4.csv'
NORMAL_TEST_FILE_4 = 'dataset/test_level_4.csv'

STOP_WORDS = [';', '\"', '\'']

keys = []
test_src = ""
test_result = ""


def clean(text):
    target = ["</", "/>", "<", ">", "=", ":", "/", "(", ")", "[", "]", "{", "}", "＜", "＞"]
    for ch in target:
        text = text.replace(ch, " ")
    return text


def data_loader(src, label):
    data = []
    with open(src) as f:
        for line in f:
            data += clean(line).split()
    return data, [label for _ in range(len(data))]


def run():
    """
    データ作成
    """
    xss_train_data, xss_train_label = data_loader(XSS_TRAIN_FILE, 'xss')
    xss_test_data, xss_test_label = data_loader(XSS_TEST_FILE, 'xss')
    normal_train_data, normal_train_label = data_loader(NORMAL_TRAIN_FILE_4, 'normal')
    normal_test_data, normal_test_label = data_loader(NORMAL_TEST_FILE_4, 'normal')

    X_train = xss_train_data + normal_train_data
    y_train = xss_train_label + normal_train_label

    X_test = xss_test_data + normal_test_data
    y_test = xss_test_label + normal_test_label

    """
    データ前処理・学習機作成
    """
    vec = TfidfVectorizer(preprocessor=clean, stop_words=STOP_WORDS)
    X_train = vec.fit_transform(X_train)
    X_train = X_train.todense()

    clf = MultinomialNB(alpha=0.6)
    clf.fit(X_train, y_train)

    """
    テスト
    """
    X_test = vec.transform(X_test)
    X_test = X_test.todense()
    pred = clf.predict(X_test)
    acc_score = accuracy_score(y_test, pred)
    conf_mat = confusion_matrix(
        pred, y_test, labels=['xss', 'normal']
    )

    # count = 0
    # print("[!] fail data")
    # for p, a in zip(pred, y_test):
    #     if p != a:
    #         print("-----------")
    #         print("ans: ", a, " pred:", p)
    #         print((normal_test_data + xss_test_data)[count])
    #     count += 1

    print("acc: \n", acc_score)
    print("confusion matrix: \n", conf_mat)


if __name__ == '__main__':
    run()
