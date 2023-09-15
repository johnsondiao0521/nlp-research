import os
import pandas as pd
import jieba
import numpy as np
from tqdm import tqdm
import pickle
import random


def get_data(file):
    all_data = pd.read_csv(file, encoding="gbk", names=["data"])
    all_data = all_data["data"].tolist()

    cut_data = []
    for data in all_data:
        word_cut = jieba.lcut(data)
        cut_data.append(word_cut)

    return cut_data[:]


def build_word_2_index(all_data):
    word_2_index = {}
    for data in all_data:
        for w in data:
            word_2_index[w] = word_2_index.get(w, len(word_2_index))
    return word_2_index


def build_word_2_onehot(len_):
    return np.eye(len_).reshape(len_, 1, len_)


def softmax(x):
    max_x = np.max(x, axis=-1)
    ex = np.exp(x - max_x)

    sum_ex = np.sum(ex, axis=1)

    result = ex / sum_ex

    return result


def get_triple(now_word_idx, words):
    now_word = words[now_word_idx]
    r_word = words[now_word_idx - n_gram:now_word_idx] + words[now_word_idx + 1:now_word_idx + 1 + n_gram]

    result = [(now_word, i, np.array([[1]])) for i in r_word]

    for i in range(negtive):
        other_word = random.choice(wf)

        if other_word in r_word or other_word == now_word:
            continue
        result.append((now_word, other_word, np.array([[0]])))

    return result


def get_word_frequence(all_data, word_2_index):
    word_2_f = {}
    wf = []
    stop_word = ["。", "，", "的", "用"]
    for words in all_data:
        for w in words:
            if w in stop_word:
                continue
            word_2_f[w] = word_2_f.get(w, 0)
            word_2_f[w] += 1

    for w, f in word_2_f.items():
        wf += [w for i in range(f)]

    return wf


def get_triple2(now_word_idx, words):
    now_word = words[now_word_idx]
    r_word = words[now_word_idx - n_gram:now_word_idx] + words[now_word_idx + 1:now_word_idx + 1 + n_gram]

    result = [[word_2_index[now_word], word_2_index[i], 1] for i in r_word]

    for i in range(negtive):
        other_word = random.choice(index_2_word)

        if other_word in r_word or other_word == now_word:
            continue
        result.append([word_2_index[now_word], word_2_index[other_word], 0])

    m = np.array(result)

    return m[:, 0], m[:, 1], m[:, 2:].T


def sigmoid(x):
    x = np.clip(x, -30, 30)
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    all_data = get_data(os.path.join("..", "data", "word2vec", "数学原始数据.csv"))
    word_2_index = build_word_2_index(all_data)
    wf = get_word_frequence(all_data, word_2_index)

    words_len = len(word_2_index)
    index_2_word = list(word_2_index)
    word_2_onehot = build_word_2_onehot(words_len)

    epoch = 10
    n_gram = 2
    embedding_num = 300
    negtive = 10
    lr = 0.07

    w1 = np.random.normal(size=(words_len, embedding_num))
    w2 = np.random.normal(size=(embedding_num, words_len))

    for e in range(epoch):
        for words in tqdm(all_data):
            for now_word_idx, now_word in enumerate(words):
                _, other_word_idx, label = get_triple2(now_word_idx, words)

                hidden = 1 * w1[now_word_idx, None]

                pre = hidden @ w2[:, other_word_idx]

                p = sigmoid(pre)

                G = p - label

                delta_w2 = hidden.T @ G
                delta_h = G @ w2[:, other_word_idx].T

                delta_w1 = delta_h

                w1[word_2_index[now_word], None] -= lr * delta_w1
                w2[:, other_word_idx] -= lr * delta_w2

    with open("wordvec.abc", "wb") as f:
        pickle.dump(w1, f)

    with open("word2index.123", "wb") as f:
        pickle.dump(word_2_index, f)
    print("训练结束")
