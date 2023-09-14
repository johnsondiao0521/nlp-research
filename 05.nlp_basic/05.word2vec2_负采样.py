import os
import pandas as pd
import jieba
import numpy as np
from tqdm import tqdm
import pickle
import random


def read_data(file):
    all_data = pd.read_csv(file, encoding='gbk', names=['data'])
    all_data = all_data['data'].tolist()

    cut_data = []
    for data in all_data:
        cut_word = jieba.lcut(data)
        cut_data.append(cut_word)
    return cut_data


def buidl_word_2_index(all_data):
    word_2_index = {}
    for data in all_data:
        for w in data:
            word_2_index[w] = word_2_index.get(w, len(word_2_index))

    return word_2_index


def build_word_2_onehot(len_):
    return np.eye(len_)


def softmax(x):
    max_x = np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x - max_x)
    sum_ex = np.sum(ex, axis=1, keepdims=True)
    result = ex / sum_ex
    return result


def get_triple(now_word_idx, words):
    now_word = words[now_word_idx]
    r_word = words[now_word_idx-n_gram: now_word_idx] + words[now_word_idx+1: now_word_idx+1+n_gram]
    result = [(words[now_word_idx], i, np.array([[1]])) for i in r_word]

    for i in range(negative):
        other_word = random.choice(index_2_word)
        if other_word in r_word or other_word == now_word:
            continue
        result.append((now_word, other_word, np.array([[0]])))
    return result


def sigmoid(x):
    x = np.clip(x, -30, 30)
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    all_data = read_data(os.path.join('..', 'data', 'word2vec', '数学原始数据.csv'))[:500]
    word_2_index = buidl_word_2_index(all_data)
    index_2_word = list(word_2_index)
    words_len = len(word_2_index)
    word_2_onehot = build_word_2_onehot(words_len)

    epoch = 10
    n_gram = 2  # 上下可以看到的单词
    embedding_num = 300
    lr = 0.1
    negative = 5

    w1 = np.random.normal(size=(words_len, embedding_num))
    w2 = np.random.normal(size=(embedding_num, words_len))

    for e in range(epoch):
        for words in tqdm(all_data):
            for now_word_idx, now_word in enumerate(words):
                triple = get_triple(now_word_idx, words)

                for now_word, other_word, label in triple:
                    now_word_onehot = word_2_onehot[word_2_index[now_word]]
                    other_word_onehot = word_2_onehot[word_2_index[other_word]]

                    hidden = w1[word_2_index[now_word]: word_2_index[now_word] + 1]
                    pre = hidden @ w2[:, word_2_index[other_word], None]
                    p = sigmoid(pre)

                    loss = -np.sum(label * np.log(p) + (1 - label) * np.log(1 - p))

                    G = p - label
                    delta_w2 = hidden.T @ G
                    delta_H = G @ w2[:, word_2_index[other_word], None].T
                    delta_w1 = 1 * delta_H

                    w1[word_2_index[now_word], None] -= lr * delta_w1
                    w2[:, word_2_index[other_word], None] -= lr * delta_w2

        with open("word2vec.abc", "wb") as f:
            pickle.dump(w1, f)

        with open("word2index.abc", "wb") as f:
            pickle.dump(word_2_index, f)

        print(loss)