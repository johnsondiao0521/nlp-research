import os
import pandas as pd
import jieba
import numpy as np
from tqdm import tqdm
import pickle


def read_data(file):
    all_data = pd.read_csv(file, encoding='gbk', names=['data'])
    all_data = all_data['data'].tolist()

    cut_data = []
    for data in all_data:
        cut_word = jieba.lcut(data)
        cut_data.append(cut_word)
    return cut_data


def build_word_2_index(all_data):
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


if __name__ == '__main__':
    all_data = read_data(os.path.join('..', 'data', 'word2vec', '数学原始数据.csv'))[:500]
    word_2_index = build_word_2_index(all_data)
    words_len = len(word_2_index)
    word_2_onehot = build_word_2_onehot(words_len)

    epoch = 10
    n = 2  # 上下可以看到的单词
    embedding_num = 300
    lr = 0.1

    w1 = np.random.normal(size=(words_len, embedding_num))
    w2 = np.random.normal(size=(embedding_num, words_len))

    for e in range(epoch):
        for words in tqdm(all_data):
            for ni, now_word in enumerate(words):

                now_word_onehot = word_2_onehot[word_2_index[now_word]]
                other_words = words[ni - n: ni] + words[ni+1: ni+1+n]

                other_words_idx = [word_2_index[i] for i in other_words]
                other_words_onehot = word_2_onehot[other_words_idx]

                hidden = other_words_onehot @ w1
                pre = hidden @ w2

                p = softmax(pre)

                loss = -np.sum(now_word_onehot * np.log(p))

                delta_pre = G = p - now_word_onehot

                delta_w2 = hidden.T @ G
                delta_hidden = G @ w2.T

                delta_w1 = other_words_onehot.T @ delta_hidden

                w1 -= lr * delta_w1
                w2 -= lr * delta_w2

        with open("word2vec.abc", "wb") as f:
            pickle.dump(w1, f)

        with open("word2index.abc", "wb") as f:
            pickle.dump(word_2_index, f)

        print(loss)