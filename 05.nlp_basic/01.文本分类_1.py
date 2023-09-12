import os
from torch.utils.data import Dataset, DataLoader
import numpy as np


def read_data(file):
    with open(file, encoding='utf-8') as f:
        all_data = f.read().split('\n')

    all_text = []
    all_label = []
    for data in all_data:
        s_data = data.split(' ')
        if len(s_data) != 2:
            continue
        text, label = s_data
        all_text.append(text)
        all_label.append(int(label))
    return all_text, all_label


class MyDataset(Dataset):
    def __init__(self, all_text, all_label):
        self.all_text = all_text
        self.all_label = all_label

    def __getitem__(self, index):
        text = self.all_text[index][:max_len]
        label = self.all_label[index]

        text_idx = [word_2_index.get(w, 0) for w in text]
        text_idx = text_idx + [0] * (max_len - len(text_idx))
        text_emb = [word_2_onehot[i] for i in text_idx]
        text_emb = np.array(text_emb)

        return text_emb, label

    def __len__(self):
        assert len(self.all_label) == len(self.all_text)
        return len(self.all_text)


def get_word_2_index(all_text):
    word_2_index = {'PAD': 0}
    for text in all_text:
        for w in text:
            word_2_index[w] = word_2_index.get(w, len(word_2_index))

    index_2_word = list(word_2_index)
    return word_2_index, index_2_word


def get_word_2_onehot(len_):
    onehot = np.zeros((len_, len_))
    for i in range(len(onehot)):
        onehot[i][i] = 1
    return onehot


def softmax(x):
    # x = np.clip(x, -1e10, 100)
    max_x = np.max(x, axis=-1, keepdims=True)
    x = x - max_x
    ex = np.exp(x)
    sum_ex = np.sum(ex, axis=1, keepdims=True)
    result = ex / sum_ex
    result = np.clip(result, 1e-10, 1e10)
    return result


def make_onehot(labels, class_num):
    result = np.zeros((len(labels), class_num))

    for idx, cls in enumerate(labels):
        result[idx][cls] = 1
    return result


if __name__ == '__main__':
    train_text, train_label = read_data(os.path.join('..', 'data', '构造分类数据集数据.txt'))
    train_label = make_onehot(train_label, 2)

    word_2_index, index_2_word = get_word_2_index(train_text)
    word_2_onehot = get_word_2_onehot(len(word_2_index))

    max_len = 8
    batch_size = 1
    epoch = 10
    lr = 0.1

    w1 = np.random.normal(0, 1, size=(len(word_2_index), 2))

    train_dataset = MyDataset(train_text, train_label)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False)

    for e in range(epoch):
        for batch_text_emb, batch_label in train_dataloader:
            batch_text_emb = batch_text_emb.numpy()
            batch_label = batch_label.numpy()
            pre = batch_text_emb @ w1
            pre_mean = np.mean(pre, axis=1)
            p = softmax(pre_mean)

            loss = -np.sum(batch_label * np.log(p) + (1 - batch_label) * np.log(1 - p))

            G = (p - batch_label) / len(batch_label)

            dpre = np.zeros_like(pre)
            for i in range(len(G)):
                for j in range(G.shape[1]):
                    dpre[i][:, j] = G[i][j]

            delta_w1 = batch_text_emb.transpose(0, 2, 1) @ dpre

            delta_w1 = np.mean(delta_w1, axis=0)
            w1 -= lr * delta_w1
            print(loss)