import os
import math
import random


def read_data(file):
    with open(file, encoding='utf8') as f:
        all_data = f.read().split('\n')

    all_text, all_label = [], []
    for data in all_data:
        data_s = data.split('\t')
        if len(data_s) != 2:
            continue
        text, label = data_s
        try:
            label = int(label)
            all_text.append(text)
            all_label.append(label)
        except:
            print("标签报错了！！！")

    assert len(all_text) == len(all_label), "数据和标签长度不一样，玩个毛啊~~"
    return all_text, all_label


class Dataset():
    def __init__(self, all_text, all_label, batch_size):
        self.all_text = all_text
        self.all_label = all_label
        self.batch_size = batch_size

    def __iter__(self):
        dataloader = DataLoader(self)
        return dataloader


class DataLoader():
    def __init__(self, dataset):
        self.dataset = dataset
        self.cursor = 0

    def __next__(self):
        if self.cursor >= len(self.dataset.all_text):
            raise StopIteration
        text = self.dataset.all_text[self.cursor: self.cursor + self.dataset.batch_size]
        label = self.dataset.all_label[self.cursor: self.cursor + self.dataset.batch_size]
        self.cursor += len(text)
        return text, label


def get_word_2_index(all_text):
    word_2_index = {}
    for text in all_text:
        for w in text:
            word_2_index[w] = word_2_index.get(w, len(word_2_index))

    return word_2_index


if __name__ == '__main__':
    all_text, all_label = read_data(os.path.join('data', 'train1.txt'))

    word_2_index = get_word_2_index(all_text)

    epoch = 2
    batch_size = 2
    dataset = Dataset(all_text, all_label, batch_size)

    for e in range(epoch):
        for data in dataset:
            print(data)

    print("")

