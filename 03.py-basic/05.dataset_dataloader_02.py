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
        return Dataloader(self)


class Dataloader():
    def __init__(self, dataset):

        self.dataset = dataset
        self.cursor = 0

        self.random_idx = [i for i in range(len(self.dataset.all_label))]
        random.shuffle(self.random_idx)

    def __next__(self):
        if self.cursor > len(self.dataset.all_label):
            return None

        batch_i = self.random_idx[self.cursor: self.cursor + self.dataset.batch_size]
        batch_text = [self.dataset.all_text[i] for i in batch_i]
        batch_label = [self.dataset.all_label[i] for i in batch_i]

        self.cursor += self.dataset.batch_size
        return batch_text, batch_label


class Model():
    def __init__(self):
        pass

    def forward(self, x):
        return "hello"


if __name__ == '__main__':
    all_text, all_label = read_data(os.path.join('data', 'train1.txt'))

    epoch = 10
    batch_size = 2

    batch_num = math.ceil(len(all_text) / batch_size)
    train_dataset = Dataset(all_text, all_label, batch_size)

    for e in range(epoch):
        print("*" * 30 + f" {e} " + "*" * 30)
        for i in train_dataset:
            if i is not None:
                print(i)
            else:
                break
    print("")