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


if __name__ == '__main__':
    all_text, all_label = read_data(os.path.join('data', 'train1.txt'))

    epoch = 10
    batch_size = 9

    batch_num = math.ceil(len(all_text) / batch_size)

    for e in range(epoch):
        print("*" * 30 + f" epoch: {e} " + "*" * 30)
        random_idx = [i for i in range(len(all_text))]
        random.shuffle(random_idx)
        for batch_idx in range(batch_num):

            batch_random_i = random_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            batch_text = [all_text[i] for i in batch_random_i]
            batch_label = [all_label[i] for i in batch_random_i]

            print(batch_text, batch_label)

    print("")