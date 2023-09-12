import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch


def read_data(file, num=None):
    with open(file, encoding='utf-8') as f:
        all_data = f.read().split('\n')

    all_text = []
    all_label = []
    for data in all_data:
        s_data = data.split('\t')
        if len(s_data) != 2:
            continue
        text, label = s_data
        all_text.append(text)
        all_label.append(int(label))
    if num and num > 0:
        return all_text[:num], all_label[:num]
    elif num and num < 0:
        return all_text[num:], all_label[num:]
    else:
        return all_text, all_label


class MyDataset(Dataset):
    def __init__(self, all_text, all_label):
        self.all_text = all_text
        self.all_label = all_label

    def __getitem__(self, index):
        text = self.all_text[index][:max_len]
        label = self.all_label[index]

        text_idx = [word_2_index.get(w, 1) for w in text]
        text_idx = text_idx + [0] * (max_len - len(text_idx))
        text_emb = [word_2_onehot[i] for i in text_idx]
        text_emb = np.array(text_emb, dtype=np.float32)

        return text_emb, label

    def __len__(self):
        assert len(self.all_label) == len(self.all_text)
        return len(self.all_text)


def get_word_2_index(all_text):
    word_2_index = {'PAD': 0, 'UNK': 1}
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


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.w = nn.Linear(len(word_2_index), class_num)
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, x, label=None):
        pre = self.w(x)
        pre = torch.mean(pre, dim=1)

        if label is not None:
            loss = self.loss_fun(pre, label)
            return loss
        return torch.argmax(pre, dim=-1)


if __name__ == '__main__':
    class_num = 10
    train_text, train_label = read_data(os.path.join('..', 'data', '文本分类', 'train.txt'), 2000)
    test_text, test_label = read_data(os.path.join('..', 'data', '文本分类', 'train.txt'), -300)

    word_2_index, index_2_word = get_word_2_index(train_text)
    word_2_onehot = get_word_2_onehot(len(word_2_index))

    max_len = 30
    batch_size = 20
    epoch = 100
    lr = 0.001
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = MyDataset(train_text, train_label)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

    test_dataset = MyDataset(test_text, test_label)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

    model = MyModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(epoch):
        for batch_text_emb, batch_label in train_dataloader:
            batch_text_emb = batch_text_emb.to(device)
            batch_label = batch_label.to(device)
            loss = model.forward(batch_text_emb, batch_label)
            loss.backward()

            opt.step()
            opt.zero_grad()

        print(f"loss: {loss}")

        right_num = 0
        for batch_text_emb, batch_label in test_dataloader:
            batch_text_emb = batch_text_emb.to(device)
            batch_label = batch_label.to(device)
            pre = model.forward(batch_text_emb)
            right_num += int(torch.sum(pre == batch_label))
        acc = right_num / len(test_dataset)
        print(f"acc: {acc:.3f}")