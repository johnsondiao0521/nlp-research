import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader


def read_data(file, num=None):
    with open(file, encoding='utf8') as f:
        all_data = f.read().split('\n')
    if num:
        return all_data[:num]
    return all_data


def get_word_2_index(all_text):
    word_2_index = {"PAD": 0, "UNK": 1}
    for text in all_text:
        for w in text:
            word_2_index[w] = word_2_index.get(w, len(word_2_index))
    index_2_word = list(word_2_index)
    return word_2_index, index_2_word


class PDataset(Dataset):
    def __init__(self, all_data):
        self.all_data = all_data

    def __getitem__(self, index):
        text = self.all_data[index]
        input_text = text[:-1]
        label_text = text[1:]

        input_idx = [word_2_index.get(i, 1) for i in input_text]
        label_idx = [word_2_index.get(i, 1) for i in label_text]
        return torch.tensor(input_idx), torch.tensor(label_idx)

    def __len__(self):
        return len(self.all_data)


class PModel(nn.Module):
    def __init__(self):
        super(PModel, self).__init__()
        self.emb = nn.Embedding(word_size, emb_dim)
        self.rnn = nn.RNN(emb_dim, rnn_hidden_num, batch_first=True, bidirectional=True)
        self.cls = nn.Linear(rnn_hidden_num, word_size)
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, x):
        pass


if __name__ == '__main__':
    train_data = read_data(os.path.join('..', 'data', '古诗生成', 'poetry_5.txt'), 300)
    word_2_index, index_2_word = get_word_2_index(train_data)

    epoch = 10
    batch_size = 10
    word_size = len(word_2_index)
    emb_dim = 200
    rnn_hidden_num = 100

    train_dataset = PDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False)

    for e in range(epoch):
        for text_idx, label_idx in train_dataloader:
            pass
    print("")