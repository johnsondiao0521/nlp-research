# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:johnsondiao
@File: bert_ner.py
@Time: 2022-12-02 17:09
@Desc:
"""
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from transformers import BertModel
from transformers import AdamW

from seqeval.metrics import accuracy_score as seq_accuracy_score
from seqeval.metrics import precision_score as seq_precision_score
from seqeval.metrics import recall_score as seq_recall_score
from seqeval.metrics import f1_score as seq_f1_score


def read_data(file):
    with open(file, "r", encoding="utf-8") as f:
        all_data = f.read().split('\n')

    all_text = []
    all_label = []

    text = []
    label = []
    for data in all_data:
        if data == '':
            all_text.append(text)
            all_label.append(label)
            text = []
            label = []
        else:
            t, l = data.split(' ')
            text.append(t)
            label.append(l)
    return all_text, all_label


def build_label(train_label):
    label_2_index = {"PAD": 0, "UNK": 1}
    for label in train_label:
        for l in label:
            if l not in label_2_index:
                label_2_index[l] = len(label_2_index)
    return label_2_index, list(label_2_index)


class BertDataset(Dataset):
    def __init__(self, all_text, all_label, label_2_index, max_len, tokenizer):
        self.all_text = all_text
        self.all_label = all_label
        self.label_2_index = label_2_index
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        text = self.all_text[index]
        label = self.all_label[index][:self.max_len]

        text_index = self.tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=self.max_len+2, padding="max_length", return_tensors='pt')
        label_index = [0] + [self.label_2_index.get(l, 1) for l in label] + [0] + (self.max_len - len(text)) * [0]

        label_index = torch.tensor(label_index)
        return text_index.reshape(-1), label_index, len(label)

    def __len__(self):
        return self.all_text.__len__()


class BertNERModel(nn.Module):
    def __init__(self, class_num):
        super(BertNERModel, self).__init__()
        self.bert = BertModel.from_pretrained(os.path.join('..', 'bert-base-chinese'))
        # for name, param in self.bert.named_parameters(): # 不更新模型的参数
        #     param.requires_grad = False
        self.classifier = nn.Linear(768, class_num)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch_text, batch_label=None):
        bert_out = self.bert(batch_text)
        bert_out0, bert_out1 = bert_out[0], bert_out[1]  # bert_out0:字符级别特征，bert_out1:篇章级别

        pre = self.classifier(bert_out0)
        if batch_label is not None:
            loss = self.loss_fn(pre.reshape(-1, pre.shape[-1]), batch_label.reshape(-1))
            return loss
        else:
            return torch.argmax(pre, dim=-1)


if __name__ == '__main__':
    train_text, train_label = read_data(os.path.join('data', 'train.txt'))
    dev_text, dev_label = read_data(os.path.join('data', 'dev.txt'))
    test_text, test_label = read_data(os.path.join('data', 'test.txt'))
    label_2_index, index_2_label = build_label(test_label)

    tokenizer = BertTokenizer.from_pretrained(os.path.join('..', 'bert-base-chinese'))

    batch_size = 64
    epoch = 100
    max_len = 30
    lr = 0.0001

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_dataset = BertDataset(train_text, train_label, label_2_index, max_len, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    dev_dataset = BertDataset(dev_text, dev_label, label_2_index, max_len, tokenizer)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    model = BertNERModel(len(label_2_index)).to(device)
    opt = AdamW(model.parameters(), lr)

    for e in range(epoch):

        model.train()
        for batch_text_index, batch_label_index, batch_len in train_dataloader:
            batch_text_index = batch_label_index.to(device)
            batch_label_index = batch_label_index.to(device)
            loss = model.forward(batch_text_index, batch_label_index)
            loss.backward()

            opt.step()
            opt.zero_grad()

            print(f'epoch: {e}, loss: {loss:.4f}')

        model.eval()

        all_pre = []
        all_tag = []
        for batch_text_index, batch_label_index, batch_len in dev_dataloader:
            if len(batch_text_index) == 0:
                print(batch_text_index)
            batch_text_index = batch_label_index.to(device)
            batch_label_index = batch_label_index.to(device)
            pre = model.forward(batch_text_index)

            pre = pre.cpu().numpy().tolist()
            tag = batch_label_index.cpu().numpy().tolist()

            for p, t, l in zip(pre, tag, batch_len):
                p = p[1: 1 + l]
                t = t[1: 1 + l]

                p = [index_2_label[i] for i in p]
                t = [index_2_label[i] for i in t]

                all_pre.append(p)
                all_tag.append(t)
            # pre = [[index_2_label[j] for j in i] for i in pre]
            # batch_label_index = [[index_2_label[j] for j in i] for i in batch_label_index]

        f1_score = seq_f1_score(all_tag, all_pre)

        print(f"f1: {f1_score}")
