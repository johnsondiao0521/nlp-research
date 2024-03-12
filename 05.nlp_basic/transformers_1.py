import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel


def read_data(file_path, num=None):
    with open(file_path, 'r', encoding='utf-8') as f:
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
    if num is not None:
        return all_text[:num], all_label[:num]
    return all_text, all_label


def get_word_2_index(all_text):
    word_2_index = {'[PAD]': 0, '[UNK]': 1}
    for text in all_text:
        for w in text:
            word_2_index[w] = word_2_index.get(w, len(word_2_index))
    return word_2_index, list(word_2_index)


class MyDataset(Dataset):
    def __init__(self, all_text, all_label):
        self.all_text = all_text
        self.all_label = all_label

    def __getitem__(self, index):
        text = self.all_text[index]
        text_idx = [word_2_index.get(w, 1) for w in text]
        label = self.all_label[index]

        return text_idx, label, len(text_idx)

    def __len__(self):
        assert len(self.all_label) == len(self.all_text)
        return len(self.all_label)


def coll_fn(batch_data):
    batch_idx, batch_label, batch_len = zip(*batch_data)
    max_len = max(batch_len)

    batch_idx_new = []
    batch_label_new = []
    for idx, label, len_ in zip(batch_idx, batch_label, batch_len):
        text_idx = idx + [0] * (max_len - len_)
        batch_idx_new.append(text_idx)
        batch_label_new.append(label)

    return torch.tensor(batch_idx_new), torch.tensor(batch_label_new), torch.tensor(batch_len)


class Positional(nn.Module):
    def __init__(self):
        super( Positional, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)


class Multi_Head_Attention(nn.Module):
    def __init__(self):
        super(Multi_Head_Attention, self).__init__()
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(self.linear(x))


class Norm(nn.Module):
    def __init__(self):
        super(Norm, self).__init__()
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        return self.norm(x)


class Feed_Forward(nn.Module):
    def __init__(self):
        super(Feed_Forward, self).__init__()
        self.linear1 = nn.Linear(embedding_dim, feed_num)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(feed_num, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        linear1_out = self.linear1(x)
        gelu_out = self.gelu(linear1_out)
        linear2_out = self.linear2(gelu_out)
        norm_out = self.norm(linear2_out)
        return self.dropout(norm_out)


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.multi_head_attention = Multi_Head_Attention()
        self.norm1 = Norm()
        self.feed_forward = Feed_Forward()
        self.norm2 = Norm()

    def forward(self, x):
        att_x = self.multi_head_attention.forward(x)
        norm1 = self.norm1.forward(att_x)
        adn_out1 = x + norm1

        ff_out = self.feed_forward.forward(adn_out1)
        norm2 = self.norm2.forward(ff_out)
        adn_out2 = adn_out1 + norm2

        return adn_out2


class MyTransformer(nn.Module):
    def __init__(self):
        super(MyTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional = Positional()
        self.encoder = nn.Sequential(*[Block() for _ in range(num_hidden_layers)])

    def forward(self, inputs):
        input_emb = self.embedding(inputs)
        positional_emb = self.positional.forward(inputs)
        input_embeddings = input_emb + positional_emb

        encoder_out = self.encoder.forward(input_embeddings)

        return encoder_out


if __name__ == '__main__':

    # bert = BertModel.from_pretrained(os.path.join('..', 'models', 'bert-base-chinese'))
    # print(bert)

    train_text, train_label = read_data(os.path.join('..', 'data', '文本分类', 'train.txt'), 2000)
    test_text, test_label = read_data(os.path.join('..', 'data', '文本分类', 'train.txt'))

    word_2_index, index_2_word = get_word_2_index(train_text)

    batch_size = 3
    epoch = 10
    vocab_size = len(word_2_index)
    embedding_dim = 768
    num_hidden_layers = 2
    feed_num = 1024

    train_dataset = MyDataset(train_text, train_label)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False, collate_fn=coll_fn)

    model = MyTransformer()
    for e in range(epoch):
        for text_idx, batch_label, batch_len in train_dataloader:
            out_ = model.forward(text_idx)
            print(out_)
    print("")


