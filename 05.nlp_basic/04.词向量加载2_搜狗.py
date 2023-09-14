import os
import pickle

import numpy as np
from tqdm import tqdm


if __name__ == '__main__':

    with open(os.path.join('..', 'models', 'sgns.sogou.char', 'sgns.sogou.char'), encoding='utf8') as f:
        all_sg_content = f.read().split('\n')[1:]

    sg_vec = []
    sg_index_2_word = []
    for content in tqdm(all_sg_content):
        s_c = content.strip().split(' ')
        if len(s_c) != 301:
            continue
        sg_index_2_word.append(s_c[0])
        sg_vec.append(s_c[1:])
    sg_word_2_index = {w: i for i, w in enumerate(sg_index_2_word)}

    with open("word2vec.abc", "rb") as f:
        w1 = pickle.load(f)

    with open("word2index.abc", "rb") as f:
        word_2_index = pickle.load(f)
    index_2_word = list(word_2_index)

    for idx, word in enumerate(index_2_word):
        if word not in sg_index_2_word:
            continue
        sg_v = [float(i) for i in sg_vec[sg_word_2_index[word]]]
        w1[idx] = np.array(sg_v)

    with open("word2index.abc", "wb") as f:
        pickle.dump(w1, f)
