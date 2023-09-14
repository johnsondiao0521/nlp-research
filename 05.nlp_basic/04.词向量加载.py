import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def get_most_sim_word(word, topn):
    if word not in word_2_index:
        print("词库没有改词语")
        return None

    vec = w1[word_2_index[word]].reshape(1, -1)
    # for other_vec in w1:
    #     other_vec = other_vec.reshape(1, -1)
    #     cosine_similarity(vec, other_vec)[0][0]
    score = cosine_similarity(vec, w1)
    score = score[0]
    sim_word_idx = np.argsort(score)[::-1][:topn]
    sim_word_score = score[sim_word_idx]
    sim_word = [index_2_word[i] for i in sim_word_idx]
    result = [(i, j) for i, j in zip(sim_word, sim_word_score)]
    return result


def get_words_score(word1, word2):
    if word1 not in word_2_index or word2 not in word_2_index:
        print("词库没有改词语")
        return None
    vec1 = w1[word_2_index[word1]: word_2_index[word1] + 1]
    vec2 = w1[word_2_index[word2]: word_2_index[word2] + 1]

    score = cosine_similarity(vec1, vec2)
    return score


if __name__ == '__main__':

    with open("word2vec.abc", "rb") as f:
        w1 = pickle.load(f)

    with open("word2index.abc", "rb") as f:
        word_2_index = pickle.load(f)
    index_2_word = list(word_2_index)

    while True:
        input_word = input("请输入一个词语:")
        sp_word = input_word.split(' ')
        if len(sp_word) == 1:
            result = get_most_sim_word(input_word, 3)
            print(result)
        elif len(sp_word) == 2:
            result = get_words_score(*sp_word)
            print(result)
