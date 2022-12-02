# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:johnsondiao
@File: hmm_cut_word.py
@Time: 2022-04-20 9:29
@Desc: 使用hmm进行分词
"""
from tqdm import tqdm
import numpy as np
import os
import pickle


def read_file_convert_to_hmm_state(raw_file_path=os.path.join("all_train_text.txt"),
                                   save_file_path=os.path.join("all_train_state.txt")):
    """
    根据语料库得到所有标识。
    B:词语开始
    M:词语中间
    E:词语结束
    S:单独成词
    @param raw_file_path: 料库文件路径
    @param save_file_path: 标识文件路径
    @return:
    """
    if os.path.exists(save_file_path):
        return

    def make_label(word_):
        """
        从单词到label的转换，如：今天----->BE，麻辣肥牛----->BMME
        @param word_:
        @return:
        """
        n = len(word_)
        if n == 1:
            return "S"
        return "B" + (n - 2) * "M" + "E"

    fw = open(save_file_path, "w", encoding="utf-8")
    with open(raw_file_path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
        for index, line in tqdm(enumerate(lines)):
            if line:
                _state = ''
                for word in line.split(" "):
                    if word:
                        _state = _state + make_label(word) + " "
                if index != len(lines) - 1:
                    _state = _state.strip() + "\n"
                fw.write(_state)
        fw.close()


class HMM(object):
    def __init__(self, file_text=os.path.join("all_train_text.txt"), file_state=os.path.join("all_train_state.txt")):
        """
        定义HMM类，最关键的三大矩阵：初始矩阵、转移矩阵、发射矩阵。
        @param file_text:
        @param file_state:
        """
        self.all_text = open(file_text, "r", encoding="utf-8").read().split("\n")
        self.all_state = open(file_state, "r", encoding="utf-8").read().split("\n")
        print("all_text: ", len(self.all_text), "all_state: ", len(self.all_state))
        self.states_to_index = {"B": 0, "M": 1, "E": 2, "S": 3}
        self.index_to_states = ["B", "M", "E", "S"]
        self.len_states = len(self.states_to_index)

        self.init_matrix = np.zeros(self.len_states)
        self.transfer_matrix = np.zeros((self.len_states, self.len_states))
        self.emit_matrix = {"B": {"total": 0}, "M": {"total": 0}, "E": {"total": 0}, "S": {"total": 0}}

    def calc_init_matrix(self, state):
        self.init_matrix[self.states_to_index[state[0]]] += 1

    def calc_transfer_matrix(self, states):
        state_join = "".join(states)
        state1 = state_join[:-1]
        state2 = state_join[1:]
        for s1, s2 in zip(state1, state2):
            self.transfer_matrix[self.states_to_index[s1], self.states_to_index[s2]] += 1

    def calc_emit_matrix(self, words, states):
        for word, state in zip("".join(words), "".join(states)):
            self.emit_matrix[state][word] = self.emit_matrix[state].get(word, 0) + 1
            self.emit_matrix[state]["total"] += 1

    def normalize(self):
        self.init_matrix = self.init_matrix / np.sum(self.init_matrix)
        self.transfer_matrix = self.transfer_matrix / np.sum(self.transfer_matrix, axis=1, keepdims=True)
        self.emit_matrix = {state: {
            word: freq / word_freq['total'] * 1000
            for word, freq in word_freq.items() if word != "total"
        }
            for state, word_freq in self.emit_matrix.items()}

    def train(self, hmm_save_path=os.path.join("hmm_matrix.pkl")):
        if os.path.exists(hmm_save_path):  # 如果模型已经存在就不训练
            self.init_matrix, self.transfer_matrix, self.emit_matrix = pickle.load(open(hmm_save_path, "rb"))
            return

        for words, states in tqdm(zip(self.all_text, self.all_state)):  # 按行读取文件，调用3个矩阵的求解函数
            words = words.split(" ")
            states = states.split(" ")
            self.calc_init_matrix(states[0])  # 统计每篇⽂章的第⼀个字是什么状态
            self.calc_transfer_matrix(states)
            self.calc_emit_matrix(words, states)
        self.normalize()

        pickle.dump([self.init_matrix, self.transfer_matrix, self.emit_matrix], open(hmm_save_path, "wb"))


def viterbi(text, hmm):
    """
    维特比算法: 从众多路径中，迅速选择出最优路径
    @param text:分词文本
    @param hmm:hmm对象
    @return:返回分词文本对应的标识符
    """
    states = hmm.index_to_states
    start_p = hmm.init_matrix
    trans_p = hmm.transfer_matrix
    emit_p = hmm.emit_matrix
    V = [{}]
    path = {}
    for y in states:
        V[0][y] = start_p[hmm.states_to_index[y]] * emit_p[y].get(text[0], 0)
        path[y] = [y]

    for t in range(1, len(text)):
        V.append({})
        newpath = {}

        neverSeen = text[t] not in emit_p['B'].keys() and text[t] not in emit_p['M'].keys() and \
                    text[t] not in emit_p['E'].keys() and text[t] not in emit_p['S'].keys()
        for y in states:
            emitP = emit_p[y].get(text[t], 0) if not neverSeen else 1.0
            temp = []
            for y0 in states:
                if V[t - 1][y0] > 0:
                    temp.append((V[t - 1][y0] * trans_p[hmm.states_to_index[y0], hmm.states_to_index[y]] * emitP, y0))
            (prob, state) = max(temp)
            V[t][y] = prob
            newpath[y] = path[state] + [y]
        path = newpath

    (prob, state) = max([(V[len(text) - 1][y], y) for y in states])  # 求最大概念的路径

    # 分词
    result = ""
    for t, s in zip(text, path[state]):
        result += t
        if s == "S" or s == "E":
            result += " "
    return result


if __name__ == '__main__':
    read_file_convert_to_hmm_state()
    hmm = HMM()
    hmm.train()

    text = "虽然一路上队伍里肃静无声"
    res = viterbi(text, hmm)
    print(res)
