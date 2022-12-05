# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:johnsondiao
@File: test.py
@Time: 2022-12-05 13:46
@Desc:
"""
from sklearn.metrics import accuracy_score as sklearn_accuracy_score
from sklearn.metrics import precision_score as sklearn_precision_score
from sklearn.metrics import recall_score as sklearn_recall_score
from sklearn.metrics import f1_score as sklearn_f1_score

from seqeval.metrics import accuracy_score as seq_accuracy_score
from seqeval.metrics import precision_score as seq_precision_score
from seqeval.metrics import recall_score as seq_recall_score
from seqeval.metrics import f1_score as seq_f1_score

if __name__ == '__main__':
    # [今天的红烧肉非常好吃, 音乐也好听]
    seq_y_true = [['O', 'O', 'O', "B-FOOD", "I-FOOD", "E-FOOD", 'O', 'S-FOOD', 'O', 'O'],
                  ['B-MUS', 'E-MUS', 'O', 'O', 'O']]
    seq_y_pred = [['O', 'O', 'O', "B-FOOD", "B-FOOD", "B-FOOD", 'O', 'S-FOOD', 'O', 'O'],
                  ['B-MUS', 'E-MUS', 'O', 'O', 'O']]

    sklearn_y_true = ['O', 'O', 'O', "B-FOOD", "I-FOOD", "E-FOOD", 'O', 'S-FOOD', 'O', 'O', 'B-MUS', 'E-MUS', 'O', 'O',
                      'O']
    sklearn_y_pred = ['O', 'O', 'O', "B-FOOD", "B-FOOD", "B-FOOD", 'O', 'S-FOOD', 'O', 'O', 'B-MUS', 'E-MUS', 'O', 'O',
                      'O']
