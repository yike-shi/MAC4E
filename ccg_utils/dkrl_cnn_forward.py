"""
与 DKRL 官方 Test_cnn.cpp 一致的 CNN 实体向量前向，及 TransE 式 L1 打分（仅用 cnn_vec + relation_vec）。

常量：window_1=2, window_2=1, n_pooling_1=4（与源码一致）。
"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np

WINDOW_1 = 2
WINDOW_2 = 1
N_POOLING_1 = 4


def _tanh_bounded(x: float) -> float:
    """(1 - exp(-2x)) / (1 + exp(-2x))，与 C++ 一致。"""
    ex = math.exp(-2.0 * x)
    return (1.0 - ex) / (1.0 + ex)


def forward_cnn_entity_vec(
    word_ids: Sequence[int],
    word_vec: np.ndarray,
    conv_w1: np.ndarray,
    conv_w2: np.ndarray,
    n: int,
    n_1: int,
    n_w: int,
) -> np.ndarray:
    """
    单实体：词 ID 序列 -> n 维 cnn_vec（与 Test_cnn.cpp run() 中构建 cnn_vec 一致）。
    word_vec: (word_num, n_w)
    conv_w1: (n_1, n_w * window_1)
    conv_w2: (n, n_1 * window_2)
    """
    words = list(word_ids)
    if len(words) == 0:
        words = [0]

    l1_length = (len(words) - 1) // N_POOLING_1 + 1
    mid_vec = np.zeros((l1_length, n_1), dtype=np.float64)

    for k in range(l1_length):
        for j in range(n_1):
            temp_max = -1e100
            for i in range(N_POOLING_1 * k, N_POOLING_1 * (k + 1)):
                if i >= len(words):
                    break
                temp_token = 0.0
                for ii in range(WINDOW_1):
                    if i + ii >= len(words):
                        break
                    wid = int(words[i + ii])
                    for iii in range(n_w):
                        temp_token += float(word_vec[wid, iii]) * float(conv_w1[j, ii * n_w + iii])
                if temp_max < temp_token:
                    temp_max = temp_token
            mid_vec[k, j] = _tanh_bounded(temp_max)

    cnn = np.zeros(n, dtype=np.float64)
    for j in range(n):
        temp_expo = 0.0
        temp_word_num = 0
        for i in range(l1_length):
            temp_token = 0.0
            for ii in range(WINDOW_2):
                if i + ii >= l1_length:
                    break
                for iii in range(n_1):
                    temp_token += float(mid_vec[i + ii, iii]) * float(conv_w2[j, ii * n_1 + iii])
            temp_expo += temp_token
            temp_word_num += 1
        if temp_word_num <= 0:
            temp_word_num = 1
        temp_expo /= temp_word_num
        ex = math.exp(-2.0 * temp_expo)
        cnn[j] = (1.0 - ex) / (1.0 + ex)

    return cnn


def build_all_cnn_vec(
    entity_words: List[List[int]],
    word_vec: np.ndarray,
    conv_w1: np.ndarray,
    conv_w2: np.ndarray,
    n: int,
    n_1: int,
    n_w: int,
) -> np.ndarray:
    """对所有实体逐条前向，返回 (entity_num, n)。"""
    e_num = len(entity_words)
    out = np.zeros((e_num, n), dtype=np.float64)
    for ent in range(e_num):
        out[ent] = forward_cnn_entity_vec(
            entity_words[ent], word_vec, conv_w1, conv_w2, n, n_1, n_w
        )
    return out


def dkrl_struct_score_l1(
    cnn_vec: np.ndarray,
    relation_vec: np.ndarray,
    h: int,
    r: int,
    t: int,
) -> float:
    """
    与 Test_cnn::calc_sum 一致：sum_i -|cnn[t]-cnn[h]-rel[r]|。
    越大表示三元组越「一致」。
    """
    d = cnn_vec[t] - cnn_vec[h] - relation_vec[r]
    return float(-np.abs(d).sum())


def dkrl_struct_confidence_sigmoid(
    cnn_vec: np.ndarray,
    relation_vec: np.ndarray,
    h: int,
    r: int,
    t: int,
) -> float:
    """与 TransE 基线类似的 (0,1) 置信度，便于阈值与 AUC。"""
    raw = dkrl_struct_score_l1(cnn_vec, relation_vec, h, r, t)
    return float(1.0 / (1.0 + math.exp(-raw)))
