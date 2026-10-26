'''实现对三元组数据的噪声注入'''
import os
import random
from typing import List, Tuple, Sequence


# 三元组和带标签三元组的类型别名
TripleId = Tuple[int, int, int]
LabeledTripleId = Tuple[int, int, int, int]  # (h, r, t, label)


def inject_noise(
    triples: Sequence[TripleId],
    num_entities: int,
    noise_ratio: float,
    mode: str = "head_tail",
) -> Tuple[List[TripleId], List[TripleId], List[LabeledTripleId]]:
    """
    在给定的干净三元组列表上注入噪声，生成一批错误三元组。
    """

    if not 0.0 <= noise_ratio <= 1.0:
        raise ValueError(f"noise_ratio 必须在 [0,1] 范围内，当前为 {noise_ratio}")

    clean_triples = list(triples)  # 拷贝一份
    num_clean = len(clean_triples)
    if num_clean == 0:
        return [], [], []

    # 需要注入的噪声条数
    noise_num = int(num_clean * noise_ratio)
    if noise_num == 0:
        all_with_label: List[LabeledTripleId] = [
            (h, r, t, 1) for (h, r, t) in clean_triples
        ]
        return clean_triples, [], all_with_label

    # 使用 set 存储已有三元组，方便快速查重，避免生成重复样本
    existing = set(clean_triples)

    # 从干净三元组中随机选取 noise_num 条作为待注噪的基础
    indices = random.sample(range(num_clean), k=noise_num)

    noisy_triples: List[TripleId] = []

    for idx in indices:
        h, r, t = clean_triples[idx]

        # 决定替换头还是替换尾
        if mode == "head_tail":
            replace_head = random.random() < 0.5
        else:
            raise ValueError(f"暂不支持的噪声模式: {mode}")

        # 反复尝试，直到生成一个不在 existing 集合中的新三元组
        # 为了避免极端情况下死循环，设置一个最大尝试次数
        max_tries = 100
        for _ in range(max_tries):
            if replace_head:
                new_h = random.randint(0, num_entities - 1)
                if new_h == h:
                    continue
                new_triple = (new_h, r, t)
            else:
                new_t = random.randint(0, num_entities - 1)
                if new_t == t:
                    continue
                new_triple = (h, r, new_t)

            if new_triple not in existing:
                noisy_triples.append(new_triple)
                existing.add(new_triple)
                break
        # 如果超过 max_tries 都没找到新的，直接跳过这一条，不强行生成

    # 带标签的组合列表：先是真实三元组 label=1，再是噪声三元组 label=0
    all_with_label: List[LabeledTripleId] = [
        (h, r, t, 1) for (h, r, t) in clean_triples
    ] + [
        (h, r, t, 0) for (h, r, t) in noisy_triples
    ]

    return clean_triples, noisy_triples, all_with_label


def inject_noise_with_pairs(
    triples: Sequence[TripleId],
    num_entities: int,
    noise_ratio: float,
    mode: str = "head_tail",
) -> Tuple[List[TripleId], List[TripleId], List[LabeledTripleId], List[Tuple[TripleId, TripleId]]]:
    """
    与 inject_noise 相同，但额外返回 pos_neg_pairs：
    每条成功生成的噪声对应一个 (干净三元组, 噪声三元组)，用于 pairwise ranking 训练
    （让模型学习 score(真) > score(假)）。
    """
    if not 0.0 <= noise_ratio <= 1.0:
        raise ValueError(f"noise_ratio 必须在 [0,1] 范围内，当前为 {noise_ratio}")

    clean_triples = list(triples)
    num_clean = len(clean_triples)
    pos_neg_pairs: List[Tuple[TripleId, TripleId]] = []
    if num_clean == 0:
        return [], [], [], []

    noise_num = int(num_clean * noise_ratio)
    if noise_num == 0:
        all_with_label: List[LabeledTripleId] = [(h, r, t, 1) for (h, r, t) in clean_triples]
        return clean_triples, [], all_with_label, []

    existing = set(clean_triples)
    indices = random.sample(range(num_clean), k=noise_num)
    noisy_triples: List[TripleId] = []

    for idx in indices:
        h, r, t = clean_triples[idx]
        pos = (h, r, t)

        if mode == "head_tail":
            replace_head = random.random() < 0.5
        else:
            raise ValueError(f"暂不支持的噪声模式: {mode}")

        max_tries = 100
        for _ in range(max_tries):
            if replace_head:
                new_h = random.randint(0, num_entities - 1)
                if new_h == h:
                    continue
                new_triple = (new_h, r, t)
            else:
                new_t = random.randint(0, num_entities - 1)
                if new_t == t:
                    continue
                new_triple = (h, r, new_t)

            if new_triple not in existing:
                noisy_triples.append(new_triple)
                existing.add(new_triple)
                pos_neg_pairs.append((pos, new_triple))
                break

    all_with_label = [
        (h, r, t, 1) for (h, r, t) in clean_triples
    ] + [
        (h, r, t, 0) for (h, r, t) in noisy_triples
    ]

    return clean_triples, noisy_triples, all_with_label, pos_neg_pairs


def save_noisy_version(
    path: str,
    labeled_triples: Sequence[LabeledTripleId],
    sep: str = "\t",
) -> None:
    """
    将带标签的三元组保存到磁盘，方便后续分析或复现实验。
    """
    # 确保输出目录存在
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for h, r, t, label in labeled_triples:
            f.write(f"{h}{sep}{r}{sep}{t}{sep}{label}\n")
