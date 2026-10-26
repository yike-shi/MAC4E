"""FB15K/FB15K-237 文本映射工具。

提供从实体/关系 ID（如 "/m/0y3_8"、"/music/genre/artists"）到可读文本的映射，
方便事实智能体将三元组转换为自然语言描述。

默认假定映射文件位于 `./data/FB15k` 目录下：
- entity2text.txt: 每行 `mid\ttext`
- relation2text.txt: 每行 `relation\ttext`

你也可以在调用时显式传入文件名。
"""

from __future__ import annotations

import os
from typing import Dict, Tuple


def load_fb15k237_text_maps(
    root: str = "./data/FB15k",
    ent_file: str = "entity2text.txt",
    rel_file: str = "relation2text.txt",
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """加载 FB15K / FB15K-237 的实体、关系文本映射表。

    参数：
    - root: 数据根目录（相对于项目根），默认 "./data/FB15k"；
    - ent_file: 实体映射文件名，默认 "entity2text.txt"；
    - rel_file: 关系映射文件名，默认 "relation2text.txt"。

    返回：
    - ent2text: {实体ID(如 /m/0y3_8) -> 可读文本(如 "Synthpop")}；
    - rel2text: {关系ID(如 /music/genre/artists) -> 文本(如 "artists")}。
    """
    ent2text: Dict[str, str] = {}
    rel2text: Dict[str, str] = {}

    ent_path = os.path.join(root, ent_file)
    rel_path = os.path.join(root, rel_file)

    if not os.path.exists(ent_path):
        raise FileNotFoundError(f"实体映射文件不存在: {ent_path}")
    if not os.path.exists(rel_path):
        raise FileNotFoundError(f"关系映射文件不存在: {rel_path}")

    # 加载实体文本
    with open(ent_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            mid, text = parts
            ent2text[mid] = text

    # 加载关系文本
    with open(rel_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            rel, text = parts
            rel2text[rel] = text

    return ent2text, rel2text


def load_wn18rr_text_maps(
    root: str = "./data/WN18RR",
    ent_file: str = "entity2text.txt",
    rel_file: str = "relation2text.txt",
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """加载 WN18RR 的实体、关系可读文本（与 FactVerificationAgent 一致：`mid\\ttext`）。

    若目录下已有 ``entity2text.txt`` / ``relation2text.txt``（与 FB 格式相同），则直接读取。
    否则从 ``entity2id.txt``、``relation2id.txt`` 合成简短描述：实体为 ``WordNet synset {id}``，
    关系为去掉前导 ``_`` 并将 ``_`` 换为空格（如 ``_hypernym`` -> ``hypernym``）。
    """
    ent2text: Dict[str, str] = {}
    rel2text: Dict[str, str] = {}
    ent_path = os.path.join(root, ent_file)
    rel_path = os.path.join(root, rel_file)

    if os.path.exists(ent_path) and os.path.exists(rel_path):
        with open(ent_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    ent2text[parts[0]] = parts[1]
        with open(rel_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    rel2text[parts[0]] = parts[1]
        return ent2text, rel2text

    ent_id_path = os.path.join(root, "entity2id.txt")
    rel_id_path = os.path.join(root, "relation2id.txt")
    if not os.path.exists(ent_id_path) or not os.path.exists(rel_id_path):
        return {}, {}

    with open(ent_id_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 1:
                eid = parts[0].strip()
                if eid:
                    ent2text[eid] = f"WordNet synset {eid}"

    with open(rel_id_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 1:
                rkey = parts[0].strip()
                if rkey:
                    rel2text[rkey] = rkey.lstrip("_").replace("_", " ")

    return ent2text, rel2text


def triple_id_to_text(
    triple: Tuple[int, int, int],
    id2entity: Dict[int, str],
    id2relation: Dict[int, str],
    ent2text: Dict[str, str],
    rel2text: Dict[str, str],
) -> Tuple[str, str, str]:
    """将 ID 形式三元组 (h_id, r_id, t_id) 转为可读文本三元组。

    若找不到对应文本，则退回用原始 ID/关系路径做简单替代。
    """
    h_id, r_id, t_id = triple

    h_mid = id2entity[h_id]
    t_mid = id2entity[t_id]
    r_key = id2relation[r_id]

    h_text = ent2text.get(h_mid, h_mid)
    t_text = ent2text.get(t_mid, t_mid)
    rel_text = rel2text.get(
        r_key,
        r_key.replace("/", " ").replace("_", " "),
    )

    return h_text, rel_text, t_text

