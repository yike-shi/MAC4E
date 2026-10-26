'''把“纯文本三元组文件”变成“可训练用的 ID 形式数据集”的工具类'''

import os
from typing import Dict, List, Tuple

#区分“字符串形式的三元组”和“ID 形式的三元组”
TripleStr = Tuple[str, str, str]
TripleId = Tuple[int, int, int]


class KGDataset:
    """
    知识图谱数据加载类，用于直接读取 FB15K / WN18RR / NELL-995 等三元组文件。

    - 假设每行格式为：head<TAB>relation<TAB>tail
    - 会自动构建实体 / 关系到 ID 的映射
    - 并把三元组全部转换成 ID 形式
    """

    def __init__(self) -> None:
        # 字符串到 id 的映射
        self.entities: Dict[str, int] = {}
        self.relations: Dict[str, int] = {}
        # id 到字符串的反向映射（如需可用）
        self.id2entity: Dict[int, str] = {}
        self.id2relation: Dict[int, str] = {}

        # 各个划分的三元组（ID 形式）
        self.train_triples: List[TripleId] = []
        self.valid_triples: List[TripleId] = []
        self.test_triples: List[TripleId] = []

    # ========= 对外接口 =========

    @classmethod
    def load_from_files(
        cls,
        train_path: str,
        valid_path: str = None,
        test_path: str = None,
    ) -> "KGDataset":
        """
        从给定的 train / valid / test 文本文件加载，并返回一个 KGDataset 实例。
        """
        ds = cls()

        # 从文件中读取原始字符串三元组
        train_triples_str = ds._read_triple_file(train_path)
        valid_triples_str: List[TripleStr] = []
        test_triples_str: List[TripleStr] = []

        if valid_path:
            valid_triples_str = ds._read_triple_file(valid_path)
        if test_path:
            test_triples_str = ds._read_triple_file(test_path)

        # 汇总所有三元组，构建实体 / 关系词表
        all_triples: List[TripleStr] = []
        all_triples.extend(train_triples_str)
        all_triples.extend(valid_triples_str)
        all_triples.extend(test_triples_str)

        ds._build_vocab(all_triples)

        # 将三元组全部映射为 ID 形式
        ds.train_triples = ds._map_triples_to_ids(train_triples_str)
        ds.valid_triples = ds._map_triples_to_ids(valid_triples_str) if valid_triples_str else []
        ds.test_triples = ds._map_triples_to_ids(test_triples_str) if test_triples_str else []

        return ds

    def get_all_triples(self, split: str = "train") -> List[TripleId]:
        """
        返回指定划分的三元组列表（ID 形式）。
        """
        split = split.lower()
        if split == "train":
            return self.train_triples
        if split == "valid":
            return self.valid_triples
        if split == "test":
            return self.test_triples
        if split == "all":
            return list(self.train_triples) + list(self.valid_triples) + list(self.test_triples)
        raise ValueError(f"未知数据划分: {split}")

    @property
    def num_entities(self) -> int:
        return len(self.entities)

    @property
    def num_relations(self) -> int:
        return len(self.relations)

    # ========= 内部辅助函数 =========

    def _read_triple_file(self, path: str) -> List[TripleStr]:
        """
        从文本文件中读取三元组，返回 (h_str, r_str, t_str) 列表。
        """
        triples: List[TripleStr] = []

        if path is None:
            return triples
        if not os.path.exists(path):
            raise FileNotFoundError(f"Triple file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 3:
                    # 兼容空格分隔的情况
                    parts = line.split()
                    if len(parts) != 3:
                        continue
                h, r, t = parts
                triples.append((h, r, t))

        return triples

    def _build_vocab(self, triples: List[TripleStr]) -> None:
        """
        根据所有三元组构建实体 / 关系到 ID 的映射。
        """
        for h, r, t in triples:
            if h not in self.entities:
                idx = len(self.entities)
                self.entities[h] = idx
                self.id2entity[idx] = h
            if t not in self.entities:
                idx = len(self.entities)
                self.entities[t] = idx
                self.id2entity[idx] = t
            if r not in self.relations:
                idx = len(self.relations)
                self.relations[r] = idx
                self.id2relation[idx] = r

    def _map_triples_to_ids(self, triples: List[TripleStr]) -> List[TripleId]:
        """
        将字符串形式三元组映射为 ID 形式三元组。
        """
        mapped: List[TripleId] = []
        for h, r, t in triples:
            h_id = self.entities[h]
            t_id = self.entities[t]
            r_id = self.relations[r]
            mapped.append((h_id, r_id, t_id))
        return mapped


if __name__ == "__main__":
    # 简单自测
    fb15k_train = "./data/FB15k/freebase_mtr100_mte100-train.txt"
    fb15k_valid = "./data/FB15k/freebase_mtr100_mte100-valid.txt"
    fb15k_test = "./data/FB15k/freebase_mtr100_mte100-test.txt"

    ds = KGDataset.load_from_files(fb15k_train, fb15k_valid, fb15k_test)
    print(f"#entities: {ds.num_entities}, #relations: {ds.num_relations}")
    print(f"#train triples: {len(ds.train_triples)}")
    print(f"#valid triples: {len(ds.valid_triples)}")
    print(f"#test triples: {len(ds.test_triples)}")

