'''功能性关系推断（原类型约束中的类型判断已移除，仅保留 head->tails 统计与 is_functional）。'''
from collections import defaultdict
from typing import Dict, Set, Tuple, Iterable

TripleId = Tuple[int, int, int]


class TypeConstraints:
    """仅用于功能性关系推断（is_functional），供 Allocator 特征等使用。类型判断已移除。"""

    def __init__(self) -> None:
        # 每个关系在训练集中出现的 head -> {tails} 映射，用于推断功能性
        self.relation_head_to_tails: Dict[int, Dict[int, Set[int]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self.functional_relations: Set[int] = set()

    def build_from_triples(self, train_triples: Iterable[TripleId]) -> None:
        """从训练集三元组统计每个关系的 head -> {tails}，并推断功能性关系。"""
        for h, r, t in train_triples:
            self.relation_head_to_tails[r][h].add(t)

        self._infer_functional_relations()

    def _infer_functional_relations(
        self,
        max_avg_tails_per_head: float = 1.2,
        max_tails_for_any_head: int = 3,
        min_support: int = 10,
    ) -> None:
        """
        基于训练数据，自动推断「功能性关系」：

        - 对每个关系 r，统计：
            - 支持它的 head 数量 H_r
            - 对每个 head，其对应的 tail 集合大小 |T(h,r)|
            - 平均每个 head 有多少个不同的 tail（avg_tails）

        - 判定为功能关系的大致条件：
            1) 支持样本数要足够大（至少 min_support 个不同 head）
            2) 平均每个 head 的 tail 数量 <= max_avg_tails_per_head
            3) 对任何一个 head，tail 的数量不会太大（<= max_tails_for_any_head）
        """
        functional: Set[int] = set()
        for r, head_to_tails in self.relation_head_to_tails.items():
            num_heads = len(head_to_tails)
            if num_heads < min_support:
                continue

            total_tails = 0
            max_tails_seen = 0
            for tails in head_to_tails.values():
                count = len(tails)
                total_tails += count
                if count > max_tails_seen:
                    max_tails_seen = count

            avg_tails = total_tails / float(num_heads)

            if avg_tails <= max_avg_tails_per_head and max_tails_seen <= max_tails_for_any_head:
                functional.add(r)

        self.functional_relations = functional

    def is_functional(self, r: int) -> bool:
        """
        返回给定关系 r 是否被当前统计结果认为是「功能性关系」。
        """
        return r in self.functional_relations

