"""
基本图构建与路径挖掘（PRA/BFS）。
"""

from collections import deque
from typing import List, Tuple, Iterable, Optional

try:
    import networkx as nx
except ImportError:
    nx = None  

TripleId = Tuple[int, int, int]
# 一条路径：节点序列 (h, e1, e2, ..., t) 或 关系序列 [r1, r2, ...]
PathNodes = Tuple[int, ...]   # (h, e1, e2, ..., t)
PathRels = Tuple[int, ...]    # (r1, r2, ...)


def build_graph(triples: Iterable[TripleId]):
    """
    从训练三元组构建有向多重图。
    """
    if nx is None:
        raise ImportError("需要安装 networkx: pip install networkx")

    G = nx.MultiDiGraph()
    for h, r, t in triples:
        # MultiDiGraph: add_edge(u, v, key=key)
        # 用 r 作为 key，这样 (h, t, r) 唯一，且能通过 G[u][v][r] 取到
        G.add_edge(h, t, key=r)
    return G


def bfs_paths(
    graph,
    h: int,
    t: int,
    max_len: int = 3,
    max_paths: int = 10,
    return_relation_sequence: bool = True,
) -> List[Tuple[PathNodes, PathRels]]:
    """
    用 BFS 从 h 出发寻找到 t 的路径，限制路径长度与数量。

    参数
    ----
    graph: nx.MultiDiGraph
        build_graph 返回的图。
    h, t: int
        起点实体 id、终点实体 id。
    max_len: int
        路径最大长度（边数），如 3 表示最多 3 条边、4 个节点。
    max_paths: int
        最多返回多少条路径（达到后提前结束）。
    return_relation_sequence: bool
        为 True 时每条路径同时返回 (节点序列, 关系序列)；为 False 时只关心关系序列也可用返回值 [1]。

    返回
    ----
    List[Tuple[PathNodes, PathRels]]
        每条元素为 ((h, e1, e2, ..., t), (r1, r2, ...))。
    """
    if nx is None:
        raise ImportError("需要安装 networkx: pip install networkx")

    if h == t:
        return [((h,), ())]  # 0 跳路径

    queue: deque = deque([(h, (h,), ())])
    found: List[Tuple[PathNodes, PathRels]] = []

    while queue and len(found) < max_paths:
        cur, node_seq, rel_seq = queue.popleft()
        cur_len = len(rel_seq)
        if cur_len >= max_len:
            continue
        if cur not in graph:
            continue
        for _, next_node, key in graph.out_edges(cur, keys=True):
            r = key
            new_node_seq = node_seq + (next_node,)
            new_rel_seq = rel_seq + (r,)
            if next_node == t:
                found.append((new_node_seq, new_rel_seq))
                if len(found) >= max_paths:
                    break
            else:
                queue.append((next_node, new_node_seq, new_rel_seq))

    return found


def count_support_paths(
    graph,
    h: int,
    r: int,
    t: int,
    max_len: int = 3,
    max_paths: int = 200,
) -> int:
    """
    对三元组 (h, r, t) 计算「支持路径」数量，用作后续特征。

    「支持路径」：从 h 能沿图走到 t 的路径。路径越多，说明 h 和 t 在图中的结构连接越丰富，
    常作为 PRA 类方法的特征，用于判断 (h,r,t) 是否合理。
    这里不区分关系类型，只数从 h 到 t 的路径数。

    max_paths: 最多枚举的路径数，达到后 BFS 提前结束，避免在稠密图上极慢。
    """
    if nx is None:
        raise ImportError("需要安装 networkx: pip install networkx")

    paths = bfs_paths(graph, h, t, max_len=max_len, max_paths=max_paths)
    return len(paths)
