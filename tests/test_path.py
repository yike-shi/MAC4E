from ccg_data.path_mining import build_graph, bfs_paths, count_support_paths

def test_path_mining():
    # 1. 构造模拟三元组 (h, r, t)
    # 模拟一个逻辑链路：北京 --(位于)--> 中国 --(首都)--> 北京 (环路测试)
    # 模拟路径：可 --(就读于)--> 北航 --(坐落于)--> 北京
    # 实体ID：可(1), 北航(2), 北京(3), 中国(4)
    # 关系ID：就读于(10), 坐落于(11), 位于(12), 首都(13)
    
    mock_triples = [
        (1, 10, 2),  # 可 --就读于--> 北航
        (2, 11, 3),  # 北航 --坐落于--> 北京
        (3, 12, 4),  # 北京 --位于--> 中国
        (4, 13, 3),  # 中国 --首都--> 北京
    ]

    # 2. 建图
    print("正在构建知识图谱...")
    G = build_graph(mock_triples)

    # 3. 测试从“可”到“北京”的路径 (预期：可->北航->北京)
    print("\n--- 路径挖掘测试 [可 -> 北京] ---")
    paths = bfs_paths(G, h=1, t=3, max_len=3)
    
    for i, (nodes, rels) in enumerate(paths):
        print(f"路径 {i+1}: 节点序列 {nodes}, 关系序列 {rels}")

    # 4. 测试支持路径计数
    # 从“北航”到“中国”应该有一条长度为 2 的路径 (北航->北京->中国)
    print("\n--- 支持度计数测试 [北航 -> 中国] ---")
    support_count = count_support_paths(G, h=2, r=999, t=4, max_len=3)
    print(f"北航到中国之间的结构支持路径数: {support_count}")

    # 验证逻辑
    assert len(paths) > 0, "未能找到有效路径"
    assert nodes == (1, 2, 3), "路径节点顺序错误"
    assert rels == (10, 11), "路径关系识别错误"
    print("\n测试通过！逻辑完全正确。")

if __name__ == "__main__":
    test_path_mining()
