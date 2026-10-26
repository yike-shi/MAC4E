# 仅测试 TypeConstraints 的功能性关系推断（类型判断 check_type 已移除）
from ccg_data.type_constraints import TypeConstraints


def test_functional():
    # 关系 100: 每个 head 对应 1 个 tail，近似功能性
    # 关系 200: 每个 head 对应多个 tail，非功能性
    mock_train_triples = [
        (1, 100, 10), (2, 100, 11), (3, 100, 12), (4, 100, 13), (5, 100, 14),
        (6, 100, 15), (7, 100, 16), (8, 100, 17), (9, 100, 18), (10, 100, 19),
        (1, 200, 50), (1, 200, 51), (1, 200, 52),
        (2, 200, 53), (2, 200, 54),
    ]

    tc = TypeConstraints()
    tc.build_from_triples(mock_train_triples)
    tc._infer_functional_relations(min_support=5)

    print("--- 功能性检测 ---")
    print(f"关系 100 是否功能性: {tc.is_functional(100)}")  # 应为 True
    print(f"关系 200 是否功能性: {tc.is_functional(200)}")  # 应为 False


if __name__ == "__main__":
    test_functional()
