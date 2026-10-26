"""
将 FB15K-237 的 Arrow 单文件数据（HuggingFace datasets 格式）转换为 train/valid/test 文本划分，
并与项目现有 KGDataset 格式兼容（每行：head\\trelation\\ttail）。

支持两种划分方式：
1. 若数据中已有 split 列，则按该列划分；
2. 否则按 FB15K-237 标准数量随机划分：train 272115, valid 17535, test 20466。
"""

from pathlib import Path
import argparse
import random

try:
    from datasets import Dataset
except ImportError:
    raise ImportError("请安装 datasets: pip install datasets")

# FB15K-237 标准划分数量（与官方 benchmark 一致）
FB15K237_TRAIN_SIZE = 272_115
FB15K237_VALID_SIZE = 17_535
FB15K237_TEST_SIZE = 20_466

# Freebase URI 前缀，输出时可选去掉以与现有 FB15k 文本格式一致
FREEBASE_PREFIX = "http://freebase.com"


def _norm(s: str) -> str:
    """去掉常见 URI 前缀，得到简短形式（如 /m/027rn）。"""
    if not s:
        return s
    s = s.strip()
    for prefix in ("http://freebase.com/", "http://freebase.com", "https://freebase.com/", "https://freebase.com"):
        if s.startswith(prefix):
            s = s[len(prefix) :].lstrip("/") or s
            break
    return "/" + s if s and not s.startswith("/") else s


def load_arrow_triples(path: Path, normalize_uri: bool = True):
    """从 HuggingFace .arrow 文件读取三元组，返回 (subject, predicate, object) 列表及 Dataset。"""
    ds = Dataset.from_file(str(path))
    cols = ds.column_names
    if "subject" in cols and "predicate" in cols and "object" in cols:
        subj, pred, obj = ds["subject"], ds["predicate"], ds["object"]
    elif "head" in cols and "relation" in cols and "tail" in cols:
        subj, pred, obj = ds["head"], ds["relation"], ds["tail"]
    else:
        raise ValueError(f"未识别的列名: {cols}，需要 subject/predicate/object 或 head/relation/tail")
    # 批量构建三元组，过滤空值并可选归一化 URI
    if normalize_uri:
        triples = [
            (_norm(str(h)), _norm(str(r)), _norm(str(t)))
            for h, r, t in zip(subj, pred, obj)
            if h is not None and r is not None and t is not None
        ]
    else:
        triples = [
            (str(h).strip(), str(r).strip(), str(t).strip())
            for h, r, t in zip(subj, pred, obj)
            if h is not None and r is not None and t is not None
        ]
    return triples, ds


def get_split_column(ds: Dataset):
    """若存在 split 列，返回列名；否则返回 None。"""
    for name in ("split", "set", "subset", "split_name"):
        if name in ds.column_names:
            return name
    return None


def write_triple_file(path: Path, triples: list, sep: str = "\t") -> None:
    """将三元组列表写入文本文件，每行 head\\trelation\\ttail。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triples:
            line = f"{h}{sep}{r}{sep}{t}\n"
            f.write(line)


def main():
    parser = argparse.ArgumentParser(description="将 FB15K-237 Arrow 数据划分为 train/valid/test")
    parser.add_argument(
        "--arrow_path",
        type=str,
        default=None,
        help="Arrow/Parquet 文件路径，默认: data/FB15k/fb15k-237-data.arrow",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="输出目录，默认与 arrow 文件同目录",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="fb15k237",
        help="输出文件名前缀，如 fb15k237 -> fb15k237-train.txt",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机划分时的种子",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    arrow_path = Path(args.arrow_path) if args.arrow_path else root / "data" / "FB15k" / "fb15k-237-data.arrow"
    out_dir = Path(args.out_dir) if args.out_dir else arrow_path.parent
    prefix = args.prefix

    if not arrow_path.exists():
        raise FileNotFoundError(f"未找到数据文件: {arrow_path}")

    print(f"读取: {arrow_path}")
    triples, table = load_arrow_triples(arrow_path)
    n = len(triples)
    print(f"共 {n} 条三元组")

    split_col_name = get_split_column(table)
    if split_col_name is not None:
        # 按已有 split 列划分
        split_col = table[split_col_name]
        train_triples = []
        valid_triples = []
        test_triples = []
        for i, (h, r, t) in enumerate(triples):
            val = split_col[i]
            s = str(val).lower() if val is not None else ""
            if "train" in s:
                train_triples.append((h, r, t))
            elif "valid" in s or "dev" in s:
                valid_triples.append((h, r, t))
            elif "test" in s:
                test_triples.append((h, r, t))
        print(f"按 split 列划分: train={len(train_triples)}, valid={len(valid_triples)}, test={len(test_triples)}")
    else:
        # 按 FB15K-237 标准数量随机划分
        if n != FB15K237_TRAIN_SIZE + FB15K237_VALID_SIZE + FB15K237_TEST_SIZE:
            print(f"警告: 总条数 {n} 与标准 310116 不一致，将按比例划分")
            n_train = int(n * FB15K237_TRAIN_SIZE / 310_116)
            n_test = int(n * FB15K237_TEST_SIZE / 310_116)
            n_valid = n - n_train - n_test
        else:
            n_train, n_valid, n_test = FB15K237_TRAIN_SIZE, FB15K237_VALID_SIZE, FB15K237_TEST_SIZE
        random.seed(args.seed)
        indices = list(range(n))
        random.shuffle(indices)
        train_triples = [triples[i] for i in indices[:n_train]]
        valid_triples = [triples[i] for i in indices[n_train : n_train + n_valid]]
        test_triples = [triples[i] for i in indices[n_train + n_valid :]]
        print(f"随机划分(seed={args.seed}): train={len(train_triples)}, valid={len(valid_triples)}, test={len(test_triples)}")

    train_path = out_dir / f"{prefix}-train.txt"
    valid_path = out_dir / f"{prefix}-valid.txt"
    test_path = out_dir / f"{prefix}-test.txt"

    write_triple_file(train_path, train_triples)
    write_triple_file(valid_path, valid_triples)
    write_triple_file(test_path, test_triples)

    print(f"已写入: {train_path}, {valid_path}, {test_path}")
    return train_path, valid_path, test_path


if __name__ == "__main__":
    main()
