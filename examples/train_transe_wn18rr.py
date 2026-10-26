import sys
from pathlib import Path

import torch

# 保证从 CC_gra 根目录可导入 ccg_pipeline
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ccg_pipeline.train_embeddings import train_transe_for_dataset


def main() -> None:
    """
    在 WN18RR 数据集上训练 TransE，并保存到 outputs/embeddings/wn18rr_transE.pt。

    路径约定：
    - 三元组数据：ccg_config/datasets.yaml 中的 wn18rr 配置，
      指向 CC_gra/data/WN18RR 下的 train/valid/test.txt。
    - 模型与日志输出：
      outputs/embeddings/wn18rr_transE.pt
      outputs/logs/train_wn18rr_*.log
    """
    dataset_name = "wn18rr"

    # 若系统有多张 GPU，可通过 CUDA_VISIBLE_DEVICES 控制使用哪一张；
    # 这里直接按当前可见设备选择 cuda，否则回退到 cpu。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 如需快速试跑可传 epochs_override，例如 50；不传则使用 model.yaml 中的 embedding.epochs
    model, dataset = train_transe_for_dataset(
        dataset_name=dataset_name,
        seed=42,
        device=device,
        epochs_override=None,
    )

    print(
        f"TransE training finished on {dataset_name}. "
        f"Entities={dataset.num_entities}, Relations={dataset.num_relations}"
    )


if __name__ == "__main__":
    main()


