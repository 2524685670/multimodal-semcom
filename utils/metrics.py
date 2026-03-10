"""
评估指标工具集
"""

import torch
import torch.nn.functional as F
from typing import Dict

def compute_recall_at_k(
    sim_matrix: torch.Tensor,
    k_values: list = [1, 5, 10],
) -> Dict[str, float]:
    """
    通用 Recall@K 计算
    假设 sim_matrix[i, i] 为正样本对

    Args:
        sim_matrix: [N, N] 相似度矩阵
        k_values: Recall@K 的 K 值列表

    Returns:
        dict, e.g. {"r@1": 50.0, "r@5": 80.0, "r@10": 90.0}
    """
    N = sim_matrix.size(0)
    results = {}

    # 降序排列
    _, sorted_indices = sim_matrix.sort(dim=-1, descending=True)

    for k in k_values:
        # 检查正确索引是否在 top-k 中
        top_k = sorted_indices[:, :k]                     # [N, k]
        correct = torch.arange(N, device=sim_matrix.device).unsqueeze(1)  # [N, 1]
        hits = (top_k == correct).any(dim=1).float()      # [N]
        results[f"r@{k}"] = hits.mean().item() * 100

    return results

def compute_mutual_nn_rate(sim_matrix: torch.Tensor) -> float:
    """
    计算互为最近邻匹配率 (Mutual Nearest Neighbor Rate)
    用于评估单元级对齐的质量

    Args:
        sim_matrix: [Nv, Nt] 单个样本的单元相似度矩阵

    Returns:
        mutual_nn_rate: 互为最近邻的比例 (0~1)
    """
    Nv, Nt = sim_matrix.shape
    # 每个视觉单元的最佳文本单元
    v2t = sim_matrix.argmax(dim=1)      # [Nv]
    # 每个文本单元的最佳视觉单元
    t2v = sim_matrix.argmax(dim=0)      # [Nt]

    mutual_count = 0
    for i in range(Nv):
        j = v2t[i].item()
        if j < Nt and t2v[j].item() == i:
            mutual_count += 1

    return mutual_count / min(Nv, Nt)