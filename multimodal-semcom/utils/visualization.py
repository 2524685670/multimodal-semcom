"""
可视化工具
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")       # 无头模式，服务器可用
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

def plot_similarity_matrix(
    sim_matrix: torch.Tensor,
    save_path: str,
    title: str = "Unit-Level Similarity Matrix",
    figsize: tuple = (8, 6),
):
    """
    绘制单元级相似度矩阵热力图

    Args:
        sim_matrix: [Nv, Nt]  单个样本的相似度矩阵
        save_path: 保存路径
        title: 图标题
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if isinstance(sim_matrix, torch.Tensor):
        sim_matrix = sim_matrix.detach().cpu().numpy()

    Nv, Nt = sim_matrix.shape

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        sim_matrix,
        ax=ax,
        cmap="viridis",
        annot=(Nv <= 16 and Nt <= 16),     # 小矩阵标数字
        fmt=".2f" if (Nv <= 16 and Nt <= 16) else "",
        xticklabels=[f"T{i}" for i in range(Nt)],
        yticklabels=[f"V{i}" for i in range(Nv)],
        vmin=-1, vmax=1,
    )
    ax.set_xlabel("Text Units")
    ax.set_ylabel("Visual Units")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Viz] 热力图已保存: {save_path}")

def plot_training_curves(
    log_dict: Dict[str, List[float]],
    save_path: str,
    title: str = "Training Curves",
    figsize: tuple = (12, 5),
):
    """
    绘制训练曲线（多指标）

    Args:
        log_dict: {metric_name: [epoch0_val, epoch1_val, ...]}
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 分成两个子图：loss 和 metrics
    loss_keys = [k for k in log_dict if "loss" in k.lower()]
    metric_keys = [k for k in log_dict if "loss" not in k.lower()]

    num_plots = (1 if loss_keys else 0) + (1 if metric_keys else 0)
    if num_plots == 0:
        return

    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    if num_plots == 1:
        axes = [axes]

    idx = 0
    if loss_keys:
        ax = axes[idx]
        for key in loss_keys:
            ax.plot(log_dict[key], label=key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Losses")
        ax.legend()
        ax.grid(True, alpha=0.3)
        idx += 1

    if metric_keys:
        ax = axes[idx]
        for key in metric_keys:
            ax.plot(log_dict[key], label=key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.set_title("Metrics")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Viz] 训练曲线已保存: {save_path}")

def plot_retrieval_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: str,
    title: str = "Retrieval Performance Comparison",
    figsize: tuple = (10, 5),
):
    """
    绘制检索指标柱状对比图

    Args:
        results: {method_name: {"i2t_r1": ..., "t2i_r1": ..., ...}}
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    methods = list(results.keys())
    metrics = ["i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10"]

    x = np.arange(len(metrics))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=figsize)
    for i, method in enumerate(methods):
        values = [results[method].get(m, 0) for m in metrics]
        ax.bar(x + i * width, values, width, label=method)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Recall (%)")
    ax.set_title(title)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(metrics, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Viz] 检索对比图已保存: {save_path}")