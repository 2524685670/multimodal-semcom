"""
图文检索任务头
图文检索不需要额外的可训练层，本质就是计算相似度 + 排序。
本模块封装检索评估逻辑。
"""

import torch
import torch.nn.functional as F
from typing import Dict

class RetrievalTaskHead:
    """图文检索评估器"""

    @staticmethod
    @torch.no_grad()
    def evaluate_retrieval(
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> Dict[str, float]:
        """
        计算图文检索指标

        Args:
            image_embeddings: [N, D]  所有图像的全局嵌入 (L2 normalized)
            text_embeddings:  [N, D]  所有文本的全局嵌入 (L2 normalized)
            假设: 第 i 张图 与第 i 条文本 互为正样本对

        Returns:
            dict with: i2t_r1, i2t_r5, i2t_r10,
                       t2i_r1, t2i_r5, t2i_r10, rsum
        """
        # L2 归一化（防御性）
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        N = image_embeddings.size(0)

        # 相似度矩阵 [N, N]
        sim = image_embeddings @ text_embeddings.T

        # ---- Image-to-Text Retrieval ----
        # 对每张图（每行），检查正确文本的排名
        i2t_ranks = []
        for i in range(N):
            row = sim[i]
            # 降序排列后，正确文本 (index=i) 的排名
            sorted_indices = row.argsort(descending=True)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
            i2t_ranks.append(rank)

        i2t_ranks = torch.tensor(i2t_ranks, dtype=torch.float)
        i2t_r1 = (i2t_ranks < 1).float().mean().item() * 100
        i2t_r5 = (i2t_ranks < 5).float().mean().item() * 100
        i2t_r10 = (i2t_ranks < 10).float().mean().item() * 100

        # ---- Text-to-Image Retrieval ----
        t2i_ranks = []
        for j in range(N):
            col = sim[:, j]
            sorted_indices = col.argsort(descending=True)
            rank = (sorted_indices == j).nonzero(as_tuple=True)[0].item()
            t2i_ranks.append(rank)

        t2i_ranks = torch.tensor(t2i_ranks, dtype=torch.float)
        t2i_r1 = (t2i_ranks < 1).float().mean().item() * 100
        t2i_r5 = (t2i_ranks < 5).float().mean().item() * 100
        t2i_r10 = (t2i_ranks < 10).float().mean().item() * 100

        rsum = i2t_r1 + i2t_r5 + i2t_r10 + t2i_r1 + t2i_r5 + t2i_r10

        return {
            "i2t_r1": i2t_r1,
            "i2t_r5": i2t_r5,
            "i2t_r10": i2t_r10,
            "t2i_r1": t2i_r1,
            "t2i_r5": t2i_r5,
            "t2i_r10": t2i_r10,
            "rsum": rsum,
        }