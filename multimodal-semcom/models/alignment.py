"""
对齐模块：投影头 + 双层对齐损失（实例级 + 单元级）+ 多样性约束

输入:
    visual_units: [B, Nv, unit_dim]
    text_units:   [B, Nt, unit_dim]
    visual_cls:   [B, visual_hidden_dim]
    text_cls:     [B, text_hidden_dim]

输出 dict:
    loss_global, loss_unit, loss_diversity,
    v_proj, t_proj, v_global, t_global, sim_matrix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignmentModule(nn.Module):

    def __init__(
        self,
        unit_dim: int = 512,
        proj_dim: int = 512,
        visual_hidden_dim: int = 768,
        text_hidden_dim: int = 512,
        temperature_init: float = 0.07,
    ):
        super().__init__()
        self.proj_dim = proj_dim

        # ---- 单元级投影头 (unit -> 公共空间) ----
        self.visual_unit_proj = nn.Sequential(
            nn.Linear(unit_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.text_unit_proj = nn.Sequential(
            nn.Linear(unit_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

        # ---- 实例级投影头 (CLIP CLS -> 公共空间) ----
        self.visual_global_proj = nn.Sequential(
            nn.Linear(visual_hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.text_global_proj = nn.Sequential(
            nn.Linear(text_hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        # ---- 可学习温度 ----
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature_init)))

        self._init_weights()

    def _init_weights(self):
        for proj in [self.visual_unit_proj, self.text_unit_proj,
                     self.visual_global_proj, self.text_global_proj]:
            for m in proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    # ---------- 投影 ----------

    def project_units(
        self,
        visual_units: torch.Tensor,
        text_units: torch.Tensor,
    ) -> tuple:
        """
        将语义单元投影到公共空间并 L2 归一化

        Args:
            visual_units: [B, Nv, unit_dim]
            text_units:   [B, Nt, unit_dim]

        Returns:
            v_proj: [B, Nv, proj_dim]  L2 normalized
            t_proj: [B, Nt, proj_dim]  L2 normalized
        """
        v_proj = self.visual_unit_proj(visual_units)        # [B, Nv, proj_dim]
        t_proj = self.text_unit_proj(text_units)            # [B, Nt, proj_dim]
        v_proj = F.normalize(v_proj, p=2, dim=-1)
        t_proj = F.normalize(t_proj, p=2, dim=-1)
        return v_proj, t_proj

    def project_global(
        self,
        visual_cls: torch.Tensor,
        text_cls: torch.Tensor,
    ) -> tuple:
        """
        将 CLIP CLS token 投影到公共空间并 L2 归一化

        Args:
            visual_cls: [B, visual_hidden_dim]
            text_cls:   [B, text_hidden_dim]

        Returns:
            v_global: [B, proj_dim]  L2 normalized
            t_global: [B, proj_dim]  L2 normalized
        """
        v_global = self.visual_global_proj(visual_cls)
        t_global = self.text_global_proj(text_cls)
        v_global = F.normalize(v_global, p=2, dim=-1)
        t_global = F.normalize(t_global, p=2, dim=-1)
        return v_global, t_global

    # ---------- 损失函数 ----------

    def compute_global_alignment_loss(
        self,
        v_global: torch.Tensor,
        t_global: torch.Tensor,
    ) -> torch.Tensor:
        """
        实例级对齐损失 (InfoNCE / CLIP-style symmetric contrastive loss)

        Args:
            v_global: [B, proj_dim]
            t_global: [B, proj_dim]

        Returns:
            loss: scalar
        """
        # 温度缩放
        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        # 相似度矩阵 [B, B]
        sim_i2t = logit_scale * (v_global @ t_global.T)
        sim_t2i = sim_i2t.T

        B = v_global.size(0)
        labels = torch.arange(B, device=v_global.device)

        loss_i2t = F.cross_entropy(sim_i2t, labels)
        loss_t2i = F.cross_entropy(sim_t2i, labels)

        return (loss_i2t + loss_t2i) / 2.0

    def compute_unit_alignment_loss(
        self,
        v_proj: torch.Tensor,
        t_proj: torch.Tensor,
    ) -> torch.Tensor:
        """
        单元级对齐损失
        基于互为最近邻 (Mutual Nearest Neighbor) 的对比学习

        Args:
            v_proj: [B, Nv, proj_dim]  L2 normalized
            t_proj: [B, Nt, proj_dim]  L2 normalized

        Returns:
            loss: scalar
        """
        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        # 单元级相似度矩阵 [B, Nv, Nt]
        sim = torch.bmm(v_proj, t_proj.transpose(1, 2)) * logit_scale

        B, Nv, Nt = sim.shape

        # ---- 行方向 (每个视觉单元找最佳文本单元) ----
        # softmax over text units
        log_prob_i2t = F.log_softmax(sim, dim=-1)           # [B, Nv, Nt]
        # 对每个视觉单元，取与其最相似的文本单元位置
        best_t_for_v = sim.argmax(dim=-1)                   # [B, Nv]

        # ---- 列方向 (每个文本单元找最佳视觉单元) ----
        log_prob_t2i = F.log_softmax(sim, dim=-2)           # [B, Nv, Nt]
        best_v_for_t = sim.argmax(dim=-2)                   # [B, Nt]

        # ---- 互为最近邻筛选 ----
        # 只有互为最近邻的对才参与损失计算
        loss_sum = torch.tensor(0.0, device=sim.device)
        count = 0

        for b in range(B):
            for i in range(Nv):
                j = best_t_for_v[b, i].item()
                # 检查是否互为最近邻
                if best_v_for_t[b, j].item() == i:
                    # 互为最近邻，最大化这对的 log 概率
                    loss_sum -= log_prob_i2t[b, i, j]
                    loss_sum -= log_prob_t2i[b, i, j]
                    count += 2

        if count == 0:
            # 没有找到互为最近邻的对，退化为全部 top-1 对
            targets_i2t = sim.argmax(dim=-1)                # [B, Nv]
            loss_i2t = F.cross_entropy(
                sim.reshape(B * Nv, Nt), targets_i2t.reshape(B * Nv)
            )
            targets_t2i = sim.argmax(dim=-2)                # [B, Nt]
            loss_t2i = F.cross_entropy(
                sim.permute(0, 2, 1).reshape(B * Nt, Nv), targets_t2i.reshape(B * Nt)
            )
            return (loss_i2t + loss_t2i) / 2.0

        return loss_sum / count

    def compute_diversity_loss(
        self,
        visual_units: torch.Tensor,
        text_units: torch.Tensor,
    ) -> torch.Tensor:
        """
        多样性约束损失：防止所有语义单元塌缩到同一个点

        最小化不同单元间的平均余弦相似度（排除对角线）

        Args:
            visual_units: [B, Nv, D]
            text_units:   [B, Nt, D]

        Returns:
            loss: scalar
        """
        def _diversity(units: torch.Tensor) -> torch.Tensor:
            # units: [B, N, D]
            units_norm = F.normalize(units, p=2, dim=-1)
            # [B, N, N]
            sim = torch.bmm(units_norm, units_norm.transpose(1, 2))
            B, N, _ = sim.shape
            # 去掉对角线
            mask = ~torch.eye(N, dtype=torch.bool, device=sim.device).unsqueeze(0)
            off_diag = sim.masked_select(mask)
            # 最小化非对角线的绝对值均值
            return off_diag.abs().mean()

        div_v = _diversity(visual_units)
        div_t = _diversity(text_units)
        return (div_v + div_t) / 2.0

    # ---------- 前向 ----------

    def forward(
        self,
        visual_units: torch.Tensor,
        text_units: torch.Tensor,
        visual_cls: torch.Tensor,
        text_cls: torch.Tensor,
    ) -> dict:
        """
        完整前向传播

        Args:
            visual_units: [B, Nv, unit_dim]
            text_units:   [B, Nt, unit_dim]
            visual_cls:   [B, visual_hidden_dim]
            text_cls:     [B, text_hidden_dim]

        Returns:
            dict with losses and projected embeddings
        """
        # 投影
        v_proj, t_proj = self.project_units(visual_units, text_units)
        v_global, t_global = self.project_global(visual_cls, text_cls)

        # 损失
        loss_global = self.compute_global_alignment_loss(v_global, t_global)
        loss_unit = self.compute_unit_alignment_loss(v_proj, t_proj)
        loss_div = self.compute_diversity_loss(visual_units, text_units)

        # 单元级相似度矩阵 (不带温度缩放, 用于后续纠错和可视化)
        sim_matrix = torch.bmm(v_proj, t_proj.transpose(1, 2))  # [B, Nv, Nt]

        return {
            "loss_global": loss_global,
            "loss_unit": loss_unit,
            "loss_diversity": loss_div,
            "v_proj": v_proj,           # [B, Nv, proj_dim]
            "t_proj": t_proj,           # [B, Nt, proj_dim]
            "v_global": v_global,       # [B, proj_dim]
            "t_global": t_global,       # [B, proj_dim]
            "sim_matrix": sim_matrix,   # [B, Nv, Nt]
        }