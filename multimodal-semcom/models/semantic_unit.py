"""
语义单元聚合模块
使用 Learnable Queries + Cross-Attention 将可变数量的 token
聚合为固定数量的语义单元。

输入:
    visual_tokens: [B, 49, 768]
    text_tokens:   [B, 77, 512]
    text_attention_mask: [B, 77] (可选，用于屏蔽 padding)

输出 dict:
    visual_units: [B, Nv, unit_dim]   # Nv 个图像语义单元
    text_units:   [B, Nt, unit_dim]   # Nt 个文本语义单元
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class CrossAttentionBlock(nn.Module):
    """
    单个 Cross-Attention 块：
    Query 来自 learnable queries，Key/Value 来自原始 token。
    结构: CrossAttn -> Add & LN -> FFN -> Add & LN
    """

    def __init__(self, embed_dim: int = 512, num_heads: int = 8,
                 ffn_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,   # 输入格式为 [B, L, D]
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ffn_ratio, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        kv_source: torch.Tensor,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query:     [B, N_query, D]  learnable queries
            kv_source: [B, L_source, D] 原始 token 特征
            kv_mask:   [B, L_source] bool, True=有效, False=padding (可选)

        Returns:
            updated query: [B, N_query, D]
        """
        # key_padding_mask 要求: True = 被忽略的位置
        key_padding_mask = None
        if kv_mask is not None:
            key_padding_mask = ~kv_mask  # 反转：True 表示需要 mask 掉

        # Cross-Attention
        attn_out, _ = self.cross_attn(
            query=query,
            key=kv_source,
            value=kv_source,
            key_padding_mask=key_padding_mask,
        )
        query = self.norm1(query + attn_out)

        # FFN
        ffn_out = self.ffn(query)
        query = self.norm2(query + ffn_out)

        return query

class SemanticUnitAggregator(nn.Module):
    """
    将 CLIP 提取的 token-level 特征聚合为固定数量的语义单元。
    """

    def __init__(
        self,
        visual_input_dim: int = 768,
        text_input_dim: int = 512,
        unit_dim: int = 512,
        num_visual_units: int = 16,
        num_text_units: int = 16,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_visual_units = num_visual_units
        self.num_text_units = num_text_units
        self.unit_dim = unit_dim

        # ---- 输入维度映射（统一到 unit_dim）----
        self.visual_input_proj = nn.Linear(visual_input_dim, unit_dim)
        self.text_input_proj = nn.Linear(text_input_dim, unit_dim)

        # ---- Learnable Queries ----
        self.visual_queries = nn.Parameter(
            torch.randn(1, num_visual_units, unit_dim) * 0.02
        )
        self.text_queries = nn.Parameter(
            torch.randn(1, num_text_units, unit_dim) * 0.02
        )

        # ---- Cross-Attention 层 ----
        self.visual_cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(unit_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.text_cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(unit_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # ---- 模态标识嵌入 ----
        # 0 = image, 1 = text
        self.modality_embedding = nn.Embedding(2, unit_dim)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.trunc_normal_(self.visual_queries, std=0.02)
        nn.init.trunc_normal_(self.text_queries, std=0.02)
        nn.init.normal_(self.modality_embedding.weight, std=0.02)

        for module in [self.visual_input_proj, self.text_input_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            visual_tokens: [B, 49, 768]  CLIP 视觉 patch tokens
            text_tokens:   [B, 77, 512]  CLIP 文本 tokens
            text_attention_mask: [B, 77]  文本有效位置掩码（可选）

        Returns:
            dict:
                visual_units: [B, Nv, unit_dim]
                text_units:   [B, Nt, unit_dim]
        """
        B = visual_tokens.size(0)
        device = visual_tokens.device

        # ---- 1. 维度映射 ----
        visual_kv = self.visual_input_proj(visual_tokens)   # [B, 49, unit_dim]
        text_kv = self.text_input_proj(text_tokens)         # [B, 77, unit_dim]

        # ---- 2. 展开 Learnable Queries 到 batch ----
        v_queries = self.visual_queries.expand(B, -1, -1)   # [B, Nv, unit_dim]
        t_queries = self.text_queries.expand(B, -1, -1)     # [B, Nt, unit_dim]

        # ---- 3. Cross-Attention 聚合 ----
        for layer in self.visual_cross_attn_layers:
            v_queries = layer(query=v_queries, kv_source=visual_kv)

        for layer in self.text_cross_attn_layers:
            t_queries = layer(
                query=t_queries,
                kv_source=text_kv,
                kv_mask=text_attention_mask,
            )

        # ---- 4. 加入模态嵌入 ----
        visual_mod_ids = torch.zeros(B, self.num_visual_units, dtype=torch.long, device=device)
        text_mod_ids = torch.ones(B, self.num_text_units, dtype=torch.long, device=device)

        visual_units = v_queries + self.modality_embedding(visual_mod_ids)
        text_units = t_queries + self.modality_embedding(text_mod_ids)

        return {
            "visual_units": visual_units,   # [B, Nv, unit_dim]
            "text_units": text_units,       # [B, Nt, unit_dim]
        }