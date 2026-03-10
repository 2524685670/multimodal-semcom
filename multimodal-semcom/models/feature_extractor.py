"""
CLIP 特征提取器
完全冻结，不参与训练，仅用于提取视觉和文本的 token-level 特征。

输入:
    images:             [B, 3, 224, 224]
    text_input_ids:     [B, 77]
    text_attention_mask: [B, 77]

输出 dict:
    visual_tokens:  [B, 49, 768]   # 49 = 7×7 patch tokens (去掉CLS)
    visual_cls:     [B, 768]       # CLS token
    text_tokens:    [B, L, 512]    # 有效 text tokens (去掉 padding)
    text_cls:       [B, 512]       # EOS/pooled token
"""

import torch
import torch.nn as nn
from transformers import CLIPModel

class CLIPFeatureExtractor(nn.Module):
    """
    封装预训练 CLIP，冻结所有参数，提取 token 级别的隐藏状态。
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        print(f"[FeatureExtractor] 加载 CLIP: {model_name}")
        self.clip = CLIPModel.from_pretrained(model_name)

        # 冻结所有参数
        for param in self.clip.parameters():
            param.requires_grad = False
        self.clip.eval()

        # 记录维度
        self.visual_hidden_dim = self.clip.config.vision_config.hidden_size   # 768
        self.text_hidden_dim = self.clip.config.text_config.hidden_size       # 512

        print(f"[FeatureExtractor] visual_hidden_dim={self.visual_hidden_dim}, "
              f"text_hidden_dim={self.text_hidden_dim}")

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> dict:
        """
        前向传播（无梯度）

        Args:
            images: [B, 3, 224, 224]
            text_input_ids: [B, seq_len]
            text_attention_mask: [B, seq_len]

        Returns:
            dict with keys:
                visual_tokens [B, num_patches, visual_hidden_dim]
                visual_cls    [B, visual_hidden_dim]
                text_tokens   [B, seq_len, text_hidden_dim]
                text_cls      [B, text_hidden_dim]
        """
        # ---- 视觉侧 ----
        vision_outputs = self.clip.vision_model(
            pixel_values=images,
            output_hidden_states=True,
            return_dict=True,
        )
        # last_hidden_state: [B, num_patches+1, hidden_dim]
        # 第一个 token 是 CLS
        visual_last_hidden = vision_outputs.last_hidden_state   # [B, 50, 768]
        visual_cls = visual_last_hidden[:, 0, :]                # [B, 768]
        visual_tokens = visual_last_hidden[:, 1:, :]            # [B, 49, 768]

        # ---- 文本侧 ----
        text_outputs = self.clip.text_model(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # last_hidden_state: [B, seq_len, hidden_dim]
        text_last_hidden = text_outputs.last_hidden_state       # [B, 77, 512]

        # pooled_output 是 EOS token 的投影 (CLIP text 用 EOS 而非 CLS)
        # 但 pooled_output 经过了额外投影层，我们用 last_hidden_state 的 EOS 位置
        # EOS 位置 = 每个样本中 input_ids 最后一个非 padding 位置
        # 简化方案：用 pooled_output 作为 text_cls
        text_cls = text_outputs.pooler_output                   # [B, 512]
        if text_cls is None:
            # fallback: 取 attention_mask 指示的最后一个有效 token
            seq_lengths = text_attention_mask.sum(dim=1) - 1     # [B]
            text_cls = text_last_hidden[
                torch.arange(text_last_hidden.size(0)), seq_lengths
            ]

        text_tokens = text_last_hidden                           # [B, 77, 512]

        return {
            "visual_tokens": visual_tokens,     # [B, 49, 768]
            "visual_cls": visual_cls,           # [B, 768]
            "text_tokens": text_tokens,         # [B, 77, 512]
            "text_cls": text_cls,               # [B, 512]
        }