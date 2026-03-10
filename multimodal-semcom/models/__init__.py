"""
models 包初始化
定义全局共享的核心数据结构
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch

@dataclass
class SemanticUnit:
    """语义单元：系统中流动的基本数据单位"""
    unit_id: int                                    # 单元序号 0..N-1
    modality: int                                   # 0=image, 1=text
    embedding: Optional[torch.Tensor] = None        # [D] 高维对齐空间中的嵌入
    rvq_indices: Optional[torch.Tensor] = None      # [S] 每个stage的码本索引
    importance_score: float = 0.0                   # 任务重要度 ΔL
    importance_level: str = "L"                     # "H" / "M" / "L"
    is_missing: Optional[List[bool]] = None         # [S] 每个stage是否缺失
    is_corrected: bool = False                      # 是否经过纠错

@dataclass
class TransmissionPacket:
    """传输包：信道中流动的单位"""
    unit_id: int
    modality: int
    stage: int                     # 当前包携带的RVQ stage编号
    index: int                     # 该stage的码本索引
    importance_level: str          # "H"/"M"/"L"
    crc_valid: bool = True         # CRC校验结果