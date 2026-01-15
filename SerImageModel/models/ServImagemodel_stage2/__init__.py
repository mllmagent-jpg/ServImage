"""
Stage2 Model - 两阶段训练的第二阶段
加载 Stage1 模型和原始 VLM 模型，进行特征拼接后训练新的分类器
"""

from .model import Stage2Model
from .modifications import SingleTaskQualityDecoder, apply_stage2_modifications, apply_lora_to_stage2
from .config import get_stage2_config

__all__ = ['Stage2Model', 'SingleTaskQualityDecoder', 'apply_stage2_modifications', 'apply_lora_to_stage2', 'get_stage2_config']
