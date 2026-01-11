"""
模型变体4：CBM (Concept Bottleneck Model) 版本
对 Qwen3VL 进行概念瓶颈模型改造，支持多维度质量评估
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from SerImageModel.models.base.qwen3vl_base import Qwen3VLBase
from .modifications import apply_variant4_modifications
from .config import get_variant4_config


class Variant4Model(Qwen3VLBase):
    """
    模型变体4 - CBM (Concept Bottleneck Model)

    特性：
    - 概念瓶颈架构：BRF + VEQ + CNS 三维度概念预测
    - 多任务学习：支持概念监督 + 最终分类
    - 可解释性：可查看每个概念的预测结果
    - 针对 train_cbm 数据集设计
    """

    def __init__(
        self,
        model_name_or_path=None,
        cache_dir=None,
        attn_implementation="flash_attention_2",
        dtype=None,
        variant4_config=None,
    ):
        """
        初始化变体4模型

        Args:
            model_name_or_path: 模型路径
            cache_dir: 缓存目录
            attn_implementation: 注意力实现方式
            dtype: 数据类型
            variant4_config: Variant4 配置字典
        """
        # 获取配置
        if variant4_config is None:
            variant4_config = get_variant4_config()

        # 如果没有提供model_name_or_path，使用配置中的默认值
        if model_name_or_path is None:
            model_name_or_path = variant4_config.get('model_name_or_path', 'Qwen/Qwen3-VL-2B-Instruct')

        # 保存配置
        self.variant4_config = variant4_config

        # 调用父类初始化
        super().__init__(
            model_name_or_path=model_name_or_path,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            dtype=dtype,
        )

        print(f"[Variant4] CBM 模型已加载: {model_name_or_path}")
        print(f"[Variant4] 配置:")
        print(f"  - Concept hidden size: {variant4_config.get('concept_hidden_size', 512)}")
        print(f"  - Max BRF dim: {variant4_config.get('max_brf_dim', 40)}")
        print(f"  - Dropout: {variant4_config.get('decoder_dropout', 0.1)}")
        print(f"  - Use concept supervision: {variant4_config.get('use_concept_supervision', True)}")

    def apply_modifications(self):
        """
        应用变体4的 CBM 架构修改
        """
        apply_variant4_modifications(self.model, self.variant4_config)
        print("[Variant4] 已应用 CBM 架构修改")
