"""
模型变体3：序列决策版本
每次评估一个子任务图片，使用单头decoder
"""

import sys
import os
import json
import torch
from pathlib import Path
from typing import Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from SerImageModel.models.base.qwen3vl_base import Qwen3VLBase
from .modifications import apply_variant3_modifications


class Variant3Model(Qwen3VLBase):
    """
    模型变体3 - 序列决策版本
    相比Variant1，每次只预测一个子任务的质量
    """

    def __init__(self, *args, variant3_config=None, **kwargs):
        """
        初始化变体3模型

        Args:
            variant3_config: Variant3 配置字典
        """
        self.variant3_config = variant3_config or {}
        super().__init__(*args, **kwargs)
        print(f"[Variant3] 模型已加载: {self.model_name_or_path}")

    def apply_modifications(self):
        """
        应用变体3的特定修改
        """
        apply_variant3_modifications(self.model, self.variant3_config)
        print("[Variant3] 已应用变体3的架构修改")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        base_model_path: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: str = "auto",
        **kwargs
    ):
        """
        从保存的 Variant3 checkpoint 加载模型

        Args:
            model_path: Variant3 checkpoint 路径
            base_model_path: Qwen3VL 基础模型路径
            torch_dtype: 数据类型
            device_map: 设备映射
            **kwargs: 其他参数

        Returns:
            加载完成的 Variant3Model 实例
        """
        from transformers import Qwen3VLForConditionalGeneration
        from peft import PeftModel

        model_path = Path(model_path)

        # 1. 加载 Variant3 配置
        variant3_config_path = model_path / "variant3_config.json"
        if variant3_config_path.exists():
            with open(variant3_config_path, 'r') as f:
                variant3_config = json.load(f)
            print(f"[Variant3] 加载配置: {variant3_config_path}")
        else:
            print("[Variant3] 未找到 variant3_config.json，使用默认配置")
            from .config import get_variant3_config
            variant3_config = get_variant3_config()

        # 2. 确定基础模型路径
        if base_model_path is None:
            base_model_path = variant3_config.get('model_name_or_path', 'Qwen/Qwen3-VL-4B-Instruct')
            print(f"[Variant3] 使用配置中的基础模型: {base_model_path}")

        # 3. 加载基础模型
        print(f"[Variant3] 加载基础模型: {base_model_path}")
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype or torch.bfloat16,
            device_map=device_map,
            **kwargs
        )

        # 4. 应用 Variant3 架构修改
        print("[Variant3] 应用架构修改...")
        base_model = apply_variant3_modifications(base_model, variant3_config)

        # 5. 加载 Vision LoRA
        vision_lora_dir = model_path / "vision_lora"
        if vision_lora_dir.exists():
            print(f"[Variant3] 加载 Vision LoRA: {vision_lora_dir}")
            base_model.visual = PeftModel.from_pretrained(
                base_model.visual,
                str(vision_lora_dir)
            )
            print("[Variant3] ✓ Vision LoRA 已加载")
        else:
            print(f"[Variant3] ⚠ 未找到 Vision LoRA: {vision_lora_dir}")

        # 6. 加载 LLM LoRA
        llm_lora_dir = model_path / "llm_lora"
        if llm_lora_dir.exists():
            print(f"[Variant3] 加载 LLM LoRA: {llm_lora_dir}")

            # 添加 dummy prepare_inputs_for_generation 方法（与训练时相同）
            if not hasattr(base_model.language_model, "prepare_inputs_for_generation"):
                def _dummy_prepare_inputs_for_generation(*args, **kwargs):
                    raise NotImplementedError("Generation is not supported for Variant3 model.")
                base_model.language_model.prepare_inputs_for_generation = _dummy_prepare_inputs_for_generation

            base_model.language_model = PeftModel.from_pretrained(
                base_model.language_model,
                str(llm_lora_dir)
            )
            print("[Variant3] ✓ LLM LoRA 已加载")
        else:
            print(f"[Variant3] ⚠ 未找到 LLM LoRA: {llm_lora_dir}")

        # 7. 加载 Quality Decoder
        quality_decoder_path = model_path / "quality_decoder" / "pytorch_model.bin"
        if quality_decoder_path.exists():
            print(f"[Variant3] 加载 Quality Decoder: {quality_decoder_path}")
            quality_state = torch.load(quality_decoder_path, map_location='cpu')

            base_model.quality_decoder.load_state_dict(quality_state, strict=False)
            print(f"[Variant3] ✓ Quality Decoder 已加载（{len(quality_state)} 个参数）")
        else:
            print(f"[Variant3] ⚠ 未找到 Quality Decoder: {quality_decoder_path}")

        # 7. 创建 Variant3Model 实例包装
        instance = cls.__new__(cls)
        instance.model = base_model
        instance.variant3_config = variant3_config
        instance.model_name_or_path = str(model_path)

        print(f"[Variant3] 模型加载完成: {model_path}")
        return instance

    def get_model(self):
        """
        获取底层的 Qwen3VLForConditionalGeneration 实例
        """
        return self.model
