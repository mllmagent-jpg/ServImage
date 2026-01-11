"""
Stage2 Model - 基于Stage1权重，使用base模型分类头
- 加载Stage1的checkpoint（base model + LoRA）
- 移除Stage1的CBMQualityDecoder（7维质量评估头）
- 添加base模型的SingleTaskQualityDecoder（二分类头）
- 在Stage1 LoRA基础上再添加新的LoRA
"""

import sys
import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
from transformers import Qwen3VLForConditionalGeneration
from peft import PeftModel

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from .modifications import apply_stage2_modifications, apply_lora_to_stage2, print_trainable_parameters
from .config import get_stage2_config


class Stage2Model(nn.Module):
    """
    Stage2 模型 - 基于Stage1权重，使用base模型分类头

    新的Stage2设计：
    1. 加载Stage1的checkpoint（包含base model + LoRA权重）
    2. 移除Stage1的CBMQualityDecoder（7维质量评估头）
    3. 添加base模型的SingleTaskQualityDecoder（二分类头）
    4. 在Stage1 LoRA基础上再添加新的LoRA（双层LoRA）
    5. 训练新LoRA + 新的quality_decoder
    """

    def __init__(
        self,
        base_model_path: str,
        stage1_checkpoint_path: str,
        stage2_config: dict = None,
        torch_dtype=torch.bfloat16,
        cache_dir: Optional[str] = None,
        attn_implementation: str = "flash_attention_2"
    ):
        """
        初始化 Stage2 模型

        Args:
            base_model_path: 原始 Qwen3VL 模型路径
            stage1_checkpoint_path: Stage1 训练完成的checkpoint路径（包含LoRA）
            stage2_config: Stage2 配置字典
            torch_dtype: 数据类型
            cache_dir: 缓存目录
            attn_implementation: attention 实现方式
        """
        super().__init__()

        if stage2_config is None:
            stage2_config = get_stage2_config()

        self.stage2_config = stage2_config
        self.base_model_path = base_model_path
        self.stage1_checkpoint_path = stage1_checkpoint_path

        print(f"\n[Stage2] ========================================")
        print(f"[Stage2] 初始化Stage2模型（基于Stage1权重）")
        print(f"[Stage2] ========================================")
        print(f"[Stage2] Base模型路径: {base_model_path}")
        print(f"[Stage2] Stage1 checkpoint: {stage1_checkpoint_path}")

        # 加载Stage1模型
        self.model = self._load_stage1_model(
            base_model_path=base_model_path,
            stage1_checkpoint=stage1_checkpoint_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation
        )

        # 应用Stage2修改（移除CBM decoder，添加SingleTask decoder）
        print(f"\n[Stage2] 应用Stage2架构修改...")
        self.model = apply_stage2_modifications(self.model, stage2_config)

        # 应用新的LoRA（在Stage1 LoRA基础上）
        print(f"\n[Stage2] 应用新的LoRA...")
        self.model = apply_lora_to_stage2(self.model, stage2_config)

        # 打印可训练参数
        print_trainable_parameters(self.model)

        print(f"\n[Stage2] ========================================")
        print(f"[Stage2] Stage2模型初始化完成！")
        print(f"[Stage2] ========================================")

    # 代理config属性
    @property
    def config(self):
        return self.model.config

    # 代理forward方法
    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    # 代理必要的方法给底层模型
    def enable_input_require_grads(self):
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def parameters(self, recurse=True):
        return self.model.parameters(recurse=recurse)

    def named_parameters(self, prefix='', recurse=True):
        return self.model.named_parameters(prefix=prefix, recurse=recurse)

    def _load_stage1_model(
        self,
        base_model_path: str,
        stage1_checkpoint: str,
        torch_dtype,
        cache_dir: Optional[str],
        attn_implementation: str
    ):
        """
        加载Stage1模型（包含base model + LoRA）

        Args:
            base_model_path: 基础 Qwen3VL 模型路径
            stage1_checkpoint: Stage1 checkpoint路径
            torch_dtype: 数据类型
            cache_dir: 缓存目录
            attn_implementation: attention 实现方式

        Returns:
            加载完成的模型（包含Stage1的LoRA）
        """
        stage1_path = Path(stage1_checkpoint)

        print(f"\n[Stage2] 加载Stage1模型...")

        # 1. 加载基础 Qwen3VL 模型
        print(f"[Stage2] 加载基础模型: {base_model_path}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
        )

        # 2. 加载 Vision LoRA（如果存在）
        vision_lora_path = stage1_path / "vision_lora"
        if vision_lora_path.exists():
            print(f"[Stage2] 加载 Stage1 Vision LoRA: {vision_lora_path}")
            model.visual = PeftModel.from_pretrained(
                model.visual,
                str(vision_lora_path)
            )
            print(f"[Stage2] ✓ Stage1 Vision LoRA 已加载")
        else:
            print(f"[Stage2] ⚠ 未找到 Stage1 Vision LoRA")

        # 3. 加载 Projector LoRA（如果存在）
        projector_lora_path = stage1_path / "projector_lora"
        if projector_lora_path.exists() and hasattr(model.visual, 'merger'):
            print(f"[Stage2] 加载 Stage1 Projector LoRA: {projector_lora_path}")
            model.visual.merger = PeftModel.from_pretrained(
                model.visual.merger,
                str(projector_lora_path)
            )
            print(f"[Stage2] ✓ Stage1 Projector LoRA 已加载")
        else:
            print(f"[Stage2] ℹ 未找到 Stage1 Projector LoRA")

        # 4. 加载 Language Model LoRA（如果存在）
        llm_lora_path = stage1_path / "llm_lora"
        if llm_lora_path.exists():
            print(f"[Stage2] 加载 Stage1 LLM LoRA: {llm_lora_path}")

            # 添加 dummy prepare_inputs_for_generation 方法
            if not hasattr(model.language_model, "prepare_inputs_for_generation"):
                def _dummy(*args, **kwargs):
                    raise NotImplementedError("Generation not supported")
                model.language_model.prepare_inputs_for_generation = _dummy

            model.language_model = PeftModel.from_pretrained(
                model.language_model,
                str(llm_lora_path)
            )
            print(f"[Stage2] ✓ Stage1 LLM LoRA 已加载")
        else:
            print(f"[Stage2] ⚠ 未找到 Stage1 LLM LoRA")

        # 5. 注意：quality_decoder会在后续步骤中被移除并替换
        quality_decoder_path = stage1_path / "quality_decoder"
        if quality_decoder_path.exists():
            print(f"[Stage2] ℹ 检测到 Stage1 quality_decoder (将在后续步骤中替换)")

        print(f"[Stage2] ✓ Stage1模型加载完成（包含所有LoRA）")

        return model

    def get_model(self):
        """
        获取底层的 Qwen3VLForConditionalGeneration 实例

        Returns:
            transformers模型实例
        """
        return self.model

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        base_model_path: Optional[str] = None,
        stage1_checkpoint_path: Optional[str] = None,
        torch_dtype=torch.bfloat16,
        **kwargs
    ):
        """
        从保存的 Stage2 checkpoint 加载模型

        Args:
            model_path: Stage2 checkpoint 路径
            base_model_path: 基础模型路径
            stage1_checkpoint_path: Stage1 checkpoint路径
            torch_dtype: 数据类型

        Returns:
            加载完成的 Stage2Model 实例
        """
        model_path = Path(model_path)

        # 加载配置
        config_path = model_path / "stage2_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                stage2_config = json.load(f)
        else:
            stage2_config = get_stage2_config()

        # 加载元数据
        metadata_path = model_path / "stage2_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            if base_model_path is None:
                base_model_path = metadata.get('base_model_path')
            if stage1_checkpoint_path is None:
                stage1_checkpoint_path = metadata.get('stage1_checkpoint_path')

        if base_model_path is None or stage1_checkpoint_path is None:
            raise ValueError("需要提供 base_model_path 和 stage1_checkpoint_path")

        # 创建实例
        instance = cls(
            base_model_path=base_model_path,
            stage1_checkpoint_path=stage1_checkpoint_path,
            stage2_config=stage2_config,
            torch_dtype=torch_dtype,
            **kwargs
        )

        # 加载 Stage2 decoder
        decoder_path = model_path / "stage2_decoder" / "pytorch_model.bin"
        if decoder_path.exists():
            print(f"[Stage2] 加载 Stage2 Decoder: {decoder_path}")
            decoder_state = torch.load(decoder_path, map_location='cpu')
            instance.model.quality_decoder.load_state_dict(decoder_state)
            print(f"[Stage2] ✓ Stage2 Decoder 已加载")

        # 加载新的LoRA（如果有）
        # TODO: 实现新LoRA的加载

        return instance

    def save_pretrained(self, output_dir: str):
        """
        保存 Stage2 模型

        保存内容：
        1. Stage2 配置 (stage2_config.json)
        2. Stage2 Decoder 权重 (stage2_decoder/pytorch_model.bin)
        3. Stage2 新LoRA权重（如果有）
        4. 元数据 (stage2_metadata.json)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Stage2] 保存模型到: {output_dir}")

        # 1. 保存配置
        config_path = output_dir / "stage2_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.stage2_config, f, indent=2)
        print(f"[Stage2] ✓ 配置已保存")

        # 2. 保存 Stage2 Decoder
        decoder_dir = output_dir / "stage2_decoder"
        decoder_dir.mkdir(exist_ok=True)
        decoder_state = self.model.quality_decoder.state_dict()
        torch.save(decoder_state, decoder_dir / "pytorch_model.bin")
        print(f"[Stage2] ✓ Stage2 Decoder 已保存")

        # 3. 保存元数据
        metadata = {
            "base_model_path": self.base_model_path,
            "stage1_checkpoint_path": self.stage1_checkpoint_path,
        }
        metadata_path = output_dir / "stage2_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"[Stage2] ✓ 元数据已保存")

        # 4. TODO: 保存新的LoRA权重

        print(f"[Stage2] 模型保存完成！")
