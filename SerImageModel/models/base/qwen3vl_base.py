"""
Qwen3VL基础模型封装
提供统一的模型加载和初始化接口
"""

import torch
from typing import Optional
from transformers import Qwen3VLForConditionalGeneration


class Qwen3VLBase:
    """
    Qwen3VL基础模型封装类
    提供统一的接口供变体模型继承
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        attn_implementation: str = "flash_attention_2",
        dtype: Optional[torch.dtype] = None,
        **kwargs
    ):
        """
        初始化基础模型
        
        Args:
            model_name_or_path: 模型路径或HuggingFace模型ID
            cache_dir: 缓存目录
            attn_implementation: 注意力实现方式
            dtype: 模型数据类型
            **kwargs: 其他参数
        """
        self.model_name_or_path = model_name_or_path
        self.cache_dir = cache_dir
        self.attn_implementation = attn_implementation
        self.dtype = dtype
        
        # 加载基础模型
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            dtype=dtype,
            **kwargs
        )
        
        # 应用模型修改的钩子
        self.apply_modifications()
    
    def apply_modifications(self):
        """
        应用模型修改的钩子方法
        子类可以重写此方法来实现特定的架构调整
        """
        pass
    
    def get_model(self):
        """
        获取模型实例
        
        Returns:
            Qwen3VLForConditionalGeneration: 模型实例
        """
        return self.model
    
    def set_trainable_parameters(
        self,
        tune_mm_vision: bool = False,
        tune_mm_mlp: bool = False,
        tune_mm_llm: bool = False
    ):
        """
        设置可训练参数
        
        Args:
            tune_mm_vision: 是否训练视觉编码器
            tune_mm_mlp: 是否训练MLP投影层
            tune_mm_llm: 是否训练语言模型
        """
        if tune_mm_vision:
            for n, p in self.model.visual.named_parameters():
                p.requires_grad = True
        else:
            for n, p in self.model.visual.named_parameters():
                p.requires_grad = False

        if tune_mm_mlp:
            for n, p in self.model.visual.merger.named_parameters():
                p.requires_grad = True
        else:
            for n, p in self.model.visual.merger.named_parameters():
                p.requires_grad = False

        if tune_mm_llm:
            for n, p in self.model.language_model.named_parameters():
                p.requires_grad = True
            self.model.lm_head.requires_grad = True
        else:
            for n, p in self.model.language_model.named_parameters():
                p.requires_grad = False
            self.model.lm_head.requires_grad = False

