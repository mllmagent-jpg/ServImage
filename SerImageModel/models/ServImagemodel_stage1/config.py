"""
模型变体4的配置定义 - CBM (Concept Bottleneck Model) 版本
针对 train_cbm 数据集的概念瓶颈模型配置
"""

import os


def _get_env_int(key, default):
    """从环境变量获取整数值"""
    val = os.environ.get(key)
    return int(val) if val is not None else default


def _get_env_float(key, default):
    """从环境变量获取浮点值"""
    val = os.environ.get(key)
    return float(val) if val is not None else default


def _get_env_bool(key, default):
    """从环境变量获取布尔值"""
    val = os.environ.get(key)
    if val is None:
        return default
    return val.lower() in ('1', 'true', 'yes', 'on')


# 变体4的默认配置
VARIANT4_CONFIG = {
    # ==================== 模型基础配置 ====================
    'model_name_or_path': 'Qwen/Qwen3-VL-4B-Instruct',

    # ==================== CBM Decoder 配置 ====================
    # 概念瓶颈模型的核心参数（简化版：仅概念预测）
    'concept_hidden_size': _get_env_int('STAGE1_CONCEPT_HIDDEN_SIZE', 512),
    'decoder_dropout': _get_env_float('STAGE1_DECODER_DROPOUT', 0.1),

    # ==================== LoRA 配置 ====================
    # Vision Encoder LoRA
    'lora_vision_enable': _get_env_bool('STAGE1_LORA_VISION_ENABLE', True),
    'lora_vision_r': _get_env_int('STAGE1_LORA_VISION_R', 8),
    'lora_vision_alpha': _get_env_int('STAGE1_LORA_VISION_ALPHA', 16),
    'lora_vision_dropout': _get_env_float('STAGE1_LORA_VISION_DROPOUT', 0.05),

    # Projector (visual.merger) LoRA
    'lora_projector_enable': _get_env_bool('STAGE1_LORA_PROJECTOR_ENABLE', True),
    'lora_projector_r': _get_env_int('STAGE1_LORA_PROJECTOR_R', 8),
    'lora_projector_alpha': _get_env_int('STAGE1_LORA_PROJECTOR_ALPHA', 16),
    'lora_projector_dropout': _get_env_float('STAGE1_LORA_PROJECTOR_DROPOUT', 0.05),

    # Language Model LoRA
    'lora_llm_enable': _get_env_bool('STAGE1_LORA_LLM_ENABLE', True),
    'lora_llm_r': _get_env_int('STAGE1_LORA_LLM_R', 8),
    'lora_llm_alpha': _get_env_int('STAGE1_LORA_LLM_ALPHA', 16),
    'lora_llm_dropout': _get_env_float('STAGE1_LORA_LLM_DROPOUT', 0.05),

    # ==================== 训练配置 ====================
    'learning_rate': 4e-5,             # 与 variant3 保持一致
    'batch_size': 4,
    'gradient_accumulation_steps': 4,
    'num_train_epochs': 50,
    'warmup_ratio': 0.03,
    'weight_decay': 0.05,
    'max_grad_norm': 1.0,
    'lr_scheduler_type': 'cosine',

    # ==================== 损失权重配置 ====================
    'final_loss_weight': 1.0,          # 最终二分类损失权重
    'concept_loss_weight': 0.3,        # 概念监督损失权重

    # ==================== 数据配置 ====================
    'max_pixels': 50176,
    'min_pixels': 784,
    'model_max_length': 8192,

    # ==================== 数据增强配置 ====================
    'shuffle_context_images': False,   # CBM 不打乱图片顺序（保持确定性）

    # ==================== 模型训练策略 ====================
    'freeze_vision_encoder': False,    # 因为有LoRA，所以基础参数冻结
    'freeze_llm': False,               # 因为有LoRA，所以基础参数冻结
    'train_quality_decoder': True,     # CBM Decoder 始终全参数训练

    # ==================== 其他配置 ====================
    'bf16': True,
    'gradient_checkpointing': True,
    'save_strategy': 'steps',
    'save_steps': 100,
    'save_total_limit': 2,
    'logging_steps': 1,
    'eval_strategy': 'no',             # 默认禁用评估
}


def get_variant4_config():
    """
    获取变体4的配置

    Returns:
        dict: 配置字典
    """
    return VARIANT4_CONFIG.copy()


def get_config():
    """
    向后兼容的配置获取函数

    Returns:
        dict: 配置字典
    """
    return get_variant4_config()
