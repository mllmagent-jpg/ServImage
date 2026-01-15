"""
模型变体3的配置定义
序列决策版本：每次评估一个子任务图片
"""

# 变体3的默认配置
VARIANT3_CONFIG = {
    # ==================== 模型基础配置 ====================
    'model_name_or_path': 'Qwen/Qwen3-VL-4B-Instruct',

    # ==================== Quality Decoder 配置 ====================
    # Variant3: 每次只预测一个子任务，不需要max_subtasks
    'decoder_hidden_size': 512,            # Decoder中间层维度
    'decoder_intermediate_size': 2048,     # Decoder第一层维度
    'decoder_dropout': 0.1,                # 添加dropout防止过拟合

    # ==================== LoRA 配置 ====================
    # Vision Encoder LoRA
    'lora_vision_enable': True,
    'lora_vision_r': 8,                    # LoRA rank
    'lora_vision_alpha': 16,               # LoRA alpha = 2 * r
    'lora_vision_dropout': 0.01,

    # Projector (visual.merger) LoRA
    'lora_projector_enable': True,
    'lora_projector_r': 8,                 # LoRA rank
    'lora_projector_alpha': 16,            # LoRA alpha = 2 * r
    'lora_projector_dropout': 0.01,

    # Language Model LoRA
    'lora_llm_enable': True,
    'lora_llm_r': 8,                       # LoRA rank
    'lora_llm_alpha': 16,                  # alpha = 2 * r
    'lora_llm_dropout': 0.01,

    # ==================== 训练配置 ====================
    # 降低学习率以避免过拟合（因为样本间相关性高）
    'learning_rate': 2e-5,                 # 降低学习率 (Variant1: 2e-4)
    'batch_size': 4,
    'gradient_accumulation_steps': 4,
    'num_train_epochs': 3,
    'warmup_ratio': 0.03,
    'weight_decay': 0.05,                  # 增加权重衰减 (Variant1: 0.0)
    'max_grad_norm': 1.0,
    'lr_scheduler_type': 'cosine',

    # ==================== 数据配置 ====================
    'max_pixels': 50176,
    'min_pixels': 784,
    'model_max_length': 8192,

    # ==================== 数据增强配置 ====================
    'shuffle_context_images': True,        # 随机打乱非当前评估的图片顺序

    # ==================== 模型训练策略 ====================
    'freeze_vision_encoder': False,        # 因为有LoRA，所以基础参数冻结
    'freeze_llm': False,                   # 因为有LoRA，所以基础参数冻结
    'train_quality_decoder': True,         # Decoder始终全参数训练

    # ==================== 其他配置 ====================
    'bf16': True,
    'gradient_checkpointing': True,
    'save_strategy': 'steps',
    'save_steps': 500,
    'save_total_limit': 2,
    'logging_steps': 10,
    'eval_strategy': 'steps',
    'eval_steps': 5000,
}


def get_variant3_config():
    """
    获取变体3的配置
    支持通过环境变量覆盖默认LoRA配置

    环境变量:
        STAGE1_LORA_VISION_ENABLE: 0/1
        STAGE1_LORA_PROJECTOR_ENABLE: 0/1
        STAGE1_LORA_LLM_ENABLE: 0/1
        STAGE1_LORA_VISION_R: int
        STAGE1_LORA_VISION_ALPHA: int
        STAGE1_LORA_VISION_DROPOUT: float
        (同样支持 PROJECTOR 和 LLM 的 R/ALPHA/DROPOUT)
        STAGE1_DECODER_DROPOUT: float

    Returns:
        dict: 配置字典
    """
    import os

    config = VARIANT3_CONFIG.copy()

    # 从环境变量读取LoRA配置（如果设置了的话）
    # Vision Encoder LoRA
    if 'STAGE1_LORA_VISION_ENABLE' in os.environ:
        config['lora_vision_enable'] = bool(int(os.environ['STAGE1_LORA_VISION_ENABLE']))
    if 'STAGE1_LORA_VISION_R' in os.environ:
        config['lora_vision_r'] = int(os.environ['STAGE1_LORA_VISION_R'])
    if 'STAGE1_LORA_VISION_ALPHA' in os.environ:
        config['lora_vision_alpha'] = int(os.environ['STAGE1_LORA_VISION_ALPHA'])
    if 'STAGE1_LORA_VISION_DROPOUT' in os.environ:
        config['lora_vision_dropout'] = float(os.environ['STAGE1_LORA_VISION_DROPOUT'])

    # Projector LoRA
    if 'STAGE1_LORA_PROJECTOR_ENABLE' in os.environ:
        config['lora_projector_enable'] = bool(int(os.environ['STAGE1_LORA_PROJECTOR_ENABLE']))
    if 'STAGE1_LORA_PROJECTOR_R' in os.environ:
        config['lora_projector_r'] = int(os.environ['STAGE1_LORA_PROJECTOR_R'])
    if 'STAGE1_LORA_PROJECTOR_ALPHA' in os.environ:
        config['lora_projector_alpha'] = int(os.environ['STAGE1_LORA_PROJECTOR_ALPHA'])
    if 'STAGE1_LORA_PROJECTOR_DROPOUT' in os.environ:
        config['lora_projector_dropout'] = float(os.environ['STAGE1_LORA_PROJECTOR_DROPOUT'])

    # Language Model LoRA
    if 'STAGE1_LORA_LLM_ENABLE' in os.environ:
        config['lora_llm_enable'] = bool(int(os.environ['STAGE1_LORA_LLM_ENABLE']))
    if 'STAGE1_LORA_LLM_R' in os.environ:
        config['lora_llm_r'] = int(os.environ['STAGE1_LORA_LLM_R'])
    if 'STAGE1_LORA_LLM_ALPHA' in os.environ:
        config['lora_llm_alpha'] = int(os.environ['STAGE1_LORA_LLM_ALPHA'])
    if 'STAGE1_LORA_LLM_DROPOUT' in os.environ:
        config['lora_llm_dropout'] = float(os.environ['STAGE1_LORA_LLM_DROPOUT'])

    # Decoder配置
    if 'STAGE1_DECODER_DROPOUT' in os.environ:
        config['decoder_dropout'] = float(os.environ['STAGE1_DECODER_DROPOUT'])

    return config


def get_config():
    """
    向后兼容的配置获取函数

    Returns:
        dict: 配置字典
    """
    return get_variant3_config()
