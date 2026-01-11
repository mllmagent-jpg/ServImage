# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
Variant3 训练脚本 - 序列决策版本
每次评估一个子任务图片
"""

import os
import logging
import pathlib
import torch
import gc
import transformers
import sys
from pathlib import Path
from typing import Optional

# ==================== 在导入 DeepSpeed 之前禁用数据一致性检查 ====================
os.environ['DEEPSPEED_CHECK_DATALOADER_INPUTS'] = '0'

try:
    import deepspeed.runtime.engine as ds_engine_module
    if hasattr(ds_engine_module, 'DeepSpeedEngine'):
        original_check = getattr(ds_engine_module.DeepSpeedEngine, 'check_dataloader_inputs_same_across_ranks', None)
        if original_check:
            def dummy_check_dataloader_inputs_same_across_ranks(self, *args, **kwargs):
                return
            ds_engine_module.DeepSpeedEngine.check_dataloader_inputs_same_across_ranks = dummy_check_dataloader_inputs_same_across_ranks

        original_broadcast = getattr(ds_engine_module.DeepSpeedEngine, 'broadcast_and_check', None)
        if original_broadcast:
            def dummy_broadcast_and_check(self, *args, **kwargs):
                return
            ds_engine_module.DeepSpeedEngine.broadcast_and_check = dummy_broadcast_and_check
except (ImportError, AttributeError):
    pass

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 尝试导入 flash_attn
try:
    from SerImageModel.train.trainer import replace_qwen2_vl_attention_class
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("[Warning] flash_attn 不可用，将使用标准attention实现")

from SerImageModel.models.ServImagemodel_base.model import Variant3Model
from SerImageModel.models.ServImagemodel_base.config import get_variant3_config
from SerImageModel.models.ServImagemodel_base.modifications import (
    apply_variant3_modifications,
    apply_lora_to_variant3,
    print_trainable_parameters
)
from SerImageModel.data.dataset import QualityEvalDatasetUnified
from SerImageModel.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor, Trainer, EvalPrediction

# 导入 SwanLab
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("[Warning] swanlab 不可用")

local_rank = None


class SwanLabCallback(transformers.TrainerCallback):
    """SwanLab 训练回调"""

    def __init__(self):
        self.swanlab_available = SWANLAB_AVAILABLE

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.swanlab_available:
            return

        if state.is_world_process_zero and logs:
            filtered_logs = {}
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    filtered_logs[key] = value

            if filtered_logs:
                swanlab.log(filtered_logs, step=state.global_step)

# todo 这里的验证改为全验证数据进行验证删除这里的采样验证，直接进行全部的验证数据集数据进行验证
class DynamicEvalSamplingCallback(transformers.TrainerCallback):
    """
    动态验证集采样回调

    在每次验证前，根据 global_step 动态采样验证集子集
    种子公式：base_seed + (global_step // eval_steps)
    """

    def __init__(self,
                 eval_dataset_full,
                 max_eval_samples: int,
                 base_seed: int = 42,
                 eval_steps: int = 5000):
        """
        Args:
            eval_dataset_full: 完整验证集（包含所有samples）
            max_eval_samples: 每次采样的样本数量
            base_seed: 基础随机种子
            eval_steps: 评估间隔步数
        """
        self.eval_dataset_full = eval_dataset_full
        self.max_eval_samples = max_eval_samples
        self.base_seed = base_seed
        self.eval_steps = eval_steps
        self.full_samples = eval_dataset_full.samples.copy()  # 备份完整样本列表
        self.last_sampled_step = -1

    def on_evaluate(self, args, state, control, **kwargs):
        """
        在每次验证开始前，动态采样验证集
        """
        import random
        import torch
        import gc

        global_step = state.global_step

        # 在评估前清理GPU缓存，降低OOM风险
        if state.is_world_process_zero:
            rank0_print(f"\n[MemoryCleanup] 评估前清理GPU缓存和梯度...")

        # 获取模型并清理梯度（零风险：验证前的梯度已经被使用过）
        model = kwargs.get('model')
        if model is not None:
            model.zero_grad(set_to_none=True)
            if state.is_world_process_zero:
                rank0_print(f"  - 已清理梯度 (set_to_none=True)")

        torch.cuda.empty_cache()
        gc.collect()

        # 同步所有进程
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # 计算当前验证的种子（基于global_step）
        eval_iteration = global_step // self.eval_steps
        current_seed = self.base_seed + eval_iteration

        # 只在新的验证步骤时重新采样（避免在同一个eval中多次采样）
        if global_step != self.last_sampled_step:
            # 使用动态种子进行采样
            random.seed(current_seed)
            sampled_indices = random.sample(
                range(len(self.full_samples)),
                min(self.max_eval_samples, len(self.full_samples))
            )

            # 更新验证集的样本列表
            self.eval_dataset_full.samples = [self.full_samples[i] for i in sorted(sampled_indices)]

            # 记录采样信息
            if state.is_world_process_zero:
                rank0_print(f"\n[DynamicSampling] Step {global_step}:")
                rank0_print(f"  - Eval iteration: {eval_iteration}")
                rank0_print(f"  - Seed: {current_seed}")
                rank0_print(f"  - Sampled {len(sampled_indices)} samples from {len(self.full_samples)}")
                rank0_print(f"  - Sample indices range: [{min(sampled_indices)}, {max(sampled_indices)}]\n")

            self.last_sampled_step = global_step

            # 采样完成后再次清理，确保评估时内存状态最优
            torch.cuda.empty_cache()


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """
    只保存可训练参数（LoRA + Quality Decoder）

    Variant3 架构与Variant1类似，但decoder更简单
    """

    if trainer.deepspeed:
        torch.cuda.synchronize()
        # DeepSpeed: 使用engine的save_checkpoint方法
        trainer.deepspeed.save_checkpoint(output_dir)
        return

    if not trainer.args.should_save:
        return

    rank0_print(f"\n[Variant3] 开始保存模型到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # 1. 保存 PEFT (LoRA) 权重
    vision_lora_dir = os.path.join(output_dir, "vision_lora")
    os.makedirs(vision_lora_dir, exist_ok=True)

    try:
        trainer.model.visual.save_pretrained(vision_lora_dir)
        rank0_print(f"[Variant3] ✓ Vision LoRA 已保存到: {vision_lora_dir}")
    except Exception as e:
        rank0_print(f"[Variant3] ⚠ Vision LoRA 保存失败: {e}")

    # 可选：保存 LLM LoRA
    llm_lora_dir = os.path.join(output_dir, "llm_lora")
    if hasattr(trainer.model, "language_model") and hasattr(trainer.model.language_model, "save_pretrained"):
        os.makedirs(llm_lora_dir, exist_ok=True)
        try:
            trainer.model.language_model.save_pretrained(llm_lora_dir)
            rank0_print(f"[Variant3] ✓ LLM LoRA 已保存到: {llm_lora_dir}")
        except Exception as e:
            rank0_print(f"[Variant3] ⚠ LLM LoRA 保存失败: {e}")

    # 2. 保存 Quality Decoder 权重
    quality_decoder_dir = os.path.join(output_dir, "quality_decoder")
    os.makedirs(quality_decoder_dir, exist_ok=True)

    quality_decoder_state = {}
    for name, param in trainer.model.named_parameters():
        if 'quality_decoder' in name:
            clean_name = name.replace('model.', '') if name.startswith('model.') else name
            quality_decoder_state[clean_name] = param.cpu()

    if quality_decoder_state:
        quality_decoder_path = os.path.join(quality_decoder_dir, "pytorch_model.bin")
        torch.save(quality_decoder_state, quality_decoder_path)
        rank0_print(f"[Variant3] ✓ Quality Decoder 已保存: {quality_decoder_path}")
        rank0_print(f"[Variant3]   参数数量: {len(quality_decoder_state)}")

    # 3. 保存模型配置
    trainer.model.config.save_pretrained(output_dir)
    rank0_print(f"[Variant3] ✓ 模型配置已保存")

    # 4. 保存 Variant3 配置
    variant3_config_path = os.path.join(output_dir, "variant3_config.json")
    import json
    variant3_cfg = get_variant3_config()
    with open(variant3_config_path, 'w') as f:
        json.dump(variant3_cfg, f, indent=2)
    rank0_print(f"[Variant3] ✓ Variant3 配置已保存: {variant3_config_path}")

    rank0_print(f"\n[Variant3] 模型保存完成！")


class Variant3Trainer(Trainer):
    """
    自定义 Trainer for Variant3
    """

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir

        safe_save_model_for_hf_trainer(self, output_dir)

        if hasattr(self, 'processing_class') and self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

    def preprocess_logits_for_metrics(self, logits, labels):
        """
        在评估时预处理logits，只保留必要信息，丢弃hidden_states等中间结果
        这样可以大幅减少评估时的显存占用

        关键优化：
        1. 只保留logits（不要完整的ModelOutput）
        2. 移到CPU避免占用GPU内存
        3. detach()去除梯度信息

        Args:
            logits: 模型输出，可能是 ModelOutput/tuple/tensor
            labels: 标签

        Returns:
            logits: (batch_size, 2) tensor，在CPU上
        """
        # 1. 处理 ModelOutput 对象（如 CausalLMOutputWithPast）
        # 这一步很关键：丢弃hidden_states等大对象，只保留logits
        if hasattr(logits, 'logits'):
            logits = logits.logits

        # 2. 处理 tuple/list（兼容旧版本）
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        # 3. 关键优化：移到CPU并去除梯度
        # - detach(): 去除梯度信息，避免保留计算图
        # - cpu(): 移到CPU，释放GPU内存（评估时不需要在GPU上累积）
        # - float(): 确保数据类型一致性
        return logits.detach().cpu().float()


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ==================== 设置随机种子 ====================
    import random
    import numpy as np

    seed = training_args.seed if hasattr(training_args, 'seed') and training_args.seed is not None else 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    rank0_print(f"[Variant3] 已设置随机种子: {seed}")

    # 分布式环境检查
    required_env = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    if not all(v in os.environ for v in required_env):
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", str(training_args.local_rank if training_args.local_rank >= 0 else 0))
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        print("[Variant3] 未检测到分布式环境变量，已设置单机默认值")

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # ==================== 初始化 SwanLab ====================
    if SWANLAB_AVAILABLE and (local_rank == 0 or local_rank == -1):
        rank0_print("\n[Variant3] 初始化 SwanLab...")
        swanlab.init(
            project="qwen3vl-variant3-quality-eval",
            experiment_name=training_args.run_name or "variant3-training",
            description="序列决策版本 - 每次评估一个子任务图片",
            mode="local",
            config={
                "model": model_args.model_name_or_path,
                "learning_rate": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "num_epochs": training_args.num_train_epochs,
                "lora_vision_r": 8,
                "lora_vision_alpha": 16,
                "lora_llm_r": 8,
                "lora_llm_alpha": 16,
                "bf16": training_args.bf16,
                "gradient_checkpointing": training_args.gradient_checkpointing,
            }
        )
        rank0_print("[Variant3] SwanLab 已初始化")

    # ==================== 1. 加载 Variant3 配置 ====================
    variant3_config = get_variant3_config()
    if os.getenv("VARIANT3_DISABLE_LORA", "").lower() in ("1", "true", "yes"):
        variant3_config["lora_vision_enable"] = False
        variant3_config["lora_projector_enable"] = False
        variant3_config["lora_llm_enable"] = False
        rank0_print("[Variant3] VARIANT3_DISABLE_LORA=1: 关闭所有LoRA，仅训练decoder")

    rank0_print("\n" + "="*80)
    rank0_print("[Variant3] 加载配置:")
    rank0_print(f"  - 模型路径: {model_args.model_name_or_path}")
    rank0_print(f"  - Vision LoRA: r={variant3_config['lora_vision_r']}, alpha={variant3_config['lora_vision_alpha']}")
    rank0_print(f"  - LLM LoRA: r={variant3_config['lora_llm_r']}, alpha={variant3_config['lora_llm_alpha']}")
    rank0_print(f"  - Decoder dropout: {variant3_config['decoder_dropout']}")
    rank0_print(f"  - Shuffle context: {variant3_config['shuffle_context_images']}")
    rank0_print("="*80 + "\n")

    # ==================== 2. 加载基础模型 ====================
    rank0_print("[Variant3] 加载基础模型...")

    variant3_model = Variant3Model(
        model_name_or_path=model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        dtype=(torch.bfloat16 if training_args.bf16 else None),
        variant3_config=variant3_config,
    )
    model = variant3_model.get_model()
    data_args.model_type = "qwen3vl"

    rank0_print(f'[Variant3] 模型类: {model.__class__.__name__}')

    # ==================== 禁用 DeepSpeed 数据一致性检查 ====================
    if training_args.deepspeed:
        os.environ['DEEPSPEED_CHECK_DATALOADER_INPUTS'] = '0'
        rank0_print("[Variant3] 已设置环境变量以禁用 DeepSpeed 数据一致性检查")

    # ==================== 4. 应用 LoRA ====================
    rank0_print("\n[Variant3] 配置 LoRA...")
    model = apply_lora_to_variant3(model, variant3_config)

    # ==================== 5. 打印可训练参数 ====================
    if local_rank == 0 or not torch.distributed.is_initialized():
        print_trainable_parameters(model)

    # ==================== 6. 加载 Processor ====================
    rank0_print("\n[Variant3] 加载 Processor...")
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

    # ==================== 7. 配置模型 ====================
    if data_args.data_flatten or data_args.data_packing:
        if FLASH_ATTN_AVAILABLE:
            replace_qwen2_vl_attention_class()
        else:
            rank0_print("[Warning] data_flatten/data_packing 需要 flash_attn，已跳过")
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # ==================== 8. 加载质量评估数据集（V3版本） ====================
    rank0_print("\n[Variant3] 加载质量评估数据集（序列决策版本）...")

    # 使用命令行参数传入的数据路径
    if data_args.train_data is None:
        raise ValueError("请通过 --train_data 参数指定训练数据路径")

    train_jsonl_path = data_args.train_data
    val_jsonl_path = data_args.val_data if data_args.val_data else None

    # 使用 data_args 中的目录路径
    bench_dir = data_args.bench_dir
    dataset_dir = data_args.dataset_dir

    rank0_print(f"[Variant3] 数据路径:")
    rank0_print(f"  - 训练集: {train_jsonl_path}")
    rank0_print(f"  - 验证集: {val_jsonl_path}")
    rank0_print(f"  - Bench目录: {bench_dir}")
    rank0_print(f"  - Dataset目录: {dataset_dir}")

    train_dataset = QualityEvalDatasetUnified(
        train_jsonl_path=train_jsonl_path,
        bench_dir=bench_dir,
        dataset_dir=dataset_dir,
        task_mode="base",  # Base 模型：独立的二分类任务
        processor=processor
    )

    rank0_print(f"[Variant3] 训练集大小: {len(train_dataset)}")

    # 加载验证数据集
    eval_dataset = None
    eval_dataset_full = None  # 保存完整验证集
    if val_jsonl_path and Path(val_jsonl_path).exists():
        eval_dataset = QualityEvalDatasetUnified(
            train_jsonl_path=val_jsonl_path,
            bench_dir=bench_dir,
            dataset_dir=dataset_dir,
            task_mode="base",  # Base 模型：独立的二分类任务
            processor=processor
        )
        rank0_print(f"[Variant3] 验证集大小: {len(eval_dataset)}")

        # 如果设置了 max_eval_samples，准备动态采样
        if data_args.max_eval_samples is not None and data_args.max_eval_samples < len(eval_dataset):
            rank0_print(f"[Variant3] 启用动态随机采样策略:")
            rank0_print(f"  - 验证集总大小: {len(eval_dataset)}")
            rank0_print(f"  - 每次采样: {data_args.max_eval_samples} 个样本")
            rank0_print(f"  - 采样策略: 基于 global_step 的伪随机（可复现）")
            rank0_print(f"  - 种子公式: base_seed (42) + (global_step // eval_steps)")

            # 保存完整验证集供动态采样使用
            eval_dataset_full = eval_dataset

            # 创建一个初始采样的数据集副本（重要：这个副本将传给Trainer）
            import random
            import copy
            random.seed(seed)  # 使用训练的随机种子
            initial_indices = random.sample(range(len(eval_dataset.samples)), data_args.max_eval_samples)

            # 创建新的数据集实例，只包含采样的样本
            eval_dataset = copy.copy(eval_dataset_full)  # 浅拷贝
            eval_dataset.samples = [eval_dataset_full.samples[i] for i in sorted(initial_indices)]

            rank0_print(f"[Variant3] 已创建初始采样数据集副本: {len(eval_dataset)} 个样本")
            # 注意：DynamicSamplingCallback 会在每次验证前更新 eval_dataset.samples
    else:
        rank0_print(f"[Variant3] 警告: 验证集文件不存在或未指定，跳过评估")

    # ==================== 8.5. 定义评估指标 ====================
    def compute_metrics(eval_pred: EvalPrediction):
        """
        计算验证集准确率（Variant3版本）
        - logits 形状: (batch, 2)
        - labels 形状: (batch,)
        """
        predictions = eval_pred.predictions if hasattr(eval_pred, "predictions") else eval_pred[0]
        label_ids = eval_pred.label_ids if hasattr(eval_pred, "label_ids") else eval_pred[1]

        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        if isinstance(label_ids, (list, tuple)):
            label_ids = label_ids[0]

        preds_tensor = torch.tensor(predictions)
        labels_tensor = torch.tensor(label_ids)

        # 简化版：直接计算准确率
        pred_classes = preds_tensor.argmax(dim=-1)
        correct = (pred_classes == labels_tensor)
        accuracy = correct.float().mean().item()

        return {"accuracy": accuracy}

    # ==================== 9. 创建 Trainer ====================
    from SerImageModel.data.dataset import QualityEvalCollatorUnified

    quality_collator = QualityEvalCollatorUnified(processor=processor, task_mode="base")

    data_module = {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": quality_collator
    }

    # 创建回调列表
    callbacks = [SwanLabCallback()] if SWANLAB_AVAILABLE else []

    # 添加动态验证集采样回调（如果启用）
    if eval_dataset_full is not None and data_args.max_eval_samples is not None:
        dynamic_sampling_callback = DynamicEvalSamplingCallback(
            eval_dataset_full=eval_dataset_full,
            max_eval_samples=data_args.max_eval_samples,
            base_seed=seed,  # 使用训练的随机种子作为基础种子
            eval_steps=training_args.eval_steps
        )
        callbacks.append(dynamic_sampling_callback)
        rank0_print(f"[Variant3] 已启用动态验证集采样回调")

    trainer = Variant3Trainer(
        model=model,
        args=training_args,
        callbacks=callbacks,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
        **data_module
    )

    # 关键：让回调更新Trainer实际使用的eval_dataset（而不是初始的引用）
    if eval_dataset_full is not None and data_args.max_eval_samples is not None:
        for callback in callbacks:
            if isinstance(callback, DynamicEvalSamplingCallback):
                # 将回调的eval_dataset_full引用改为Trainer实际使用的数据集
                # 这样回调修改samples时，会影响Trainer的DataLoader重新创建
                callback.eval_dataset_full = trainer.eval_dataset
                rank0_print(f"[Variant3] 已将Trainer的评估数据集引用传递给动态采样回调")
                break

    # ==================== 11. 开始训练 ====================
    rank0_print("\n[Variant3] 开始训练...")
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("[Variant3] 发现检查点，恢复训练")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    # ==================== 11. 保存模型 ====================
    model.config.use_cache = True
    rank0_print(f"\n[Variant3] 保存模型到: {training_args.output_dir}")
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    # ==================== 12. 测试集评估 ====================
    test_jsonl_path = data_args.test_data if hasattr(data_args, 'test_data') else None
    if test_jsonl_path and Path(test_jsonl_path).exists():
        rank0_print(f"\n[Variant3] 开始测试集评估...")
        rank0_print(f"  - 测试数据: {test_jsonl_path}")

        try:
            # 加载测试数据集
            test_dataset = QualityEvalDatasetUnified(
                train_jsonl_path=test_jsonl_path,
                bench_dir=bench_dir,
                dataset_dir=dataset_dir,
                task_mode="base",  # Base 模型：独立的二分类任务
                processor=processor
            )
            rank0_print(f"[Variant3] 测试集大小: {len(test_dataset)}")

            # 在测试前清理GPU缓存，释放训练遗留的内存
            rank0_print(f"[MemoryCleanup] 测试前清理GPU缓存...")
            torch.cuda.empty_cache()
            gc.collect()

            # 同步所有进程
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            rank0_print(f"[Variant3] 开始评估全部 {len(test_dataset)} 个测试样本...")

            # 运行测试集评估
            test_results = trainer.evaluate(
                eval_dataset=test_dataset,
                metric_key_prefix="test"
            )

            rank0_print(f"\n[Variant3] 测试集结果:")
            rank0_print(f"  - 测试准确率: {test_results.get('test_accuracy', 0):.2%}")
            rank0_print(f"  - 测试Loss: {test_results.get('test_loss', 0):.4f}")
            rank0_print(f"  - 评估样本数: {len(test_dataset)}")

        except Exception as e:
            rank0_print(f"[Variant3] 测试集评估失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        rank0_print(f"\n[Variant3] 未找到测试集，跳过测试评估")

    rank0_print("\n[Variant3] 训练完成！")

    # ==================== 13. 完成 SwanLab 记录 ====================
    if SWANLAB_AVAILABLE and (local_rank == 0 or local_rank == -1):
        swanlab.finish()
        rank0_print("[Variant3] SwanLab 记录已完成")


if __name__ == "__main__":
    attn_impl = "flash_attention_2" if FLASH_ATTN_AVAILABLE else "eager"
    print(f"[Variant3] 使用 attention 实现: {attn_impl}")
    train(attn_implementation=attn_impl)
