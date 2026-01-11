#!/bin/bash

set -euo pipefail

# 自动切换到脚本所在项目的根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"
echo "工作目录已切换到: ${PROJECT_ROOT}"
echo ""

# 激活所需的 Conda 环境
set +u
CONDA_ROOT=${CONDA_ROOT:-"${HOME}/miniconda3"}
if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate qwen-vl
else
  echo "[WARNING] 未找到 conda.sh，请确保已激活正确的 Python 环境"
fi
set -u

# ==============================
# Base 模型训练配置（单卡版本）
# 说明：独立的二分类基础模型
#       - Accept/Reject 预测（二分类，0/1标签）
#       - 使用图片压缩策略减少视觉token数量
#       - 当前结果图：1张，原样输入
#       - 输入图：最多1张，原样输入
#       - 其它结果图：压缩成最多2张拼贴图（动态网格）
#       - 参考图：压缩成最多2张拼贴图（动态网格）
#       - 总图片数最多6张
# ==============================
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${NNODES:-1}

# 强制只使用 GPU 1（必须在脚本开始时export）
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"1"}

# 自动推导可用 GPU 数
IFS=',' read -ra GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=${#GPU_LIST[@]}

if [[ "${NUM_GPUS}" -ne 1 ]]; then
  echo "[ERROR] 该脚本仅用于 1 卡 (总显存约 48GB)，当前检测到 ${NUM_GPUS} 卡: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi

# DeepSpeed通信配置
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export DEEPSPEED_CHECK_DATALOADER_INPUTS=0
export VARIANT3_DISABLE_LORA=${VARIANT3_DISABLE_LORA:-1}

# PyTorch 内存分配优化（解决碎片化问题）
# 启用可扩展内存段以避免大块内存分配失败
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# DeepSpeed 配置文件（单卡也可以使用DeepSpeed的内存优化）
deepspeed_config=${DEEPSPEED_CONFIG:-scripts/deepspeed_tp2.json}

# ==============================
# 训练与数据配置（Base模型 - 单卡）
# ==============================
llm=${MODEL_PATH:-"weights/Qwen3-VL-4B-Thinking"}

# Base 模型特定配置（单卡可能需要调整batch_size和grad_accum_steps）
lr=${LR:-4e-5}
batch_size=${BATCH_SIZE:-2} # 2
grad_accum_steps=${GRAD_ACCUM_STEPS:-8}  # 单卡增加梯度累积以保持等效batch size # 16
num_epochs=${NUM_EPOCHS:-2}
weight_decay=${WEIGHT_DECAY:-0.01}

save_steps=${SAVE_STEPS:-500}  # 定期保存检查点（与eval_steps对齐）
eval_strategy=${EVAL_STRATEGY:-"steps"}  # 启用评估
eval_steps=${EVAL_STEPS:-500}  # 每多少步评估一次（减少评估频率，降低显存峰值）
max_eval_samples=${MAX_EVAL_SAMPLES:-10000}  # 验证集采样数量：每次评估2000个样本（约36%的数据，已添加显存优化）

# 数据路径配置（相对于项目根目录）
bench_dir=${BENCH_DIR:-"${PROJECT_ROOT}/data/ServImage_Bench"}
dataset_dir=${DATASET_DIR:-"${PROJECT_ROOT}/data/ServImage_Dataset"}

# Base 数据集路径（standard_split - 包含0/1标签）
base_data_dir="${PROJECT_ROOT}/data/Label_data/standard_split"
train_data=${TRAIN_DATA:-"${base_data_dir}/train.jsonl"}
val_data=${VAL_DATA:-"${base_data_dir}/val.jsonl"}
test_data=${TEST_DATA:-"${base_data_dir}/test.jsonl"}

# 训练脚本
entry_file=${ENTRY_FILE:-"train/train_base.py"}
run_name="Qwen-4B-Think-non-$(date +%Y%m%d_%H%M%S)"
output_dir=${OUTPUT_DIR:-"./output/${run_name}"}

echo "========================================="
echo "Base 模型训练（独立二分类模型 - 单卡版本）"
echo "========================================="
echo "模型: ${llm}"
echo "学习率: ${lr}"
echo "权重衰减: ${weight_decay}"
echo "Train batch/GPU: ${batch_size}"
echo "Gradient Accum: ${grad_accum_steps}"
echo "有效 Batch Size: $((batch_size * grad_accum_steps * NUM_GPUS))"
echo "Epochs: ${num_epochs}"
echo "GPU数量: ${NUM_GPUS} (单卡)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "DeepSpeed 配置: ${deepspeed_config}"
echo "输出目录: ${output_dir}"
echo ""
echo "Base 模型特性:"
echo "  - 任务模式: base (独立的Accept/Reject二分类)"
echo "  - 输出: 0/1 标签"
echo "  - 图片压缩策略：减少视觉token数量"
echo "  - 当前结果图：1张原图（不压缩）"
echo "  - 输入图：最多1张原图（不压缩）"
echo "  - 其它结果图：压缩为最多2张拼贴图（动态网格）"
echo "  - 参考图：压缩为最多2张拼贴图（动态网格）"
echo "  - 总图片数：最多6张"
echo "  - 确定性策略：不随机打乱图片顺序"
echo "  - 二分类器：简单的Accept/Reject预测"
echo ""
echo "数据配置:"
echo "  - Bench 目录: ${bench_dir}"
echo "  - Dataset 目录: ${dataset_dir}"
echo "  - Train 数据: ${train_data}"
echo "  - Val 数据: ${val_data}"
echo "  - Test 数据: ${test_data}"
echo ""
echo "评估配置:"
echo "  - 策略: ${eval_strategy}"
echo "  - 评估步数: ${eval_steps}"
echo "  - 验证集采样数: ${max_eval_samples} (全部数据)"
echo "  - 保存步数: ${save_steps}"
echo ""
echo "监控与日志:"
echo "  - SwanLab: 已集成（自动初始化）"
echo "  - 项目名: qwen3vl-base-quality-eval"
echo "  - 训练进度条: 已启用（disable_tqdm=False）"
echo "  - 检查点保存: 每${save_steps}步保存一次 (最多保留2个)"
echo ""
echo "单卡训练提示:"
echo "  - 当前梯度累积步数: ${grad_accum_steps} (相比双卡版本翻倍)"
echo "  - 如遇显存不足，可降低 batch_size 或增加 grad_accum_steps"
echo "  - 可通过环境变量调整: BATCH_SIZE=1 GRAD_ACCUM_STEPS=32"
echo "========================================="

args="
    --model_name_or_path ${llm} \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs ${num_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --eval_strategy ${eval_strategy} \
    --eval_steps ${eval_steps} \
    --save_strategy steps \
    --save_steps ${save_steps} \
    --save_total_limit 2 \
    --learning_rate ${lr} \
    --weight_decay ${weight_decay} \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --logging_first_step True \
    --ddp_find_unused_parameters False \
    --disable_tqdm False \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --run_name ${run_name} \
    --report_to none \
    --deepspeed ${deepspeed_config} \
    --data_flatten False \
    --data_packing False \
    --max_pixels 50176 \
    --min_pixels 784 \
    --remove_unused_columns False \
    --train_data ${train_data} \
    --val_data ${val_data} \
    --test_data ${test_data} \
    --bench_dir ${bench_dir} \
    --dataset_dir ${dataset_dir} \
    --max_eval_samples ${max_eval_samples} \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --greater_is_better True"

echo ""
echo "使用 DeepSpeed 启动：1 卡 ZeRO-2（单卡内存优化）"
echo "物理 GPU: ${CUDA_VISIBLE_DEVICES} (进程内重新编号为 GPU 1)"
echo ""

# 使用 DeepSpeed 启动
# 注意：使用 --include localhost:1 明确指定物理 GPU 1
# 不使用 --num_gpus 参数，避免 DeepSpeed 忽略 CUDA_VISIBLE_DEVICES 设置
deepspeed --master_addr=${MASTER_ADDR} \
          --master_port=${MASTER_PORT} \
          --include localhost:1 \
          ${entry_file} ${args}

echo ""
echo "训练完成，输出目录: ${output_dir}"
echo ""
echo "Base 模型训练说明（单卡版本）："
echo "  - 任务类型：独立的二分类基础模型"
echo "  - 输出：Accept (1) / Reject (0)"
echo "  - 图片压缩：其它结果图和参考图使用动态网格拼贴"
echo "  - 显存优化：使用图片压缩策略 + DeepSpeed ZeRO-2"
echo "  - 单卡配置：batch_size=${batch_size}, grad_accum=${grad_accum_steps}"
echo "  - 有效批次大小：$((batch_size * grad_accum_steps)) (等效于双卡配置)"
echo "  - 视觉token：每张拼贴图约49个token（224×224）"
echo "  - 总token数：约6张图 × 49 = 294个视觉token"
echo "  - 最佳模型：已自动保存基于验证集 accuracy 的最佳检查点"
echo ""
echo "监控训练进度："
echo "  - SwanLab 面板：训练过程中会自动记录指标"
echo "  - 命令行进度条：实时显示训练进度（tqdm）"
echo "  - 日志文件：${output_dir}/trainer_state.json"
echo ""
echo "下一步："
echo "  - 训练完成后，可以使用此模型进行 Accept/Reject 预测"
echo "  - 与 Stage1/Stage2 模型对比性能"
echo "  - 评估在不同数据集上的泛化能力"
echo ""
