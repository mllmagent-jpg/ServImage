#!/bin/bash

set -euo pipefail

# 切换到项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
cd "${PROJECT_ROOT}"
echo "工作目录: $(pwd)"

# 激活所需的 Conda 环境
set +u
CONDA_ROOT=${CONDA_ROOT:-"${HOME}/miniconda3"}
if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate servimage
else
  echo "[WARNING] 未找到 conda.sh，请确保已激活正确的 Python 环境"
fi
set -u

# ==============================
# Stage2 训练配置（基于Stage1特征的决策训练）
# 说明：第二阶段训练，基于Stage1特征进行Accept/Reject决策
#       - 任务模式：stage2
#       - 输出：Accept/Reject (0/1标签) + 7维度评分
#       - 依赖：需要 Stage1 训练完成的模型
#       - 训练策略：冻结Stage1，特征拼接 + 新分类头
#       - 数据集：CBM标准拆分数据集
# ==============================
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${NNODES:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}

# 自动推导可用 GPU 数
IFS=',' read -ra GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=${#GPU_LIST[@]}

if [[ "${NUM_GPUS}" -ne 4 ]]; then
  echo "[WARNING] 该脚本推荐使用 4 卡训练，当前检测到 ${NUM_GPUS} 卡: ${CUDA_VISIBLE_DEVICES}" >&2
  echo "[WARNING] 将继续使用 ${NUM_GPUS} 卡进行训练..." >&2
fi

# DeepSpeed通信配置
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export DEEPSPEED_CHECK_DATALOADER_INPUTS=0

# PyTorch 内存分配优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# 禁用 SwanLab（避免I/O错误）
export DISABLE_SWANLAB=1

# ==============================
# Stage2 LoRA 配置
# ==============================
# Vision Encoder LoRA
export STAGE2_LORA_VISION_ENABLE=${STAGE2_LORA_VISION_ENABLE:-1}        # 1=启用, 0=禁用
export STAGE2_LORA_VISION_R=${STAGE2_LORA_VISION_R:-8}                  # LoRA rank
export STAGE2_LORA_VISION_ALPHA=${STAGE2_LORA_VISION_ALPHA:-16}         # LoRA alpha
export STAGE2_LORA_VISION_DROPOUT=${STAGE2_LORA_VISION_DROPOUT:-0.01}   # LoRA dropout

# Projector (visual.merger) LoRA
export STAGE2_LORA_PROJECTOR_ENABLE=${STAGE2_LORA_PROJECTOR_ENABLE:-1}        # 1=启用, 0=禁用
export STAGE2_LORA_PROJECTOR_R=${STAGE2_LORA_PROJECTOR_R:-8}                  # LoRA rank
export STAGE2_LORA_PROJECTOR_ALPHA=${STAGE2_LORA_PROJECTOR_ALPHA:-16}         # LoRA alpha
export STAGE2_LORA_PROJECTOR_DROPOUT=${STAGE2_LORA_PROJECTOR_DROPOUT:-0.01}   # LoRA dropout

# Language Model LoRA
export STAGE2_LORA_LLM_ENABLE=${STAGE2_LORA_LLM_ENABLE:-0}               # 1=启用, 0=禁用
export STAGE2_LORA_LLM_R=${STAGE2_LORA_LLM_R:-8}                         # LoRA rank
export STAGE2_LORA_LLM_ALPHA=${STAGE2_LORA_LLM_ALPHA:-16}                # LoRA alpha
export STAGE2_LORA_LLM_DROPOUT=${STAGE2_LORA_LLM_DROPOUT:-0.01}          # LoRA dropout

# Decoder 配置
export STAGE2_DECODER_HIDDEN_SIZE=${STAGE2_DECODER_HIDDEN_SIZE:-512}             # Decoder 中间层维度
export STAGE2_DECODER_INTERMEDIATE_SIZE=${STAGE2_DECODER_INTERMEDIATE_SIZE:-2048} # Decoder 第一层维度
export STAGE2_DECODER_DROPOUT=${STAGE2_DECODER_DROPOUT:-0.1}                     # Decoder dropout

# DeepSpeed 配置文件（4卡使用ZeRO-3）
deepspeed_config=${DEEPSPEED_CONFIG:-scripts/deepspeed_zero3.json}

# ==============================
# 训练与数据配置（Stage2）
# ==============================
llm=${MODEL_PATH:-"weights/Qwen3-VL-4B-Thinking"}

# Stage1 模型路径（必须指定）
stage1_model_path=${STAGE1_MODEL_PATH:-"output/ServImageModel_Stage1_4binstr/checkpoint-3040"}
if [[ -z "${stage1_model_path}" ]]; then
  echo "[ERROR] 请通过 STAGE1_MODEL_PATH 环境变量指定 Stage1 训练完成的模型路径" >&2
  echo "示例: STAGE1_MODEL_PATH=./output/stage1-xxx bash scripts/train_stage2.sh" >&2
  exit 1
fi

if [[ ! -d "${stage1_model_path}" ]]; then
  echo "[ERROR] Stage1 模型路径不存在: ${stage1_model_path}" >&2
  exit 1
fi

echo "[Stage2] 检测到 Stage1 模型: ${stage1_model_path}"

# Stage2 特定配置
lr=${LR:-4e-5}  # Stage2 使用较小学习率
batch_size=${BATCH_SIZE:-2}  # 4卡训练，每卡batch size=2
grad_accum_steps=${GRAD_ACCUM_STEPS:-8}  # 梯度累积步数，有效batch size = 2*4*8 = 64
num_epochs=${NUM_EPOCHS:-2}  # Stage2 训练轮数
weight_decay=${WEIGHT_DECAY:-0.01}

save_steps=${SAVE_STEPS:-500}  # 定期保存检查点（必须与eval_steps匹配）
eval_strategy=${EVAL_STRATEGY:-"steps"}  # 启用评估
eval_steps=${EVAL_STEPS:-500}  # 每500步评估一次
max_eval_samples=${MAX_EVAL_SAMPLES:-10000}  # 验证集采样数量：每次评估2000个样本（约36%的数据，已添加显存优化）

# 数据路径配置
bench_dir=${BENCH_DIR:-"${PROJECT_ROOT}/data/ServImage_Bench"}
dataset_dir=${DATASET_DIR:-"${PROJECT_ROOT}/data/ServImage_Dataset"}

# Stage2 数据集路径（标准拆分 - Accept/Reject 二分类标签）
# 数据格式：每行一个子任务，包含 label 字段 (0=Reject, 1=Accept)
# 与 Stage1 的区别：Stage1 学习 7 维度质量理解，Stage2 基于质量特征做决策
stage2_data_dir="${PROJECT_ROOT}/data/Label_data/standard_split"
train_data=${TRAIN_DATA:-"${stage2_data_dir}/train.jsonl"}
val_data=${VAL_DATA:-"${stage2_data_dir}/val.jsonl"}
test_data=${TEST_DATA:-"${stage2_data_dir}/test.jsonl"}

# 训练脚本
entry_file=${ENTRY_FILE:-"train/train_stage2.py"}
run_name="stage2-accept-reject-2b-$(date +%Y%m%d_%H%M%S)"
output_dir=${OUTPUT_DIR:-"./output/${run_name}"}

echo "========================================="
echo "Stage2 训练（基于Stage1特征的决策训练）"
echo "========================================="
echo "模型: ${llm}"
echo "Stage1 模型: ${stage1_model_path}"
echo "学习率: ${lr}"
echo "权重衰减: ${weight_decay}"
echo "Train batch/GPU: ${batch_size}"
echo "Gradient Accum: ${grad_accum_steps}"
echo "Epochs: ${num_epochs}"
echo "GPU数量: ${NUM_GPUS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "DeepSpeed 配置: ${deepspeed_config}"
echo "输出目录: ${output_dir}"
echo ""
echo "Stage2 LoRA 配置:"
echo "  Vision Encoder LoRA:"
echo "    - 启用: $([ ${STAGE2_LORA_VISION_ENABLE} -eq 1 ] && echo 'Yes' || echo 'No')"
echo "    - Rank: ${STAGE2_LORA_VISION_R}"
echo "    - Alpha: ${STAGE2_LORA_VISION_ALPHA}"
echo "    - Dropout: ${STAGE2_LORA_VISION_DROPOUT}"
echo "  Projector (visual.merger) LoRA:"
echo "    - 启用: $([ ${STAGE2_LORA_PROJECTOR_ENABLE} -eq 1 ] && echo 'Yes' || echo 'No')"
echo "    - Rank: ${STAGE2_LORA_PROJECTOR_R}"
echo "    - Alpha: ${STAGE2_LORA_PROJECTOR_ALPHA}"
echo "    - Dropout: ${STAGE2_LORA_PROJECTOR_DROPOUT}"
echo "  Language Model LoRA:"
echo "    - 启用: $([ ${STAGE2_LORA_LLM_ENABLE} -eq 1 ] && echo 'Yes' || echo 'No')"
echo "    - Rank: ${STAGE2_LORA_LLM_R}"
echo "    - Alpha: ${STAGE2_LORA_LLM_ALPHA}"
echo "    - Dropout: ${STAGE2_LORA_LLM_DROPOUT}"
echo "  Decoder 配置:"
echo "    - Hidden Size: ${STAGE2_DECODER_HIDDEN_SIZE}"
echo "    - Intermediate Size: ${STAGE2_DECODER_INTERMEDIATE_SIZE}"
echo "    - Dropout: ${STAGE2_DECODER_DROPOUT}"
echo ""
echo "Stage2 特性:"
echo "  - 任务模式: stage2 (第二阶段训练)"
echo "  - 输入: Stage1的7维度评分 + 原始VLM特征"
echo "  - 输出: Accept/Reject (0/1标签)"
echo "  - 训练策略: 冻结Stage1模型"
echo "    * 提取Stage1最后一层隐藏状态"
echo "    * 提取原始VLM最后一层隐藏状态（无LoRA）"
echo "    * 拼接两个特征向量"
echo "    * 训练新的二分类头"
echo "  - 价格感知: Prompt包含任务价格信息"
echo "  - 图片压缩策略：与Stage1相同"
echo "  - 当前结果图：1张原图（不压缩）"
echo "  - 输入图：最多1张原图（不压缩）"
echo "  - 其它结果图：压缩为最多2张拼贴图（动态网格）"
echo "  - 参考图：压缩为最多2张拼贴图（动态网格）"
echo "  - 总图片数：最多6张"
echo ""
echo "数据配置（Standard Split - 二分类）:"
echo "  - Bench 目录: ${bench_dir}"
echo "  - Dataset 目录: ${dataset_dir}"
echo "  - Train 数据: ${train_data}"
echo "  - Val 数据: ${val_data}"
echo "  - Test 数据: ${test_data}"
echo ""
echo "评估配置:"
echo "  - 策略: ${eval_strategy}"
echo "  - 评估步数: ${eval_steps}"
echo "  - 验证集采样数: ${max_eval_samples} (动态采样)"
echo "  - 保存步数: ${save_steps}"
echo ""
echo "监控与日志:"
echo "  - SwanLab: 已禁用（避免I/O错误）"
echo "  - 项目名: qwen3vl-stage2-accept-reject-quality-eval"
echo "  - 训练进度条: 已启用（disable_tqdm=False）"
echo "  - 检查点保存: 每${save_steps}步保存一次 (最多保留2个)"
echo "========================================="

args="
    --model_name_or_path ${llm} \
    --stage1_model_path ${stage1_model_path} \
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
echo "使用 DeepSpeed 启动：${NUM_GPUS} 卡 ZeRO-3"
echo "指定 GPU: ${CUDA_VISIBLE_DEVICES}"
echo ""

# 禁用 SwanLab（避免I/O错误）
export DISABLE_SWANLAB=1

# 使用 --include 参数明确指定 GPU
deepspeed --include localhost:${CUDA_VISIBLE_DEVICES} \
          --master_addr=${MASTER_ADDR} \
          --master_port=${MASTER_PORT} \
          ${entry_file} ${args}

echo ""
echo "训练完成，输出目录: ${output_dir}"
echo ""
