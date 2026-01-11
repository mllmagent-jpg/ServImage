#!/bin/bash

set -euo pipefail

# 设置CUDA内存分配优化（避免碎片化）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
# Stage1 训练配置（7维度评分学习）
# 说明：第一阶段训练，学习7维度详细评分
#       - 任务模式：stage1
#       - 输出：7维度评分 (BRF, VEQ-Clarity, VEQ-Realism, VEQ-Aesthetic, VEQ-Text, CNS-Edit, CNS-Set)
#       - 数据集：CBM标准拆分数据集
#       - 训练策略：LoRA + 7个评分头
#       - 图片压缩：当前图+输入图原图，其他图压缩为拼贴图
# ==============================
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${NNODES:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}

# 自动推导可用 GPU 数
IFS=',' read -ra GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=${#GPU_LIST[@]}

if [[ "${NUM_GPUS}" -ne 4 ]]; then
  echo "[ERROR] 该脚本仅用于 4 卡 (约 48GB 显存/卡)，当前检测到 ${NUM_GPUS} 卡: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi

# DeepSpeed通信配置
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export DEEPSPEED_CHECK_DATALOADER_INPUTS=0

# NCCL调试和超时设置
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# PyTorch 内存分配优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# ==============================
# Stage1 LoRA 配置
# ==============================
# Vision Encoder LoRA
export STAGE1_LORA_VISION_ENABLE=${STAGE1_LORA_VISION_ENABLE:-1}        # 1=启用, 0=禁用
export STAGE1_LORA_VISION_R=${STAGE1_LORA_VISION_R:-16}                 # LoRA rank
export STAGE1_LORA_VISION_ALPHA=${STAGE1_LORA_VISION_ALPHA:-16}         # LoRA alpha
export STAGE1_LORA_VISION_DROPOUT=${STAGE1_LORA_VISION_DROPOUT:-0.01}   # LoRA dropout

# Projector (visual.merger) LoRA
export STAGE1_LORA_PROJECTOR_ENABLE=${STAGE1_LORA_PROJECTOR_ENABLE:-1}        # 1=启用, 0=禁用
export STAGE1_LORA_PROJECTOR_R=${STAGE1_LORA_PROJECTOR_R:-16}                 # LoRA rank
export STAGE1_LORA_PROJECTOR_ALPHA=${STAGE1_LORA_PROJECTOR_ALPHA:-16}         # LoRA alpha
export STAGE1_LORA_PROJECTOR_DROPOUT=${STAGE1_LORA_PROJECTOR_DROPOUT:-0.01}   # LoRA dropout

# Language Model LoRA
export STAGE1_LORA_LLM_ENABLE=${STAGE1_LORA_LLM_ENABLE:-1}               # 1=启用, 0=禁用
export STAGE1_LORA_LLM_R=${STAGE1_LORA_LLM_R:-16}                        # LoRA rank
export STAGE1_LORA_LLM_ALPHA=${STAGE1_LORA_LLM_ALPHA:-16}                # LoRA alpha
export STAGE1_LORA_LLM_DROPOUT=${STAGE1_LORA_LLM_DROPOUT:-0.01}          # LoRA dropout

# Decoder 配置
export STAGE1_CONCEPT_HIDDEN_SIZE=${STAGE1_CONCEPT_HIDDEN_SIZE:-512}     # 概念层隐藏维度
export STAGE1_DECODER_DROPOUT=${STAGE1_DECODER_DROPOUT:-0.1}             # Decoder dropout

# DeepSpeed 配置文件（双GPU，ZeRO-2）
deepspeed_config=${DEEPSPEED_CONFIG:-scripts/deepspeed_basic.json}

# ==============================
# 训练与数据配置（Stage1）
# ==============================
llm=${MODEL_PATH:-"weights/Qwen3-VL-4B-Thinking"}

# Stage1 特定配置
lr=${LR:-4e-5}
batch_size=${BATCH_SIZE:-1}  # 每个GPU的batch size
grad_accum_steps=${GRAD_ACCUM_STEPS:-8}  # 有效batch size = 1*4*8 = 32
num_epochs=${NUM_EPOCHS:-5}
weight_decay=${WEIGHT_DECAY:-0.01}

save_steps=${SAVE_STEPS:-500}  # 定期保存检查点
eval_strategy=${EVAL_STRATEGY:-"steps"}  # 启用评估
eval_steps=${EVAL_STEPS:-500}  # 每500步评估一次
max_eval_samples=${MAX_EVAL_SAMPLES:-10000}  # 验证集采样数量：每次评估2000个样本（约36%的数据，已添加显存优化）

# 数据路径配置（相对于项目根目录）
bench_dir=${BENCH_DIR:-"data/ServImage_Bench"}
dataset_dir=${DATASET_DIR:-"data/ServImage_Dataset"}

# CBM 数据集路径（标准拆分 - 包含7维度评分）
# 数据格式：每行一个子任务，包含 brf_overall_score, veq_clarity, veq_realism, veq_aesthetic, veq_text, cns_edit_consistency, cns_set_consistency
cbm_data_dir="data/Label_data/cbm_split/standard"
train_data=${TRAIN_DATA:-"${cbm_data_dir}/train.jsonl"}
val_data=${VAL_DATA:-"${cbm_data_dir}/val.jsonl"}
test_data=${TEST_DATA:-"${cbm_data_dir}/test.jsonl"}

# 训练脚本
entry_file=${ENTRY_FILE:-"train/train_stage1.py"}
run_name="stage1-7dim-scores-4b-$(date +%Y%m%d_%H%M%S)"
output_dir=${OUTPUT_DIR:-"./output/${run_name}"}

echo "========================================="
echo "Stage1 训练（7维度评分学习）"
echo "========================================="
echo "模型: ${llm}"
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
echo "Stage1 LoRA 配置:"
echo "  Vision Encoder LoRA:"
echo "    - 启用: $([ ${STAGE1_LORA_VISION_ENABLE} -eq 1 ] && echo 'Yes' || echo 'No')"
echo "    - Rank: ${STAGE1_LORA_VISION_R}"
echo "    - Alpha: ${STAGE1_LORA_VISION_ALPHA}"
echo "    - Dropout: ${STAGE1_LORA_VISION_DROPOUT}"
echo "  Projector (visual.merger) LoRA:"
echo "    - 启用: $([ ${STAGE1_LORA_PROJECTOR_ENABLE} -eq 1 ] && echo 'Yes' || echo 'No')"
echo "    - Rank: ${STAGE1_LORA_PROJECTOR_R}"
echo "    - Alpha: ${STAGE1_LORA_PROJECTOR_ALPHA}"
echo "    - Dropout: ${STAGE1_LORA_PROJECTOR_DROPOUT}"
echo "  Language Model LoRA:"
echo "    - 启用: $([ ${STAGE1_LORA_LLM_ENABLE} -eq 1 ] && echo 'Yes' || echo 'No')"
echo "    - Rank: ${STAGE1_LORA_LLM_R}"
echo "    - Alpha: ${STAGE1_LORA_LLM_ALPHA}"
echo "    - Dropout: ${STAGE1_LORA_LLM_DROPOUT}"
echo "  Decoder 配置:"
echo "    - Concept Hidden Size: ${STAGE1_CONCEPT_HIDDEN_SIZE}"
echo "    - Dropout: ${STAGE1_DECODER_DROPOUT}"
echo ""
echo "Stage1 特性:"
echo "  - 任务模式: stage1 (第一阶段训练)"
echo "  - 输出: 7维度评分"
echo "    * BRF: 基础需求完成度 (0-5分)"
echo "    * VEQ-Clarity: 清晰度与细节 (0-5分)"
echo "    * VEQ-Realism: 真实性与伪影 (0-5分)"
echo "    * VEQ-Aesthetic: 美学质量 (0-5分)"
echo "    * VEQ-Text: 文字质量 (0-5分或N/A)"
echo "    * CNS-Edit: 编辑一致性 (1-5分或N/A)"
echo "    * CNS-Set: 集合一致性 (1-5分或N/A)"
echo "  - 训练目标: 学习图像质量理解"
echo "  - 图片压缩策略：减少视觉token数量"
echo "  - 当前结果图：1张原图（不压缩）"
echo "  - 输入图：最多1张原图（不压缩）"
echo "  - 其它结果图：压缩为最多2张拼贴图（动态网格）"
echo "  - 参考图：压缩为最多2张拼贴图（动态网格）"
echo "  - 总图片数：最多6张"
echo ""
echo "数据配置（CBM标准拆分）:"
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
echo "  - SwanLab: 已集成（自动初始化）"
echo "  - 项目名: qwen3vl-stage1-7dim-quality-eval"
echo "  - 训练进度条: 已启用（disable_tqdm=False）"
echo "  - 检查点保存: 每${save_steps}步保存一次 (最多保留2个)"
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
    --dataloader_num_workers 4 \
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
echo "使用 DeepSpeed 启动：4 卡 ZeRO-0（四卡训练，GPU 0,1,2,3）"
echo "指定 GPU: ${CUDA_VISIBLE_DEVICES}"
echo ""

# 使用 --include 参数明确指定 GPU
deepspeed --include localhost:${CUDA_VISIBLE_DEVICES} \
          --master_addr=${MASTER_ADDR} \
          --master_port=${MASTER_PORT} \
          ${entry_file} ${args}

echo ""
echo "训练完成，输出目录: ${output_dir}"
echo ""
echo "Stage1 训练说明："
echo "  - 任务类型：第一阶段训练 - 7维度评分学习"
echo "  - 输出：7个维度的回归预测值"
echo "  - 训练策略：Vision LoRA + Projector LoRA + LLM LoRA + 7个评分头"
echo "  - 图片压缩：其它结果图和参考图使用动态网格拼贴"
echo "  - 显存优化：使用图片压缩策略，batch size=4"
echo "  - 视觉token：每张拼贴图约49个token（224×224）"
echo "  - 总token数：约6张图 × 49 = 294个视觉token"
echo "  - 最佳模型：已自动保存基于验证集 loss 的最佳检查点"
echo ""
echo "监控训练进度："
echo "  - SwanLab 面板：训练过程中会自动记录7维度loss"
echo "  - 命令行进度条：实时显示训练进度（tqdm）"
echo "  - 日志文件：${output_dir}/trainer_state.json"
echo ""
echo "下一步："
echo "  - 训练完成后，使用此模型的特征进行 Stage2 训练"
echo "  - Stage2 会冻结 Stage1 模型，提取特征进行决策训练"
echo "  - 评估7维度预测的准确性和相关性"
echo "  - 根据验证集结果调整学习率和权重"
echo ""
