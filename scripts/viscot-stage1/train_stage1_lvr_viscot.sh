#!/usr/bin/env bash
# Stage-1 LVR SFT on Visual-CoT (parquet)
# 
# 训练目标：训练 LVR head 和 connector，使其能够预测 ViT 视觉 token
# 冻结组件：LLM, ViT, VAE
# 可训练组件：connector, lvr_head
#
# 使用前请确保已运行数据预处理：
#   bash scripts/prepare_viscot_stage1_v2.sh

set -euo pipefail

# ============== 设置 Python 路径 ==============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 脚本在 scripts/viscot-stage1/ 下，需要回退两级到项目根目录
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

# ============== GPU 配置 ==============
# 指定使用的 GPU（避开被占用的 GPU 0,1,5）
# 强制使用空闲 GPU，除非通过 USE_GPUS 显式指定
export CUDA_VISIBLE_DEVICES=${USE_GPUS:-0,1}
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

# ============== 分布式训练配置 ==============
num_nodes=${num_nodes:-1}
node_rank=${node_rank:-0}
nproc_per_node=${nproc_per_node:-$NUM_GPUS}
master_addr=${master_addr:-127.0.0.1}
master_port=${master_port:-29500}

# ============== 路径配置 ==============
MODEL_PATH=${MODEL_PATH:-/hpc2hdd/home/fzhai598/project/Bagel/models/BAGEL-7B-MoT}
OUTPUT_DIR=${OUTPUT_DIR:-/hpc2hdd/home/fzhai598/project/Bagel/results/viscot_lvr_stage1_1500}
CKPT_DIR=${CKPT_DIR:-/hpc2hdd/home/fzhai598/project/Bagel/results/viscot_lvr_stage1_1500/checkpoints}
DATASET_CONFIG=${DATASET_CONFIG:-./data/configs/viscot_lvr_stage1.yaml}

# ============== 预训练权重加载配置 ==============
# 默认从 MODEL_PATH 加载 ema.safetensors 作为预训练权重
# 这样可以在冻结 LLM/ViT 的情况下正确初始化模型
RESUME_FROM=${RESUME_FROM:-$MODEL_PATH}
RESUME_MODEL_ONLY=${RESUME_MODEL_ONLY:-True}
FINETUNE_FROM_EMA=${FINETUNE_FROM_EMA:-True}

# ============== 训练超参数 ==============
# 参考官方 TRAIN.md: https://github.com/ByteDance-Seed/Bagel/blob/main/TRAIN.md
# 微调预训练模型时，学习率建议使用 2e-5 (官方微调示例)
LR=${LR:-2e-5}
TOTAL_STEPS=${TOTAL_STEPS:-1500}     # 完整训练: 50000 步
WARMUP_STEPS=${WARMUP_STEPS:-150}     # 完整训练: 500 步 (1%)
SAVE_EVERY=${SAVE_EVERY:-300}        # 完整训练: 5000 步保存一次
LOG_EVERY=${LOG_EVERY:-10}
EMA=${EMA:-0.9999}                    # 官方默认 0.9999
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}   # 官方默认 1.0
AUTO_RESUME=${AUTO_RESUME:-True}      # 官方微调示例推荐

# ============== 序列长度配置 ==============
# 5卡 A800 80GB 可以处理更大的 batch
# 增大 tokens 数量可提高 GPU 利用率和训练速度
EXPECTED_NUM_TOKENS=${EXPECTED_NUM_TOKENS:-16384}
MAX_NUM_TOKENS=${MAX_NUM_TOKENS:-18432}
MAX_NUM_TOKENS_PER_SAMPLE=${MAX_NUM_TOKENS_PER_SAMPLE:-16384}

# ============== LVR 配置 ==============
# LVR_WEIGHT 控制 LVR loss 的权重，增大可以加强视觉定位能力的学习
# 建议范围: 1.0 ~ 5.0，默认 2.0
LVR_WEIGHT=${LVR_WEIGHT:-2.0}
LVR_LOSS_FCT=${LVR_LOSS_FCT:-mse}  # 可选: mse, mae, cosine
LVR_HEAD_TYPE=${LVR_HEAD_TYPE:-simple}  # 可选: simple, glu

# ============== WandB 配置 ==============
WANDB_NAME=${WANDB_NAME:-viscot_lvr_stage1}
WANDB_RUNID=${WANDB_RUNID:-$(date +%Y%m%d_%H%M%S)}

# ============== DataLoader 配置 ==============
NUM_WORKERS=${NUM_WORKERS:-8}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-4}

# ============== FSDP 配置 ==============
# num_shard 必须 <= GPU 数量，双卡设为2，8卡设为8
NUM_SHARD=${NUM_SHARD:-$nproc_per_node}

# ============== FlexAttention 配置 ==============
# use_flex=True 可提高 GPU 利用率，但需要验证兼容性
# 官方 TRAIN.md 在 pre-training 示例中使用 use_flex=True
USE_FLEX=${USE_FLEX:-True}

# ============== 检查数据是否准备好 ==============
DATA_ROOT=/hpc2hdd/home/fzhai598/project/Bagel/Visual-CoT
PARQUET_DIR=$DATA_ROOT/viscot_parquet

if [ ! -d "$PARQUET_DIR" ] || [ -z "$(ls -A $PARQUET_DIR 2>/dev/null)" ]; then
    echo "ERROR: Parquet data not found at $PARQUET_DIR"
    echo "Please run data preparation first:"
    echo "  bash scripts/prepare_viscot_stage1_v2.sh"
    exit 1
fi

parquet_count=$(ls "$PARQUET_DIR"/*.parquet 2>/dev/null | wc -l)
echo "Found $parquet_count parquet files in $PARQUET_DIR"

# ============== 创建输出目录 ==============
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CKPT_DIR"

echo "=========================================="
echo "Stage-1 LVR Training Configuration"
echo "(based on official TRAIN.md recommendations)"
echo "=========================================="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $NUM_GPUS"
echo "nproc_per_node: $nproc_per_node"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Dataset config: $DATASET_CONFIG"
echo "----------------------------------------"
echo "Training hyperparameters:"
echo "  Total steps: $TOTAL_STEPS"
echo "  Learning rate: $LR"
echo "  Warmup steps: $WARMUP_STEPS"
echo "  Save every: $SAVE_EVERY"
echo "  EMA: $EMA"
echo "  Max grad norm: $MAX_GRAD_NORM"
echo "  Auto resume: $AUTO_RESUME"
echo "  Use FlexAttention: $USE_FLEX"
echo "----------------------------------------"
echo "LVR config:"
echo "  LVR weight: $LVR_WEIGHT"
echo "  LVR loss: $LVR_LOSS_FCT"
echo "  LVR head type: $LVR_HEAD_TYPE"
echo "----------------------------------------"
echo "WandB: $WANDB_NAME (run: $WANDB_RUNID)"
if [ -n "$RESUME_FROM" ]; then
    echo "Resume from: $RESUME_FROM"
    echo "  model_only=$RESUME_MODEL_ONLY, finetune_from_ema=$FINETUNE_FROM_EMA"
fi
echo "=========================================="

if [ -n "$RESUME_FROM" ] && [ ! -e "$RESUME_FROM" ]; then
    echo "ERROR: RESUME_FROM path does not exist: $RESUME_FROM"
    exit 1
fi

RESUME_ARGS=()
if [ -n "$RESUME_FROM" ]; then
    RESUME_ARGS+=(--resume_from "$RESUME_FROM")
    RESUME_ARGS+=(--resume_model_only "$RESUME_MODEL_ONLY")
    RESUME_ARGS+=(--finetune_from_ema "$FINETUNE_FROM_EMA")
fi

torchrun \
  --nnodes=$num_nodes \
  --node_rank=$node_rank \
  --nproc_per_node=$nproc_per_node \
  --master_addr=$master_addr \
  --master_port=$master_port \
  train/pretrain_unified_navit.py \
  --dataset_config_file $DATASET_CONFIG \
  --model_path $MODEL_PATH \
  --finetune_from_hf True \
  --max_latent_size 64 \
  --results_dir $OUTPUT_DIR \
  --checkpoint_dir $CKPT_DIR \
  "${RESUME_ARGS[@]}" \
  --visual_gen False \
  --visual_und True \
  --lvr_head True \
  --lvr_head_type $LVR_HEAD_TYPE \
  --lvr_weight $LVR_WEIGHT \
  --lvr_loss_fct $LVR_LOSS_FCT \
  --freeze_llm True \
  --freeze_vit True \
  --freeze_vae True \
  --freeze_und False \
  --lr $LR \
  --total_steps $TOTAL_STEPS \
  --warmup_steps $WARMUP_STEPS \
  --save_every $SAVE_EVERY \
  --log_every $LOG_EVERY \
  --num_workers $NUM_WORKERS \
  --prefetch_factor $PREFETCH_FACTOR \
  --expected_num_tokens $EXPECTED_NUM_TOKENS \
  --max_num_tokens $MAX_NUM_TOKENS \
  --max_num_tokens_per_sample $MAX_NUM_TOKENS_PER_SAMPLE \
  --ce_weight 1.0 \
  --gradient_accumulation_steps 1 \
  --lr_scheduler constant \
  --ema $EMA \
  --max_grad_norm $MAX_GRAD_NORM \
  --auto_resume $AUTO_RESUME \
  --use_flex $USE_FLEX \
  --num_shard $NUM_SHARD \
  --wandb_offline ${WANDB_OFFLINE:-False} \
  --wandb_name $WANDB_NAME \
  --wandb_runid $WANDB_RUNID
