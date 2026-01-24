#!/bin/bash
#SBATCH -p i64m1tga800u          
#SBATCH -o train_output_%j.txt    
#SBATCH -e train_err_%j.txt       
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:6              
#SBATCH -J lvr_stage1  
#SBATCH -D /hpc2hdd/home/fzhai598/project/Bagel 

set -euo pipefail

source /hpc2hdd/home/fzhai598/project/miniconda3/etc/profile.d/conda.sh
conda activate bagel

echo "=========================================="
echo "Job started at $(date)"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "Job ID: $SLURM_JOB_ID"
    echo "Node: ${SLURM_NODELIST:-N/A}"
fi
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "=========================================="


PROJECT_ROOT=/hpc2hdd/home/fzhai598/project/Bagel
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

# 自动检测GPU数量（6卡配置）
if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
    NUM_GPUS=$SLURM_GPUS_ON_NODE
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 6)
fi
NUM_GPUS=${NUM_GPUS:-6}  # 默认6卡
echo "Detected $NUM_GPUS GPUs"


num_nodes=${num_nodes:-1}
node_rank=${node_rank:-0}
nproc_per_node=${nproc_per_node:-$NUM_GPUS}
master_addr=${master_addr:-127.0.0.1}
master_port=${master_port:-29500}


MODEL_PATH=${MODEL_PATH:-/hpc2hdd/home/fzhai598/project/Bagel/models/BAGEL-7B-MoT}
OUTPUT_DIR=${OUTPUT_DIR:-/hpc2hdd/home/fzhai598/project/Bagel/models/results/stage1_lvr_3000}
CKPT_DIR=${CKPT_DIR:-$OUTPUT_DIR/checkpoints}
DATASET_CONFIG=${DATASET_CONFIG:-./data/configs/stage1_lvr.yaml}


RESUME_FROM=${RESUME_FROM:-$MODEL_PATH}
RESUME_MODEL_ONLY=${RESUME_MODEL_ONLY:-True}
FINETUNE_FROM_EMA=${FINETUNE_FROM_EMA:-True}


LR=${LR:-1e-5}
TOTAL_STEPS=${TOTAL_STEPS:-3000}     
WARMUP_STEPS=${WARMUP_STEPS:-100}     
SAVE_EVERY=${SAVE_EVERY:-500}        
LOG_EVERY=${LOG_EVERY:-10}
EMA=${EMA:-0.9999}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
AUTO_RESUME=${AUTO_RESUME:-True}


# 6卡优化配置：平衡显存与样本覆盖率
EXPECTED_NUM_TOKENS=${EXPECTED_NUM_TOKENS:-3584}          # 3.5K tokens (平衡配置)
MAX_NUM_TOKENS=${MAX_NUM_TOKENS:-4608}                    # 4.5K上限
MAX_NUM_TOKENS_PER_SAMPLE=${MAX_NUM_TOKENS_PER_SAMPLE:-3584}

# LVR 配置
LVR_HEAD=${LVR_HEAD:-False}
LVR_HEAD_TYPE=${LVR_HEAD_TYPE:-simple}
LVR_WEIGHT=${LVR_WEIGHT:-0.1}
LVR_LOSS_FCT=${LVR_LOSS_FCT:-mse}
LVR_LATENT_END_TOKEN=${LVR_LATENT_END_TOKEN:-False}
LVR_LATENT_END_WEIGHT=${LVR_LATENT_END_WEIGHT:-0.1}


WANDB_NAME=${WANDB_NAME:-stage1_lvr}
WANDB_RUNID=${WANDB_RUNID:-$(date +%Y%m%d_%H%M%S)}


NUM_WORKERS=${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-4}


CPU_OFFLOAD=${CPU_OFFLOAD:-False}

NUM_SHARD=${NUM_SHARD:-$nproc_per_node}
USE_FLEX=${USE_FLEX:-True}


DATA_ROOT=/hpc2hdd/home/fzhai598/project/Bagel/Visual-CoT
PARQUET_DIR=$DATA_ROOT/viscot_parquet

if [ ! -d "$PARQUET_DIR" ] || [ -z "$(ls -A $PARQUET_DIR 2>/dev/null)" ]; then
    echo "ERROR: Parquet data not found at $PARQUET_DIR"
    exit 1
fi

parquet_count=$(ls "$PARQUET_DIR"/*.parquet 2>/dev/null | wc -l)
echo "Found $parquet_count parquet files in $PARQUET_DIR"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$CKPT_DIR"

echo "=========================================="
echo "Stage-1 LVR Training Configuration (6 GPUs)"
echo "=========================================="
echo "Number of GPUs: $NUM_GPUS"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "----------------------------------------"
echo "Batch & Performance:"
echo "  Expected tokens/batch: $EXPECTED_NUM_TOKENS (~$((EXPECTED_NUM_TOKENS/NUM_GPUS)) per GPU)"
echo "  Max tokens/batch: $MAX_NUM_TOKENS"
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-5}
echo "  Gradient accumulation: $GRADIENT_ACCUMULATION (effective batch = $((EXPECTED_NUM_TOKENS * GRADIENT_ACCUMULATION)) tokens)"
echo "  CPU offload: $CPU_OFFLOAD (平衡显存与速度)"
echo "  Num workers: $NUM_WORKERS"
echo "  Prefetch factor: $PREFETCH_FACTOR"
echo "----------------------------------------"
echo "LVR config:"
echo "  LVR weight: $LVR_WEIGHT"
echo "  Freeze: Vision=True, LLM=False"
echo "=========================================="

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
  --resume_from "$RESUME_FROM" \
  --resume_model_only "$RESUME_MODEL_ONLY" \
  --finetune_from_ema "$FINETUNE_FROM_EMA" \
  --visual_gen False \
  --visual_und True \
  --lvr_head $LVR_HEAD \
  --lvr_head_type $LVR_HEAD_TYPE \
  --lvr_latent_end_token $LVR_LATENT_END_TOKEN \
  --lvr_weight $LVR_WEIGHT \
  --lvr_latent_end_weight $LVR_LATENT_END_WEIGHT \
  --lvr_loss_fct $LVR_LOSS_FCT \
  --freeze_llm False \
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
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
  --lr_scheduler constant \
  --ema $EMA \
  --max_grad_norm $MAX_GRAD_NORM \
  --auto_resume $AUTO_RESUME \
  --use_flex $USE_FLEX \
  --num_shard $NUM_SHARD \
  --cpu_offload $CPU_OFFLOAD \
  --wandb_offline ${WANDB_OFFLINE:-False} \
  --wandb_name $WANDB_NAME \
  --wandb_runid $WANDB_RUNID

echo "=========================================="
echo "Job ended at $(date)"
echo "=========================================="
