#!/bin/bash
#SBATCH -p i64m1tga800u          # 指定GPU队列
#SBATCH -o eval_output_%j.txt    # 指定作业标准输出文件，%j为作业号
#SBATCH -e eval_err_%j.txt       # 指定作业标准错误输出文件
#SBATCH --gres=gpu:2             # 指定GPU卡数
#SBATCH --mem=128G               # 指定内存
#SBATCH --cpus-per-task=16       # 指定CPU核心数
#SBATCH -D /hpc2hdd/home/fzhai598/project/Bagel  # 指定作业执行路径
#SBATCH -J stage1_lvr_eval           # 作业名称


echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# 激活 conda 环境
source /hpc2hdd/home/fzhai598/project/miniconda3/etc/profile.d/conda.sh
conda activate bagel
echo "Conda environment: $CONDA_DEFAULT_ENV"

set -x

# Set proxy and API key
export OPENAI_API_KEY=$openai_api_key

# 使用2张GPU
export GPUS=2

CHECKPOINTS=("500" "1000" "1500" "2000" "2500" "3000")

DATASETS=("mme" "mmbench-dev-en" "mmvet" "mmmu-val" "mmvp")
# DATASETS=("mathvista-testmini")
# DATASETS=("mmmu-val_cot")

DATASETS_STR="${DATASETS[*]}"
export DATASETS_STR


for step in "${CHECKPOINTS[@]}"; do
    model_path="/hpc2hdd/home/fzhai598/project/Bagel/models/stage1.2/stage1_lvr_${step}"
    output_path="/hpc2hdd/home/fzhai598/project/Bagel/outputs_eval/und/stage1.2/stage1_lvr_${step}"
    
    
    if [ ! -d "$model_path" ]; then
        echo "Model not found: $model_path, skipping..."
        continue
    fi
    
    
    if [ ! -d "$output_path" ]; then
        mkdir -p "$output_path"
        echo "Created output directory: $output_path"
    fi
    
    echo "=========================================="
    echo "Evaluating checkpoint: viscot-stage1-${step}"
    echo "Model path: $model_path"
    echo "Output path: $output_path"
    echo "=========================================="
    
    bash scripts/eval/eval_vlm.sh \
        $output_path \
        --model-path $model_path \

    
    echo "Finished evaluating viscot-stage1-${step}"
    echo ""
done

echo "All checkpoints evaluated!"
echo "Job ended at $(date)"