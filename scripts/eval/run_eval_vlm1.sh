# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -x

# Set proxy and API key
export OPENAI_API_KEY=$openai_api_key

export GPUS=1


model_path="/hpc2hdd/home/fzhai598/project/Bagel/models/viscot-stage1-1000"
output_path="/hpc2hdd/home/fzhai598/project/Bagel/outputs_eval/und/viscot_lvr_stage1_1000"

# 检查输出目录是否存在，不存在则创建
if [ ! -d "$output_path" ]; then
    mkdir -p "$output_path"
    echo "Created output directory: $output_path"
fi

DATASETS=("mme" "mmbench-dev-en" "mmvet" "mmmu-val" "mmvp")
# DATASETS=("mathvista-testmini")
# DATASETS=("mmmu-val_cot")

DATASETS_STR="${DATASETS[*]}"
export DATASETS_STR

bash scripts/eval/eval_vlm.sh \
    $output_path \
    --model-path $model_path