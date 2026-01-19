#!/usr/bin/env bash
# Prepare Visual-CoT data for stage-1 LVR training.
# 优化版本：
# 1. 修复 tar 解压问题
# 2. 自动更新 dataset_info.py
# 3. 添加数据验证

set -euo pipefail

# 配置路径
DATA_ROOT=${DATA_ROOT:-/hpc2hdd/home/fzhai598/project/Bagel/Visual-CoT}
TAR_ROOT=${TAR_ROOT:-$DATA_ROOT/cot_images_tar_split}
IMAGE_ROOT=${IMAGE_ROOT:-$DATA_ROOT}
PARQUET_DIR=${PARQUET_DIR:-$DATA_ROOT/viscot_parquet}
INFO_PATH=${INFO_PATH:-$DATA_ROOT/viscot_parquet_info/viscot_lvr_info.json}
JSON_PATH=${JSON_PATH:-$DATA_ROOT/viscot_363k.json}
METADATA_DIR=${METADATA_DIR:-$DATA_ROOT/metadata}
CONFIG_PATH=${CONFIG_PATH:-/hpc2hdd/home/fzhai598/project/Bagel/data/configs/viscot_lvr_stage1.yaml}
DATASET_INFO_PATH=${DATASET_INFO_PATH:-/hpc2hdd/home/fzhai598/project/Bagel/data/dataset_info.py}

# 处理参数
MAX_SAMPLES=${MAX_SAMPLES:-0}
SHARD_SIZE=${SHARD_SIZE:-500}
ROW_GROUP_SIZE=${ROW_GROUP_SIZE:-128}
USE_METADATA_PRIORITY=${USE_METADATA_PRIORITY:-false}

echo "=========================================="
echo "Visual-CoT Stage-1 Data Preparation"
echo "=========================================="
echo "DATA_ROOT: $DATA_ROOT"
echo "PARQUET_DIR: $PARQUET_DIR"
echo "JSON_PATH: $JSON_PATH"
echo ""

# Step 1: 解压图像
if [ ! -d "$DATA_ROOT/cot_image_data" ]; then
    echo "[1/4] Extracting images from tar splits..."
    
    if [ -d "$TAR_ROOT" ]; then
        cd "$TAR_ROOT"
        
        # 检查文件类型
        first_file=$(ls cot_images_* 2>/dev/null | head -1)
        if [ -z "$first_file" ]; then
            echo "ERROR: No tar files found in $TAR_ROOT"
            exit 1
        fi
        
        file_type=$(file "$first_file" | grep -oE 'tar|gzip|split' || echo "unknown")
        echo "Detected file type: $file_type"
        
        # 根据文件类型选择解压方式
        if file "$first_file" | grep -q "split"; then
            # 分割文件，需要合并
            echo "Files appear to be split tar archives, concatenating..."
            cat cot_images_* | tar -xvf - -C "$DATA_ROOT"
        elif file "$first_file" | grep -q "gzip"; then
            # gzip 压缩的 tar
            echo "Extracting gzipped tar files..."
            for f in cot_images_*; do
                echo "Extracting $f..."
                tar -xzf "$f" -C "$DATA_ROOT"
            done
        elif file "$first_file" | grep -q "tar"; then
            # 普通 tar 文件（可能是分卷）
            echo "Extracting tar files..."
            # 尝试合并解压
            cat cot_images_* | tar -xf - -C "$DATA_ROOT" 2>/dev/null || {
                # 如果合并失败，逐个解压
                echo "Concatenated extraction failed, trying individual extraction..."
                for f in cot_images_*; do
                    echo "Extracting $f..."
                    tar -xf "$f" -C "$DATA_ROOT" 2>/dev/null || true
                done
            }
        else
            # 尝试直接合并解压
            echo "Unknown file type, attempting concatenated extraction..."
            cat cot_images_* | tar -xf - -C "$DATA_ROOT" --checkpoint=10000 --checkpoint-action=dot
        fi
        
        cd - > /dev/null
    else
        echo "WARNING: TAR_ROOT ($TAR_ROOT) does not exist."
        echo "Please ensure images are already extracted or provide correct TAR_ROOT."
    fi
else
    echo "[1/4] Found extracted images in cot_image_data, skip extraction."
fi

# 验证图像目录
echo ""
echo "Checking image directories..."
for dir in "cot_image_data" "cot"; do
    if [ -d "$DATA_ROOT/$dir" ]; then
        count=$(find "$DATA_ROOT/$dir" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) 2>/dev/null | wc -l)
        echo "  $dir: $count images found"
    fi
done

# Step 2: 验证输入数据
echo ""
echo "[2/4] Validating input data..."

if [ ! -f "$JSON_PATH" ]; then
    echo "ERROR: viscot JSON file not found: $JSON_PATH"
    exit 1
fi

json_size=$(du -h "$JSON_PATH" | cut -f1)
echo "  viscot JSON: $JSON_PATH ($json_size)"

if [ -d "$METADATA_DIR" ]; then
    metadata_count=$(ls "$METADATA_DIR"/*.jsonl 2>/dev/null | wc -l)
    echo "  Metadata files: $metadata_count jsonl files in $METADATA_DIR"
else
    echo "  Metadata directory not found (optional)"
    METADATA_DIR=""
fi

# Step 3: 转换为 parquet
echo ""
echo "[3/4] Converting Visual-CoT json to parquet..."

# 构建转换命令参数
CONVERT_ARGS=(
    --viscot_json "$JSON_PATH"
    --image_root "$IMAGE_ROOT"
    --output_dir "$PARQUET_DIR"
    --info_path "$INFO_PATH"
    --shard_size "$SHARD_SIZE"
    --row_group_size "$ROW_GROUP_SIZE"
)

if [ "$MAX_SAMPLES" -gt 0 ]; then
    CONVERT_ARGS+=(--max_samples "$MAX_SAMPLES")
fi

if [ -n "$METADATA_DIR" ]; then
    CONVERT_ARGS+=(--metadata_dir "$METADATA_DIR")
fi

if [ "$USE_METADATA_PRIORITY" = "true" ]; then
    CONVERT_ARGS+=(--use_metadata_priority)
fi

# 运行转换脚本
python /hpc2hdd/home/fzhai598/project/Bagel/scripts/viscot-stage1/convert_viscot_to_parquet_v2.py "${CONVERT_ARGS[@]}"

# Step 4: 更新配置文件
echo ""
echo "[4/4] Updating configuration files..."

# 统计 parquet 文件
parquet_count=$(ls "$PARQUET_DIR"/*.parquet 2>/dev/null | wc -l)
total_rows=$(python3 -c "
import pyarrow.parquet as pq
from pathlib import Path
total = 0
for f in Path('$PARQUET_DIR').glob('*.parquet'):
    total += pq.ParquetFile(f).metadata.num_rows
print(total)
")

echo "  Parquet files: $parquet_count"
echo "  Total rows: $total_rows"

# 更新 YAML 配置
cat > "$CONFIG_PATH" << EOF
vlm_sft_parquet:
  dataset_names:
  - viscot_lvr
  image_transform_args:
    image_stride: 14
    max_image_size: 980
    min_image_size: 378
    max_pixels: 2_007_040
  vit_image_transform_args:
    image_stride: 14
    max_image_size: 980
    min_image_size: 378
    max_pixels: 2_007_040
  is_mandatory: true
  num_used_data:
  - $parquet_count
  weight: 1
EOF
echo "  Updated: $CONFIG_PATH"

# 更新 dataset_info.py
python3 << PY
import re
from pathlib import Path

dataset_info_path = Path("$DATASET_INFO_PATH")
content = dataset_info_path.read_text()

# 更新 viscot_lvr 配置
new_viscot_config = """'viscot_lvr': {
            'data_dir': '$PARQUET_DIR',
            'num_files': $parquet_count,
            'num_total_samples': $total_rows,
            'parquet_info_path': '$INFO_PATH',
        },"""

# 查找并替换 viscot_lvr 配置
pattern = r"'viscot_lvr':\s*\{[^}]+\},"
if re.search(pattern, content):
    content = re.sub(pattern, new_viscot_config, content)
    dataset_info_path.write_text(content)
    print(f"  Updated: $DATASET_INFO_PATH")
else:
    print(f"  WARNING: Could not find viscot_lvr in $DATASET_INFO_PATH")
    print(f"  Please manually add the following configuration:")
    print(new_viscot_config)
PY

echo ""
echo "=========================================="
echo "Data preparation completed!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Parquet output: $PARQUET_DIR"
echo "  - Parquet info: $INFO_PATH"
echo "  - Config file: $CONFIG_PATH"
echo "  - Total samples: $total_rows"
echo ""
echo "You can now run training with:"
echo "  bash scripts/viscot-stage1/train_stage1_lvr_viscot.sh"
