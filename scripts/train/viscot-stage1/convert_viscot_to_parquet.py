#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# 优化版本：
# 1. 修复 bbox 坐标归一化问题
# 2. 支持从 metadata 目录读取更丰富的数据
# 3. 添加数据验证和统计
# 4. 支持多种数据源

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Dict, Any

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


def resolve_image_path(image_root: Path, image_name: str, dataset: str = None) -> Optional[Path]:
    """
    解析图像路径，支持多种目录结构：
    - cot/dataset_name/image.jpg
    - cot_image_data/dataset_name/image.jpg
    - dataset_name/image.jpg (直接在 image_root 下)
    """
    # 直接路径
    image_path = image_root / image_name
    if image_path.exists():
        return image_path
    
    # cot/ -> cot_image_data/
    if image_name.startswith("cot/"):
        alt = image_root / ("cot_image_data/" + image_name[4:])
        if alt.exists():
            return alt
    
    # cot_image_data/ -> cot/
    if image_name.startswith("cot_image_data/"):
        alt = image_root / ("cot/" + image_name[len("cot_image_data/"):])
        if alt.exists():
            return alt
    
    # 尝试在 cot_image_data 中查找（用于 metadata 中只有文件名的情况）
    if dataset:
        for prefix in ["cot_image_data", "cot"]:
            alt = image_root / prefix / dataset / image_name
            if alt.exists():
                return alt
    
    return None


def load_image_bytes_and_size(image_root: Path, image_name: str, dataset: str = None):
    """
    加载图像字节和原始尺寸
    返回: (bytes, width, height) 或 (None, None, None)
    """
    image_path = resolve_image_path(image_root, image_name, dataset)
    if image_path is None:
        return None, None, None
    
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            img = img.convert("RGB")
            
            # 如果是支持的格式，直接读取字节
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                buffer = image_path.read_bytes()
                return buffer, width, height
            
            # 否则转换为 JPEG
            import io
            byte_io = io.BytesIO()
            img.save(byte_io, format="JPEG", quality=95)
            return byte_io.getvalue(), width, height
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None, None


def normalize_bbox(bbox: List[float], width: int, height: int) -> List[float]:
    """
    将像素坐标的 bbox 归一化到 [0, 1] 范围
    输入: [x1, y1, x2, y2] 像素坐标或归一化坐标
    输出: [x1, y1, x2, y2] 归一化坐标
    """
    if not bbox or len(bbox) != 4:
        return []
    
    x1, y1, x2, y2 = bbox
    
    # 判断是否已经是归一化坐标
    if max(x1, y1, x2, y2) <= 1.0 and min(x1, y1, x2, y2) >= 0.0:
        return [x1, y1, x2, y2]
    
    # 像素坐标 -> 归一化坐标
    if width > 0 and height > 0:
        return [
            max(0.0, min(1.0, x1 / width)),
            max(0.0, min(1.0, y1 / height)),
            max(0.0, min(1.0, x2 / width)),
            max(0.0, min(1.0, y2 / height)),
        ]
    
    return []


def iter_json_array(json_path: Path):
    """流式解析大型 JSON 数组"""
    decoder = json.JSONDecoder()
    buffer = ""
    idx = 0
    
    with json_path.open("r", encoding="utf-8") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                return
            buffer += chunk
            while idx < len(buffer) and buffer[idx].isspace():
                idx += 1
            if idx < len(buffer):
                if buffer[idx] != "[":
                    raise ValueError("Invalid JSON array")
                idx += 1
                break
        
        while True:
            while idx < len(buffer) and buffer[idx].isspace():
                idx += 1
            while idx < len(buffer) and buffer[idx] == ",":
                idx += 1
                while idx < len(buffer) and buffer[idx].isspace():
                    idx += 1
            if idx < len(buffer) and buffer[idx] == "]":
                return
            try:
                obj, end = decoder.raw_decode(buffer, idx)
                yield obj
                idx = end
            except json.JSONDecodeError:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    return
                buffer = buffer[idx:] + chunk
                idx = 0


def parse_bbox_from_conversation(conversations: List[Dict]) -> List[List[float]]:
    """
    从对话中解析 bbox（归一化坐标）
    优先返回 GPT 回复中的 bbox
    """
    for conv in conversations:
        if conv.get("from") != "gpt":
            continue
        value = str(conv.get("value", "")).strip()
        if value.startswith("[") and value.endswith("]"):
            try:
                bbox = json.loads(value)
            except Exception:
                continue
            if isinstance(bbox, list) and len(bbox) == 4:
                # 验证是否为有效的归一化坐标
                if all(isinstance(v, (int, float)) for v in bbox):
                    return [bbox]
    return []


def parse_bbox_from_image_field(image_field: Any, width: int = 0, height: int = 0) -> List[List[float]]:
    """
    从 image 字段解析 bbox（像素坐标），并归一化
    image 字段格式: ["path.jpg", "path.jpg###[x1,y1,x2,y2]"]
    """
    if not isinstance(image_field, list) or len(image_field) < 2:
        return []
    
    second = image_field[1]
    if "###" not in second:
        return []
    
    try:
        _, bbox_str = second.split("###", 1)
        bbox = json.loads(bbox_str)
        if isinstance(bbox, list) and len(bbox) == 4:
            # 归一化 bbox
            normalized = normalize_bbox(bbox, width, height)
            return [normalized] if normalized else []
    except Exception:
        pass
    
    return []


def parse_question_answer(conversations: List[Dict]):
    """解析问题和答案"""
    question = ""
    answer = ""
    
    for conv in conversations:
        if conv.get("from") == "human" and not question:
            question = str(conv.get("value", "")).replace("<image>", "").strip()
        if conv.get("from") == "gpt":
            value = str(conv.get("value", "")).strip()
            # 跳过只包含 bbox 的回复
            if value.startswith("[") and value.endswith("]"):
                try:
                    json.loads(value)
                    continue
                except:
                    pass
            answer = value
    
    return question, answer


def load_metadata_index(metadata_dir: Path) -> Dict[str, Dict]:
    """
    加载 metadata 目录中的所有 jsonl 文件，建立索引
    返回: {dataset_name: {(image_name, question): metadata_entry}}
    """
    metadata_index = defaultdict(dict)
    
    if not metadata_dir.exists():
        return metadata_index
    
    for jsonl_file in metadata_dir.glob("*.jsonl"):
        dataset_name = jsonl_file.stem.replace("_cot_train", "").replace("_cot_val", "")
        print(f"Loading metadata from {jsonl_file.name}...")
        
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    image_name = entry.get("image", "")
                    question = entry.get("question", "")
                    if image_name and question:
                        key = (image_name, question[:100])  # 截断问题作为 key
                        metadata_index[dataset_name][key] = entry
                except Exception:
                    continue
    
    return metadata_index


def write_parquet_shards(rows_iter, output_dir: Path, shard_size: int, row_group_size: int):
    """分片写入 parquet 文件"""
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_idx = 0
    buffer = []
    total_rows = 0
    
    for row in rows_iter:
        buffer.append(row)
        if len(buffer) >= shard_size:
            table = pa.Table.from_pylist(buffer)
            shard_path = output_dir / f"viscot_lvr_{shard_idx:05d}.parquet"
            pq.write_table(table, shard_path, row_group_size=row_group_size)
            total_rows += len(buffer)
            shard_idx += 1
            buffer = []
    
    if buffer:
        table = pa.Table.from_pylist(buffer)
        shard_path = output_dir / f"viscot_lvr_{shard_idx:05d}.parquet"
        pq.write_table(table, shard_path, row_group_size=row_group_size)
        total_rows += len(buffer)
        shard_idx += 1
    
    return shard_idx, total_rows


def build_parquet_info(parquet_dir: Path, output_path: Path) -> Dict:
    """构建 parquet 文件信息"""
    info = {}
    total_rows = 0
    
    for parquet_file in sorted(parquet_dir.glob("*.parquet")):
        pf = pq.ParquetFile(parquet_file)
        info[str(parquet_file)] = {
            "num_row_groups": pf.num_row_groups,
            "num_rows": pf.metadata.num_rows,
        }
        total_rows += pf.metadata.num_rows
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(info, indent=2))
    
    return {"num_files": len(info), "total_rows": total_rows}


def update_dataset_info(dataset_info_path: Path, parquet_dir: str, info_path: str, num_files: int, total_rows: int):
    """更新 dataset_info.py 中的配置"""
    content = f'''# Auto-generated viscot_lvr dataset info
VISCOT_LVR_INFO = {{
    'data_dir': '{parquet_dir}',
    'num_files': {num_files},
    'num_total_samples': {total_rows},
    'parquet_info_path': '{info_path}',
}}
'''
    dataset_info_path.write_text(content)
    print(f"Updated {dataset_info_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Visual-CoT to parquet format with optimizations")
    parser.add_argument("--image_root", required=True, help="Root folder containing Visual-CoT images")
    parser.add_argument("--output_dir", required=True, help="Directory to write parquet shards")
    parser.add_argument("--info_path", required=True, help="Output parquet info json path")
    parser.add_argument("--shard_size", type=int, default=500, help="Number of samples per shard")
    parser.add_argument("--row_group_size", type=int, default=128, help="Row group size in parquet")
    parser.add_argument("--max_samples", type=int, default=0, help="Max samples to process (0 for all)")
    parser.add_argument("--viscot_json", required=True, help="Path to viscot_363k.json")
    parser.add_argument("--metadata_dir", default="", help="Path to metadata directory (optional)")
    parser.add_argument("--use_metadata_priority", action="store_true", 
                        help="Prefer metadata bbox over viscot_json bbox")
    args = parser.parse_args()
    
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    info_path = Path(args.info_path)
    viscot_json = Path(args.viscot_json)
    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else image_root / "metadata"
    
    # 加载 metadata 索引
    metadata_index = load_metadata_index(metadata_dir)
    print(f"Loaded metadata for {len(metadata_index)} datasets")
    
    # 统计信息
    stats = defaultdict(int)
    
    def row_generator():
        count = 0
        
        for ex in tqdm(iter_json_array(viscot_json), desc="Processing"):
            if args.max_samples and count >= args.max_samples:
                break
            
            conversations = ex.get("conversations") or []
            question, answer = parse_question_answer(conversations)
            
            if not question or not answer:
                stats["skipped_no_qa"] += 1
                continue
            
            image_field = ex.get("image", [])
            image_name = image_field[0] if isinstance(image_field, list) and image_field else ex.get("image")
            dataset_name = ex.get("dataset", "")
            
            if not image_name:
                stats["skipped_no_image"] += 1
                continue
            
            # 加载图像并获取原始尺寸
            image_bytes, orig_width, orig_height = load_image_bytes_and_size(
                image_root, image_name, dataset_name
            )
            
            if image_bytes is None:
                stats["skipped_image_not_found"] += 1
                continue
            
            # 解析 bbox - 优先使用对话中的归一化坐标
            bboxs = parse_bbox_from_conversation(conversations)
            
            if not bboxs:
                # 尝试从 image 字段解析并归一化
                bboxs = parse_bbox_from_image_field(image_field, orig_width, orig_height)
            
            # 尝试从 metadata 获取更多信息
            metadata_entry = None
            if dataset_name in metadata_index:
                # 从 image_name 提取文件名
                image_basename = Path(image_name).name
                key = (image_basename, question[:100])
                metadata_entry = metadata_index[dataset_name].get(key)
            
            # 如果 metadata 有更好的 bbox，使用它
            if metadata_entry and args.use_metadata_priority:
                meta_bboxs = metadata_entry.get("bboxs", [])
                meta_width = metadata_entry.get("width", orig_width)
                meta_height = metadata_entry.get("height", orig_height)
                
                if meta_bboxs:
                    # metadata 中的 bbox 通常是像素坐标，需要归一化
                    normalized_bboxs = []
                    for bbox in meta_bboxs:
                        normalized = normalize_bbox(bbox, meta_width, meta_height)
                        if normalized:
                            normalized_bboxs.append(normalized)
                    if normalized_bboxs:
                        bboxs = normalized_bboxs
            
            # 获取 full_answer（如果有）
            full_answer = answer
            if metadata_entry:
                full_answer = metadata_entry.get("full_answer") or metadata_entry.get("answer") or answer
            
            count += 1
            stats["processed"] += 1
            stats[f"dataset_{dataset_name}"] += 1
            
            if bboxs:
                stats["with_bbox"] += 1
            else:
                stats["without_bbox"] += 1
            
            yield {
                "image": image_bytes,
                "question": question,
                "answer": answer,
                "full_answer": full_answer,
                "bboxs": bboxs,
                "dataset": dataset_name,
            }
    
    # 写入 parquet
    print(f"\nConverting {viscot_json} to parquet...")
    num_shards, total_rows = write_parquet_shards(
        row_generator(), output_dir, args.shard_size, args.row_group_size
    )
    
    # 构建 parquet info
    info_stats = build_parquet_info(output_dir, info_path)
    
    # 打印统计信息
    print("\n" + "=" * 50)
    print("Conversion Statistics:")
    print("=" * 50)
    print(f"Total processed: {stats['processed']}")
    print(f"With bbox: {stats['with_bbox']}")
    print(f"Without bbox: {stats['without_bbox']}")
    print(f"Skipped (no Q&A): {stats['skipped_no_qa']}")
    print(f"Skipped (no image): {stats['skipped_no_image']}")
    print(f"Skipped (image not found): {stats['skipped_image_not_found']}")
    print(f"\nOutput: {num_shards} shards, {total_rows} total rows")
    print(f"Parquet info: {info_path}")
    
    print("\nPer-dataset statistics:")
    for key, value in sorted(stats.items()):
        if key.startswith("dataset_"):
            print(f"  {key[8:]}: {value}")
    
    # 生成配置更新建议
    print("\n" + "=" * 50)
    print("Please update data/dataset_info.py with:")
    print("=" * 50)
    print(f"""
'vlm_sft_parquet': {{
    'viscot_lvr': {{
        'data_dir': '{output_dir}',
        'num_files': {info_stats['num_files']},
        'num_total_samples': {info_stats['total_rows']},
        'parquet_info_path': '{info_path}',
    }},
}},
""")


if __name__ == "__main__":
    main()
