#!/usr/bin/env python3
"""
验证 Visual-CoT 数据预处理结果的脚本

功能：
1. 检查 parquet 文件是否存在
2. 验证数据格式是否正确
3. 统计有/无 bbox 的样本数
4. 检查配置文件是否一致
"""

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq


def check_parquet_files(parquet_dir: Path) -> dict:
    """检查 parquet 文件"""
    if not parquet_dir.exists():
        return {"error": f"Parquet directory does not exist: {parquet_dir}"}
    
    parquet_files = list(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        return {"error": f"No parquet files found in {parquet_dir}"}
    
    total_rows = 0
    with_bbox = 0
    without_bbox = 0
    sample_data = None
    
    for pf_path in sorted(parquet_files)[:5]:  # 只检查前 5 个文件
        pf = pq.ParquetFile(pf_path)
        df = pf.read().to_pandas()
        total_rows += len(df)
        
        for _, row in df.iterrows():
            bboxs = row.get("bboxs") or row.get("bboxes") or []
            if bboxs and len(bboxs) > 0:
                with_bbox += 1
                if sample_data is None:
                    # 保存一个有 bbox 的样本作为示例
                    sample_data = {
                        "question": row.get("question", "")[:100],
                        "answer": row.get("answer", "")[:100],
                        "bboxs": bboxs[:2],  # 只显示前 2 个
                        "has_image": row.get("image") is not None,
                    }
            else:
                without_bbox += 1
    
    # 估算总数
    all_files_count = len(parquet_files)
    checked_files = min(5, all_files_count)
    estimated_total = total_rows * all_files_count // checked_files if checked_files > 0 else 0
    estimated_with_bbox = with_bbox * all_files_count // checked_files if checked_files > 0 else 0
    
    return {
        "num_files": all_files_count,
        "checked_files": checked_files,
        "checked_rows": total_rows,
        "estimated_total_rows": estimated_total,
        "checked_with_bbox": with_bbox,
        "checked_without_bbox": without_bbox,
        "estimated_with_bbox": estimated_with_bbox,
        "bbox_ratio": f"{with_bbox / total_rows * 100:.1f}%" if total_rows > 0 else "N/A",
        "sample_data": sample_data,
    }


def check_config_consistency(
    yaml_path: Path, 
    dataset_info_path: Path, 
    parquet_info_path: Path,
    num_parquet_files: int
) -> dict:
    """检查配置文件一致性"""
    issues = []
    
    # 检查 YAML 配置
    if yaml_path.exists():
        import yaml
        with open(yaml_path) as f:
            yaml_config = yaml.safe_load(f)
        
        num_used_data = yaml_config.get("vlm_sft_parquet", {}).get("num_used_data", [])
        if num_used_data:
            yaml_num = num_used_data[0]
            if yaml_num != num_parquet_files:
                issues.append(
                    f"YAML num_used_data ({yaml_num}) != actual parquet files ({num_parquet_files})"
                )
    else:
        issues.append(f"YAML config not found: {yaml_path}")
    
    # 检查 parquet info
    if parquet_info_path.exists():
        with open(parquet_info_path) as f:
            parquet_info = json.load(f)
        info_files_count = len(parquet_info)
        if info_files_count != num_parquet_files:
            issues.append(
                f"Parquet info ({info_files_count} files) != actual files ({num_parquet_files})"
            )
    else:
        issues.append(f"Parquet info not found: {parquet_info_path}")
    
    # 检查 dataset_info.py（简单检查）
    if dataset_info_path.exists():
        content = dataset_info_path.read_text()
        if f"'num_files': {num_parquet_files}" not in content:
            issues.append(
                f"dataset_info.py num_files may not match actual files ({num_parquet_files})"
            )
    
    return {
        "consistent": len(issues) == 0,
        "issues": issues,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate Visual-CoT data preparation")
    parser.add_argument(
        "--data_root", 
        default="/hpc2hdd/home/fzhai598/project/Bagel/Visual-CoT",
        help="Visual-CoT data root"
    )
    parser.add_argument(
        "--config_root",
        default="/hpc2hdd/home/fzhai598/project/Bagel",
        help="Bagel project root"
    )
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    config_root = Path(args.config_root)
    
    print("=" * 60)
    print("Visual-CoT Data Validation")
    print("=" * 60)
    
    # 1. 检查 parquet 文件
    print("\n[1] Checking parquet files...")
    parquet_dir = data_root / "viscot_parquet"
    parquet_result = check_parquet_files(parquet_dir)
    
    if "error" in parquet_result:
        print(f"  ERROR: {parquet_result['error']}")
        print("\n  Please run data preparation first:")
        print("    bash scripts/prepare_viscot_stage1_v2.sh")
        sys.exit(1)
    
    print(f"  Parquet files: {parquet_result['num_files']}")
    print(f"  Estimated total rows: {parquet_result['estimated_total_rows']}")
    print(f"  Checked rows (from {parquet_result['checked_files']} files): {parquet_result['checked_rows']}")
    print(f"  With bbox: {parquet_result['checked_with_bbox']} ({parquet_result['bbox_ratio']})")
    print(f"  Without bbox: {parquet_result['checked_without_bbox']}")
    
    if parquet_result.get("sample_data"):
        print("\n  Sample data with bbox:")
        sample = parquet_result["sample_data"]
        print(f"    Question: {sample['question']}...")
        print(f"    Answer: {sample['answer']}...")
        print(f"    Bboxs: {sample['bboxs']}")
    
    # 检查 bbox 比例
    if parquet_result['checked_rows'] > 0:
        bbox_ratio = parquet_result['checked_with_bbox'] / parquet_result['checked_rows']
        if bbox_ratio < 0.1:
            print(f"\n  WARNING: Only {bbox_ratio*100:.1f}% samples have bbox!")
            print("  This means LVR supervision will be very sparse.")
        elif bbox_ratio < 0.5:
            print(f"\n  NOTE: {bbox_ratio*100:.1f}% samples have bbox.")
            print("  Consider whether this is expected for your dataset.")
    
    # 2. 检查配置一致性
    print("\n[2] Checking configuration consistency...")
    config_result = check_config_consistency(
        yaml_path=config_root / "data/configs/viscot_lvr_stage1.yaml",
        dataset_info_path=config_root / "data/dataset_info.py",
        parquet_info_path=data_root / "viscot_parquet_info/viscot_lvr_info.json",
        num_parquet_files=parquet_result["num_files"],
    )
    
    if config_result["consistent"]:
        print("  All configurations are consistent!")
    else:
        print("  Configuration issues found:")
        for issue in config_result["issues"]:
            print(f"    - {issue}")
        print("\n  Please update configurations to match the actual data.")
    
    # 3. 验证可训练性
    print("\n[3] Training readiness check...")
    ready = True
    
    if parquet_result["num_files"] == 0:
        print("  ERROR: No parquet files found")
        ready = False
    
    if parquet_result['checked_rows'] > 0 and parquet_result['checked_with_bbox'] == 0:
        print("  ERROR: No samples with bbox found - LVR training will not work!")
        ready = False
    
    if not config_result["consistent"]:
        print("  WARNING: Configuration inconsistencies may cause issues")
    
    if ready:
        print("  Data is ready for training!")
        print("\n  Next steps:")
        print("    1. Fix any configuration issues above (if any)")
        print("    2. Run training: bash scripts/train_stage1_lvr_viscot.sh")
    else:
        print("\n  Please fix the issues above before training.")
        sys.exit(1)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
