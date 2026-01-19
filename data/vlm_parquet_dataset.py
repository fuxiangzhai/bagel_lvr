# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import numpy as np
from PIL import Image

from .data_utils import pil_img2rgb
from .interleave_datasets.interleave_t2i_dataset import (
    InterleavedBaseIterableDataset,
    ParquetStandardIterableDataset,
)


class VLMParquetLVRIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):
    def _bbox_to_vit_indexes(self, image_tensor, bboxs):
        """
        将 bbox 坐标转换为 ViT patch 索引
        
        支持两种坐标格式：
        1. 归一化坐标 [0, 1] - 所有值都在 [0, 1] 范围内
        2. 像素坐标 - 至少有一个值 > 1.0
        
        注意：推荐使用归一化坐标，因为它们与图像 transform 无关
        """
        # 安全检查：处理 None、空列表和空数组
        if image_tensor is None:
            return []
        if bboxs is None:
            return []
        if isinstance(bboxs, np.ndarray):
            if bboxs.size == 0:
                return []
            bboxs = bboxs.tolist()  # 转换为 Python list
        elif isinstance(bboxs, list) and len(bboxs) == 0:
            return []
        
        _, img_h, img_w = image_tensor.shape
        patch_size = self.vit_transform.stride
        num_patches_w = max(img_w // patch_size, 1)
        num_patches_h = max(img_h // patch_size, 1)
        indexes = []
        
        for bbox in bboxs:
            # 安全处理 bbox：可能是 list、numpy array 或 None
            if bbox is None:
                continue
            if isinstance(bbox, np.ndarray):
                if bbox.size != 4:
                    continue
                bbox = bbox.tolist()
            elif not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            
            try:
                x1, y1, x2, y2 = [float(v) for v in bbox]
            except (ValueError, TypeError):
                continue
            
            # 判断坐标类型：如果所有值都在 [0, 1.05] 范围内，认为是归一化坐标
            # 使用 1.05 而不是 1.0 以容忍一些边界情况
            is_normalized = all(0 <= v <= 1.05 for v in [x1, y1, x2, y2])
            
            if is_normalized:
                # 归一化坐标 -> 像素坐标（相对于变换后的图像）
                cx = (x1 + x2) * 0.5 * img_w
                cy = (y1 + y2) * 0.5 * img_h
            else:
                # 像素坐标 - 警告：这些坐标可能来自原始图像，
                # 如果图像经过 resize，坐标可能不准确
                # 尝试检测是否需要缩放
                max_coord = max(x1, y1, x2, y2)
                if max_coord > max(img_w, img_h) * 2:
                    # 坐标远大于图像尺寸，可能是原始图像的像素坐标
                    # 尝试归一化（假设原始尺寸未知，使用坐标范围估算）
                    scale_x = img_w / max(x2, 1)
                    scale_y = img_h / max(y2, 1)
                    cx = (x1 + x2) * 0.5 * min(scale_x, 1.0)
                    cy = (y1 + y2) * 0.5 * min(scale_y, 1.0)
                else:
                    # 直接使用像素坐标
                    cx = (x1 + x2) * 0.5
                    cy = (y1 + y2) * 0.5
            
            # 转换为 patch 索引
            px = int(max(0, min(img_w - 1, cx)) // patch_size)
            py = int(max(0, min(img_h - 1, cy)) // patch_size)
            px = min(max(px, 0), num_patches_w - 1)
            py = min(max(py, 0), num_patches_h - 1)
            
            indexes.append(py * num_patches_w + px)
        
        return indexes

    def parse_row(self, row):
        question = row.get("question", "")
        answer = row.get("full_answer") or row.get("answer", "")
        
        # 安全获取 bboxs，处理 numpy 数组
        bboxs = row.get("bboxs")
        if bboxs is None:
            bboxs = row.get("bboxes")
        if bboxs is None:
            bboxs = []
        elif isinstance(bboxs, np.ndarray):
            bboxs = bboxs.tolist() if bboxs.size > 0 else []
        
        if not question or not answer:
            return {}

        try:
            image_bytes = row["image"]
            image = pil_img2rgb(Image.open(io.BytesIO(image_bytes)))
        except Exception:
            return {}

        data = self._init_data()
        data = self._add_text(
            data,
            f"<image>\n{question}",
            need_loss=False,
            enable_cfg=False,
        )

        vit_image_tensor = self.vit_transform(image)
        height, width = vit_image_tensor.shape[1:]
        data["num_tokens"] += width * height // self.vit_transform.stride ** 2
        data["image_tensor_list"].append(vit_image_tensor)
        data["sequence_plan"].append(
            {
                "type": "vit_image",
                "enable_cfg": 0,
                "loss": 0,
                "special_token_loss": 0,
                "special_token_label": None,
            }
        )

        data["sequence_plan"].append(
            {
                "type": "lvr_token",
                "enable_cfg": 0,
                "loss": 0,
                "special_token_loss": 0,
                "special_token_label": None,
                "split_start": True,
                "split_end": True,
            }
        )
        data["num_tokens"] += 1

        data = self._add_text(data, answer, need_loss=True, enable_cfg=False)
        data["lvr_local_vit_indexes"] = self._bbox_to_vit_indexes(vit_image_tensor, bboxs)
        return data
