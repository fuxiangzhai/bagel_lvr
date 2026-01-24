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
    def __init__(self, *args, num_lvr_tokens_per_bbox=None, fixed_num_lvr_tokens=None, 
                 use_latent_end_token=False, use_area_mode=False, **kwargs):
        """
        Args:
            num_lvr_tokens_per_bbox: 每个bbox生成的LVR token数量(默认为None，使用bbox数量)
            fixed_num_lvr_tokens: 固定的LVR token总数(忽略bbox数量，默认为None)
            use_latent_end_token: 是否使用可学习的latent end token
            use_area_mode: 是否使用区域模式计算bbox索引（借鉴LVR的实现）
                - False（默认）: 每个bbox返回一个中心点索引
                - True: 每个bbox返回覆盖区域内的所有patch索引
        """
        super().__init__(*args, **kwargs)
        self.num_lvr_tokens_per_bbox = num_lvr_tokens_per_bbox
        self.fixed_num_lvr_tokens = fixed_num_lvr_tokens
        self.use_latent_end_token = use_latent_end_token
        self.use_area_mode = use_area_mode
    
    def _bbox_to_vit_indexes(self, image_tensor, bboxs, use_area_mode=False):
        """
        将 bbox 坐标转换为 ViT patch 索引
        
        借鉴 LVR 的实现，支持两种模式：
        1. 中心点模式 (use_area_mode=False): 每个 bbox 返回一个中心点索引
        2. 区域模式 (use_area_mode=True): 每个 bbox 返回覆盖区域内的所有 patch 索引（类似 LVR）
        
        坐标格式要求：归一化坐标 [0, 1]
        
        Args:
            image_tensor: 图像张量 [C, H, W]
            bboxs: bbox 列表，每个 bbox 为 [x0, y0, x1, y1] 归一化坐标
            use_area_mode: 是否使用区域模式（返回 bbox 覆盖的所有 patch）
            
        Returns:
            如果 use_area_mode=False: 返回单个索引列表 [idx1, idx2, ...]
            如果 use_area_mode=True: 返回索引列表的列表 [[idx1, idx2], [idx3, idx4, idx5], ...]
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
        
        if use_area_mode:
            # 区域模式：借鉴 LVR 的实现，返回 bbox 覆盖的所有 patch
            token_idxs_list = []
            for bbox in bboxs:
                # 安全处理 bbox
                if bbox is None:
                    continue
                if isinstance(bbox, np.ndarray):
                    if bbox.size != 4:
                        continue
                    bbox = bbox.tolist()
                elif not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue
                
                try:
                    x0, y0, x1, y1 = [float(v) for v in bbox]
                except (ValueError, TypeError):
                    continue
                
                # 验证归一化坐标
                if not all(0 <= v <= 1.05 for v in [x0, y0, x1, y1]):
                    # 不是归一化坐标，跳过（可以添加警告日志）
                    continue
                
                # 将归一化坐标映射到 patch grid
                # 使用 floor 和 ceil 来确保覆盖整个 bbox 区域
                x0_patch = max(0, min(int(np.floor(x0 * num_patches_w)), num_patches_w - 1))
                x1_patch = max(0, min(int(np.ceil(x1 * num_patches_w)), num_patches_w))
                y0_patch = max(0, min(int(np.floor(y0 * num_patches_h)), num_patches_h - 1))
                y1_patch = max(0, min(int(np.ceil(y1 * num_patches_h)), num_patches_h))
                
                # 生成区域内所有 patch 的索引
                idxs = [
                    int(yy * num_patches_w + xx)
                    for yy in range(y0_patch, y1_patch)
                    for xx in range(x0_patch, x1_patch)
                ]
                
                # 即使是空列表也添加，保持索引对应关系
                token_idxs_list.append(idxs)
            
            return token_idxs_list
        else:
            # 中心点模式：原有逻辑，每个 bbox 返回一个中心点索引
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
                    x0, y0, x1, y1 = [float(v) for v in bbox]
                except (ValueError, TypeError):
                    continue
                
                # 判断坐标类型：如果所有值都在 [0, 1.05] 范围内，认为是归一化坐标
                is_normalized = all(0 <= v <= 1.05 for v in [x0, y0, x1, y1])
                
                if is_normalized:
                    # 归一化坐标 -> 像素坐标（相对于变换后的图像）
                    cx = (x0 + x1) * 0.5 * img_w
                    cy = (y0 + y1) * 0.5 * img_h
                else:
                    # 像素坐标 - 警告：这些坐标可能来自原始图像，
                    # 如果图像经过 resize，坐标可能不准确
                    # 尝试检测是否需要缩放
                    max_coord = max(x0, y0, x1, y1)
                    if max_coord > max(img_w, img_h) * 2:
                        # 坐标远大于图像尺寸，可能是原始图像的像素坐标
                        scale_x = img_w / max(x1, 1)
                        scale_y = img_h / max(y1, 1)
                        cx = (x0 + x1) * 0.5 * min(scale_x, 1.0)
                        cy = (y0 + y1) * 0.5 * min(scale_y, 1.0)
                    else:
                        # 直接使用像素坐标
                        cx = (x0 + x1) * 0.5
                        cy = (y0 + y1) * 0.5
                
                # 转换为 patch 索引
                px = int(max(0, min(img_w - 1, cx)) // patch_size)
                py = int(max(0, min(img_h - 1, cy)) // patch_size)
                px = min(max(px, 0), num_patches_w - 1)
                py = min(max(py, 0), num_patches_h - 1)
                
                indexes.append(py * num_patches_w + px)
            
            return indexes

    def parse_row(self, row):
        """
        解析 parquet 行数据，生成训练样本
        
        借鉴 LVR 的数据处理流程：
        1. 提取和验证基本字段（question, answer, bboxes）
        2. 加载和处理图像
        3. 构建序列：question + image + lvr_tokens + answer
        4. 计算 bbox 对应的 ViT patch 索引
        """
        # Step 1: 提取和验证数据
        question = row.get("question", "").strip()
        # 优先使用 full_answer，类似 LVR 的处理方式
        answer = (row.get("full_answer") or row.get("answer", "")).strip()
        
        # 安全获取 bboxs，处理多种数据格式
        bboxs = row.get("bboxs")
        if bboxs is None:
            bboxs = row.get("bboxes")  # 兼容不同命名
        if bboxs is None:
            bboxs = []
        elif isinstance(bboxs, np.ndarray):
            # 处理 numpy 数组
            bboxs = bboxs.tolist() if bboxs.size > 0 else []
        elif not isinstance(bboxs, list):
            # 确保是列表类型
            bboxs = []
        
        # 验证必需字段
        if not question or not answer:
            return {}

        # Step 2: 加载和处理图像
        try:
            image_bytes = row["image"]
            image = pil_img2rgb(Image.open(io.BytesIO(image_bytes)))
        except Exception as e:
            # 图像加载失败，返回空字典
            return {}

        # Step 3: 初始化数据并添加问题文本
        data = self._init_data()
        data = self._add_text(
            data,
            f"<image>\n{question}",
            need_loss=False,
            enable_cfg=False,
        )

        # Step 4: 处理图像并添加到序列
        vit_image_tensor = self.vit_transform(image)
        height, width = vit_image_tensor.shape[1:]
        num_image_tokens = width * height // self.vit_transform.stride ** 2
        
        data["num_tokens"] += num_image_tokens
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

        # Step 5: 计算 LVR token 数量
        # 借鉴 LVR 的灵活计算方式
        num_lvr_tokens = self._calculate_num_lvr_tokens(bboxs)
        
        # Step 6: 添加 LVR tokens 到序列
        # 格式：<|lvr_start|> + N个<|lvr|> + [可选<|lvr_latent_end|>] + <|lvr_end|>
        data["sequence_plan"].append(
            {
                "type": "lvr_token",
                "enable_cfg": 0,
                "loss": 0,
                "special_token_loss": 0,
                "special_token_label": None,
                "split_start": True,
                "split_end": True,
                "num_lvr_tokens": num_lvr_tokens,
                "use_latent_end_token": self.use_latent_end_token,
            }
        )
        
        # 更新 token 计数
        # 注意：这里只计算实际的LVR tokens数量，不包括特殊标记tokens（lvr_start, lvr_end, lvr_latent_end）
        # 特殊标记tokens会在dataset_base.py的estimated_extra_tokens中估算（已分配3个tokens的余量）
        data["num_tokens"] += num_lvr_tokens

        # Step 7: 添加答案文本（需要计算 loss）
        data = self._add_text(data, answer, need_loss=True, enable_cfg=False)
        
        # Step 8: 计算 bbox 对应的 ViT patch 索引
        # 根据配置选择中心点模式或区域模式
        data["lvr_local_vit_indexes"] = self._bbox_to_vit_indexes(
            vit_image_tensor, 
            bboxs, 
            use_area_mode=self.use_area_mode
        )
        
        return data
    
    def _calculate_num_lvr_tokens(self, bboxs):
        """
        计算需要的 LVR token 数量
        
        借鉴 LVR 的逻辑，支持三种模式：
        1. 固定数量模式：使用 fixed_num_lvr_tokens
        2. 按 bbox 数量计算：len(bboxs) * num_lvr_tokens_per_bbox
        3. 默认模式：每个 bbox 对应一个 token
        
        Args:
            bboxs: bbox 列表
            
        Returns:
            int: LVR token 数量
        """
        # 模式 1: 固定数量（优先级最高）
        if self.fixed_num_lvr_tokens is not None:
            return self.fixed_num_lvr_tokens
        
        # 模式 2 和 3: 根据 bbox 数量计算
        num_bboxs = len(bboxs) if bboxs else 0
        
        if num_bboxs > 0:
            # 有 bbox 的情况
            if self.num_lvr_tokens_per_bbox is not None:
                # 每个 bbox 对应指定数量的 token
                return num_bboxs * self.num_lvr_tokens_per_bbox
            else:
                # 默认：每个 bbox 对应一个 token
                return num_bboxs
        else:
            # 没有 bbox 的情况：使用默认值
            # 类似 LVR 的处理，至少生成一个 token
            return self.num_lvr_tokens_per_bbox if self.num_lvr_tokens_per_bbox else 1
