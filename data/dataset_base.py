# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


import random
import json

import numpy as np
import torch

from .data_utils import (
    get_flattened_position_ids_interpolate,
    get_flattened_position_ids_extrapolate, 
    len2weight,
    patchify, 
    prepare_attention_mask_per_sample, 
)
from .dataset_info import DATASET_INFO, DATASET_REGISTRY
from .transforms import ImageTransform
from .video_utils import FrameSampler


class DataConfig:
    def __init__(
        self, 
        grouped_datasets, 
        text_cond_dropout_prob=0.1,
        vit_cond_dropout_prob=0.4,
        vae_cond_dropout_prob=0.1,
        vae_image_downsample=16,
        max_latent_size=32,
        vit_patch_size=14,
        max_num_patch_per_side=70,
    ):
        self.grouped_datasets = grouped_datasets
        self.text_cond_dropout_prob = text_cond_dropout_prob
        self.vit_cond_dropout_prob = vit_cond_dropout_prob
        self.vit_patch_size = vit_patch_size
        self.max_num_patch_per_side = max_num_patch_per_side
        self.vae_cond_dropout_prob = vae_cond_dropout_prob
        self.vae_image_downsample = vae_image_downsample
        self.max_latent_size = max_latent_size


class PackedDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        data_config, 
        tokenizer, 
        special_tokens,
        local_rank, 
        world_size, 
        num_workers,
        expected_num_tokens=32768, 
        max_num_tokens_per_sample=16384,
        max_num_tokens=36864,
        prefer_buffer_before=16384,
        max_buffer_size=50,
        interpolate_pos=False,
        use_flex=False,
        data_status=None,
    ):
        super().__init__()
        self.expected_num_tokens = expected_num_tokens
        self.max_num_tokens_per_sample = max_num_tokens_per_sample
        self.prefer_buffer_before = prefer_buffer_before
        self.max_num_tokens = max_num_tokens
        self.max_buffer_size = max_buffer_size
        self.tokenizer = tokenizer
        self.local_rank = local_rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.use_flex = use_flex
        for k, v in special_tokens.items():
            setattr(self, k, v)

        grouped_datasets, is_mandatory, grouped_weights = self.build_datasets(
            data_config.grouped_datasets, data_status
        )
        self.grouped_datasets = grouped_datasets
        self.dataset_iters = [iter(dataset) for dataset in grouped_datasets]
        self.is_mandatory = is_mandatory
        self.grouped_weights = grouped_weights
        self.data_config = data_config
        self.interpolate_pos = interpolate_pos
        if self.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

    def build_datasets(self, datasets_metainfo, data_status):
        datasets = []
        is_mandatory = []
        grouped_weights = []
        for grouped_dataset_name, dataset_args in datasets_metainfo.items():
            is_mandatory.append(dataset_args.pop('is_mandatory', False))
            grouped_weights.append(dataset_args.pop('weight', 0.0))

            if 'frame_sampler_args' in dataset_args.keys():
                frame_sampler = FrameSampler(**dataset_args.pop('frame_sampler_args'))
                dataset_args['frame_sampler'] = frame_sampler
            if 'image_transform_args' in dataset_args.keys():
                transform = ImageTransform(**dataset_args.pop('image_transform_args'))
                dataset_args['transform'] = transform
            if 'vit_image_transform_args' in dataset_args.keys():
                vit_transform = ImageTransform(**dataset_args.pop('vit_image_transform_args'))
                dataset_args['vit_transform'] = vit_transform

            assert 'dataset_names' in dataset_args.keys()
            dataset_names = dataset_args.pop('dataset_names')
            dataset_args['data_dir_list'] = []
            for item in dataset_names:
                if self.local_rank == 0:
                    print(f'Preparing Dataset {grouped_dataset_name}/{item}')
                meta_info = DATASET_INFO[grouped_dataset_name][item]
                dataset_args['data_dir_list'].append(meta_info['data_dir'])

                if "parquet_info_path" in meta_info.keys():
                    if 'parquet_info' not in dataset_args.keys():
                        dataset_args['parquet_info'] = {}
                    with open(meta_info['parquet_info_path'], 'r') as f:
                        parquet_info = json.load(f)
                    dataset_args['parquet_info'].update(parquet_info)

                if 'json_dir' in meta_info.keys():
                    # parquet/tar with json
                    if 'json_dir_list' not in dataset_args.keys():
                        dataset_args['json_dir_list'] = [meta_info['json_dir']]
                    else:
                        dataset_args['json_dir_list'].append(meta_info['json_dir'])

                if 'jsonl_path' in meta_info.keys():
                    # jsonl with jpeg
                    if 'jsonl_path_list' not in dataset_args.keys():
                        dataset_args['jsonl_path_list'] = [meta_info['jsonl_path']]
                    else:
                        dataset_args['jsonl_path_list'].append(meta_info['jsonl_path'])

            resume_data_status = dataset_args.pop('resume_data_status', True)
            if data_status is not None and grouped_dataset_name in data_status.keys() and resume_data_status:
                data_status_per_group = data_status[grouped_dataset_name]
            else:
                data_status_per_group = None
            dataset = DATASET_REGISTRY[grouped_dataset_name](
                dataset_name=grouped_dataset_name,
                tokenizer=self.tokenizer,
                local_rank=self.local_rank,
                world_size=self.world_size,
                num_workers=self.num_workers,
                data_status=data_status_per_group,
                **dataset_args
            )
            datasets.append(dataset)

        return datasets, is_mandatory, grouped_weights

    def set_epoch(self, seed):
        for dataset in self.grouped_datasets:
            dataset.set_epoch(seed)

    def set_sequence_status(self):
        sequence_status = dict(
            curr                        = 0,
            sample_lens                 = list(),
            packed_position_ids         = list(),
            nested_attention_masks      = list(),
            split_lens                  = list(),
            attn_modes                  = list(),
            packed_text_ids             = list(), 
            packed_text_indexes         = list(),
            packed_label_ids            = list(),
            ce_loss_indexes             = list(),
            ce_loss_weights             = list(),
            vae_image_tensors           = list(), 
            packed_latent_position_ids  = list(),
            vae_latent_shapes           = list(), 
            packed_vae_token_indexes    = list(), 
            packed_timesteps            = list(), 
            mse_loss_indexes            = list(),
            packed_vit_tokens           = list(), 
            vit_token_seqlens           = list(),
            packed_vit_position_ids     = list(),
            packed_vit_token_indexes    = list(), 
            lvr_target_vit_indexes      = list(),
            # LVR mode switch training: track lvr_start positions and lvr token positions
            lvr_start_positions         = list(),  # positions of <|lvr_start|> tokens
            lvr_token_positions         = list(),  # positions of <|lvr|> tokens  
            lvr_end_positions           = list(),  # positions of <|lvr_end|> tokens
            lvr_latent_end_positions    = list(),  # positions for latent end prediction target
        )
        return sequence_status

    def to_tensor(self, sequence_status):
        actual_sequence_length = sum(sequence_status['sample_lens'])
        
        data = dict(
            sequence_length=actual_sequence_length,
            sample_lens=sequence_status['sample_lens'].copy(),  # 使用副本，避免修改原列表
            packed_text_ids=torch.tensor(sequence_status['packed_text_ids']),
            packed_text_indexes=torch.tensor(sequence_status['packed_text_indexes']),
            packed_position_ids=torch.tensor(sequence_status['packed_position_ids']),
        )
        
        if not self.use_flex:
            data['nested_attention_masks'] = sequence_status['nested_attention_masks']
        else:
            pad_len = self.max_num_tokens - actual_sequence_length
            
            # 验证split_lens和sample_lens的sum必须严格相等
            split_lens_sum = sum(sequence_status['split_lens'])
            sample_lens_sum = sum(sequence_status['sample_lens'])
            
            if split_lens_sum != sample_lens_sum:
                raise ValueError(
                    f"CRITICAL BUG: sum(split_lens)={split_lens_sum} != sum(sample_lens)={sample_lens_sum}\n"
                    f"split_lens={sequence_status['split_lens']}\n"
                    f"sample_lens={sequence_status['sample_lens']}\n"
                    f"attn_modes={sequence_status['attn_modes']}"
                )
            
            # 添加padding以对齐到max_num_tokens
            if pad_len > 0:
                data['split_lens'] = sequence_status['split_lens'] + [pad_len]
                data['attn_modes'] = sequence_status['attn_modes'] + ['causal']
                data['sample_lens'] = data['sample_lens'] + [pad_len]
                data['sequence_length'] = actual_sequence_length + pad_len
                
                # CRITICAL: 也需要 padding packed_position_ids，否则会导致 rotary embedding 维度不匹配
                # padding 部分使用 0 作为 position id（不会影响实际计算，因为 padding 部分不会被使用）
                last_position_id = sequence_status['packed_position_ids'][-1] if sequence_status['packed_position_ids'] else 0
                padding_position_ids = list(range(last_position_id + 1, last_position_id + 1 + pad_len))
                data['packed_position_ids'] = torch.tensor(sequence_status['packed_position_ids'] + padding_position_ids)
            else:
                # 实际长度已超过限制，不需要padding（理论上不应该走到这里）
                data['split_lens'] = sequence_status['split_lens']
                data['attn_modes'] = sequence_status['attn_modes']
                if pad_len < 0:
                    print(f"Warning: sequence length ({actual_sequence_length}) exceeds max_num_tokens ({self.max_num_tokens}) by {-pad_len} tokens")

        # if the model has a convnet vae (e.g., as visual tokenizer)
        if len(sequence_status['vae_image_tensors']) > 0:
            image_tensors = sequence_status.pop('vae_image_tensors')
            image_sizes = [item.shape for item in image_tensors]
            max_image_size = [max(item) for item in list(zip(*image_sizes))]
            padded_images = torch.zeros(size=(len(image_tensors), *max_image_size))
            for i, image_tensor in enumerate(image_tensors):
                padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2]] = image_tensor

            data['padded_images'] = padded_images
            data['patchified_vae_latent_shapes'] = sequence_status['vae_latent_shapes']
            data['packed_latent_position_ids'] = torch.cat(sequence_status['packed_latent_position_ids'], dim=0)
            data['packed_vae_token_indexes'] = torch.tensor(sequence_status['packed_vae_token_indexes'])

        # if the model has a vit (e.g., as visual tokenizer)
        if len(sequence_status['packed_vit_tokens']) > 0:
            data['packed_vit_tokens'] = torch.cat(sequence_status['packed_vit_tokens'], dim=0)
            data['packed_vit_position_ids'] = torch.cat(sequence_status['packed_vit_position_ids'], dim=0)
            data['packed_vit_token_indexes'] = torch.tensor(sequence_status['packed_vit_token_indexes'])
            data['vit_token_seqlens'] = torch.tensor(sequence_status['vit_token_seqlens'])
            if sequence_status['lvr_target_vit_indexes']:
                data['lvr_target_vit_indexes'] = torch.tensor(sequence_status['lvr_target_vit_indexes'])
            
            # LVR mode switch training data
            if sequence_status['lvr_start_positions']:
                data['lvr_start_positions'] = torch.tensor(sequence_status['lvr_start_positions'])
            if sequence_status['lvr_token_positions']:
                data['lvr_token_positions'] = torch.tensor(sequence_status['lvr_token_positions'])
            if sequence_status['lvr_end_positions']:
                data['lvr_end_positions'] = torch.tensor(sequence_status['lvr_end_positions'])
            if sequence_status['lvr_latent_end_positions']:
                data['lvr_latent_end_positions'] = torch.tensor(sequence_status['lvr_latent_end_positions'])

        # if the model is required to perform visual generation
        if len(sequence_status['packed_timesteps']) > 0:
            data['packed_timesteps'] = torch.tensor(sequence_status['packed_timesteps'])
            data['mse_loss_indexes'] = torch.tensor(sequence_status['mse_loss_indexes'])

        # if the model is required to perform text generation
        if len(sequence_status['packed_label_ids']) > 0:
            data['packed_label_ids'] = torch.tensor(sequence_status['packed_label_ids'])
            data['ce_loss_indexes'] = torch.tensor(sequence_status['ce_loss_indexes'])
            data['ce_loss_weights'] = torch.tensor(sequence_status['ce_loss_weights'])

        # 最终验证：确保所有长度一致性（仅对flex attention）
        if self.use_flex and 'split_lens' in data:
            final_seqlen = sum(data['sample_lens'])
            final_split_sum = sum(data['split_lens'])
            final_sequence_length = data['sequence_length']
            
            if final_seqlen != final_split_sum:
                raise ValueError(
                    f"FINAL CHECK FAILED: sum(sample_lens)={final_seqlen} != sum(split_lens)={final_split_sum}\n"
                    f"sample_lens={data['sample_lens']}\n"
                    f"split_lens={data['split_lens']}"
                )
            
            if final_sequence_length != final_seqlen:
                raise ValueError(
                    f"FINAL CHECK FAILED: sequence_length={final_sequence_length} != sum(sample_lens)={final_seqlen}\n"
                    f"sample_lens={data['sample_lens']}\n"
                    f"This will cause index out of bounds in FlexAttention!"
                )

        return data

    def estimate_sample_length(self, sample):
        """
        估算sample打包后的实际长度，包括特殊tokens
        
        Args:
            sample: 包含'num_tokens'和'sequence_plan'的字典
            
        Returns:
            估算的总token数量
        """
        estimated_extra_tokens = 0
        for plan_item in sample.get('sequence_plan', []):
            item_type = plan_item.get('type', '')
            if item_type == 'vit_image':
                # startofimage + endofimage
                estimated_extra_tokens += 2
            elif item_type == 'text':
                # bos token
                estimated_extra_tokens += 1
            elif item_type == 'lvr_token':
                # lvr_start + lvr_end + 可能的lvr_latent_end
                estimated_extra_tokens += 3
            elif item_type == 'vae_image':
                # startofimage + endofimage
                estimated_extra_tokens += 2
        
        return sample.get('num_tokens', 0) + estimated_extra_tokens

    def __iter__(self):
        total_weights = sum(self.grouped_weights)
        assert total_weights > 0.0
        group_cumprobs = [sum(self.grouped_weights[:i + 1]) / total_weights 
                          for i in range(len(self.grouped_weights))]
        sequence_status = self.set_sequence_status()
        batch_data_indexes = []

        buffer = []
        while True:
            # Ensure at least one sample from each group
            if sequence_status['curr'] == 0:
                mandatory_overflow = False
                for group_index, group_iter in enumerate(self.dataset_iters):
                    if self.is_mandatory[group_index]:
                        while True:
                            sample = next(group_iter)
                            # if a sample is too long, skip it
                            num_tokens = self.estimate_sample_length(sample)
                            if num_tokens < self.max_num_tokens_per_sample:
                                sequence_status = self.pack_sequence(sample, sequence_status)
                                batch_data_indexes.append(sample['data_indexes'])
                                # 检查实际长度是否超过限制
                                if sequence_status['curr'] > self.max_num_tokens:
                                    print(f"Warning: mandatory samples exceed max_num_tokens ({sequence_status['curr']} > {self.max_num_tokens}), restarting batch")
                                    mandatory_overflow = True
                                break
                            else:
                                print(f"skip a sample with length {num_tokens}")
                                continue
                    if mandatory_overflow:
                        break
                # 如果 mandatory 样本导致超长，重新开始
                if mandatory_overflow:
                    sequence_status = self.set_sequence_status()
                    batch_data_indexes = []
                    continue

            if sequence_status['curr'] < self.prefer_buffer_before and len(buffer) > 0:
                sample = buffer.pop(0)
                sample_from_buffer = True
            else:
                # sample normally across all groups
                n = random.random()
                group_index = 0
                for i, cumprob in enumerate(group_cumprobs):
                    if n < cumprob:
                        group_index = i
                        break
                sample = next(self.dataset_iters[group_index])
                sample_from_buffer = False

            # if a sample is too long, skip it
            num_tokens = self.estimate_sample_length(sample)
            if num_tokens > self.max_num_tokens_per_sample:
                print(f"skip a sample with length {num_tokens}")
                continue

            if sequence_status['curr'] + num_tokens > self.max_num_tokens:
                if len(buffer) < self.max_buffer_size and not sample_from_buffer:
                    buffer.append(sample)
                    continue
                else:
                    # 修复：yield当前batch后，将sample添加到新batch
                    if sequence_status['curr'] > 0:
                        print(f"Yielding data with length {sum(sequence_status['sample_lens'])}")
                        data = self.to_tensor(sequence_status)
                        data['batch_data_indexes'] = batch_data_indexes
                        yield data
                        sequence_status = self.set_sequence_status()
                        batch_data_indexes = []
                    
                    # 关键修复：检查单样本是否本身就超过max_num_tokens
                    # 如果是，跳过这个样本，避免索引越界
                    if num_tokens > self.max_num_tokens:
                        print(f"skip a sample with length {num_tokens} (exceeds max_num_tokens={self.max_num_tokens})")
                        continue

            sequence_status = self.pack_sequence(sample, sequence_status)
            batch_data_indexes.append(sample['data_indexes'])
            
            # 关键检查：pack_sequence后的实际长度可能超过估算值
            # 如果实际长度超过max_num_tokens，丢弃这个batch并重新开始
            actual_len = sequence_status['curr']
            if actual_len > self.max_num_tokens:
                print(f"Warning: actual sequence length ({actual_len}) exceeds max_num_tokens ({self.max_num_tokens}), discarding batch and restarting")
                sequence_status = self.set_sequence_status()
                batch_data_indexes = []
                continue

            if sequence_status['curr'] >= self.expected_num_tokens:
                data = self.to_tensor(sequence_status)
                data['batch_data_indexes'] = batch_data_indexes
                yield data
                sequence_status = self.set_sequence_status()
                batch_data_indexes = []

    def pack_sequence(self, sample, sequence_status):
        image_tensor_list = sample['image_tensor_list']
        text_ids_list = sample['text_ids_list']
        sequence_plan = sample['sequence_plan']
        pending_lvr_local_vit_indexes = list(sample.get('lvr_local_vit_indexes', []))

        split_lens, attn_modes = list(), list()
        curr = sequence_status['curr']
        curr_rope_id = 0
        sample_lens = 0
        vit_token_cursor = sum(sequence_status['vit_token_seqlens'])
        lvr_targets_mapped = False

        for item in sequence_plan:
            split_start = item.get('split_start', True)
            if split_start:
                curr_split_len = 0

            if item['type'] == 'lvr_token':
                lvr_token_id = getattr(self, "lvr_token_id", None)
                lvr_start_id = getattr(self, "lvr_start_id", None)
                lvr_end_id = getattr(self, "lvr_end_id", None)
                lvr_latent_end_id = getattr(self, "lvr_latent_end_id", None)
                if lvr_token_id is None:
                    continue
                
                # 获取LVR token数量和配置
                num_lvr_tokens = item.get('num_lvr_tokens', 1)
                use_latent_end_token = item.get('use_latent_end_token', False)
                
                # Add <|lvr_start|> token first (this is where the model enters LVR mode)
                if lvr_start_id is not None:
                    sequence_status['packed_text_ids'].append(lvr_start_id)
                    sequence_status['packed_text_indexes'].append(curr)
                    sequence_status['lvr_start_positions'].append(curr)
                    curr += 1
                    curr_split_len += 1
                
                # Add N <|lvr|> tokens (the actual latent reasoning positions)
                for _ in range(num_lvr_tokens):
                    sequence_status['packed_text_ids'].append(lvr_token_id)
                    sequence_status['packed_text_indexes'].append(curr)
                    sequence_status['lvr_token_positions'].append(curr)
                    curr += 1
                    curr_split_len += 1
                
                # Add <|lvr_latent_end|> token (for learning when to exit LVR mode)
                # 只在最后一个lvr token后添加
                if use_latent_end_token and lvr_latent_end_id is not None:
                    sequence_status['packed_text_ids'].append(lvr_latent_end_id)
                    sequence_status['packed_text_indexes'].append(curr)
                    sequence_status['lvr_latent_end_positions'].append(curr)
                    curr += 1
                    curr_split_len += 1
                
                # Add <|lvr_end|> token (this is where the model exits LVR mode)
                if lvr_end_id is not None:
                    sequence_status['packed_text_ids'].append(lvr_end_id)
                    sequence_status['packed_text_indexes'].append(curr)
                    sequence_status['lvr_end_positions'].append(curr)
                    curr += 1
                    curr_split_len += 1

                attn_modes.append("causal")
                sequence_status['packed_position_ids'].extend(range(curr_rope_id, curr_rope_id + curr_split_len))
                curr_rope_id += curr_split_len

            elif item['type'] == 'text':
                text_ids = text_ids_list.pop(0)
                if item['enable_cfg'] == 1 and random.random() < self.data_config.text_cond_dropout_prob:
                    continue

                shifted_text_ids = [self.bos_token_id] + text_ids
                sequence_status['packed_text_ids'].extend(shifted_text_ids)
                sequence_status['packed_text_indexes'].extend(range(curr, curr + len(shifted_text_ids)))
                if item['loss'] == 1:
                    sequence_status['ce_loss_indexes'].extend(range(curr, curr + len(shifted_text_ids)))
                    sequence_status['ce_loss_weights'].extend(
                        [len2weight(len(shifted_text_ids))] * len(shifted_text_ids)
                    )
                    sequence_status['packed_label_ids'].extend(text_ids + [self.eos_token_id])
                curr += len(shifted_text_ids)
                curr_split_len += len(shifted_text_ids)

                # add a <|im_end|> token
                sequence_status['packed_text_ids'].append(self.eos_token_id)
                sequence_status['packed_text_indexes'].append(curr)
                if item['special_token_loss'] == 1: # <|im_end|> may have loss
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1

                # update sequence status
                attn_modes.append("causal")
                sequence_status['packed_position_ids'].extend(range(curr_rope_id, curr_rope_id + curr_split_len))
                curr_rope_id += curr_split_len

            elif item['type'] == 'vit_image':
                image_tensor = image_tensor_list.pop(0)
                if item['enable_cfg'] == 1 and random.random() < self.data_config.vit_cond_dropout_prob:
                    curr_rope_id += 1
                    continue

                # add a <|startofimage|> token
                sequence_status['packed_text_ids'].append(self.start_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                curr += 1
                curr_split_len += 1

                # preprocess image
                vit_tokens = patchify(image_tensor, self.data_config.vit_patch_size)
                num_img_tokens = vit_tokens.shape[0]
                image_vit_offset = vit_token_cursor
                sequence_status['packed_vit_token_indexes'].extend(range(curr, curr + num_img_tokens))
                curr += num_img_tokens
                curr_split_len += num_img_tokens
                vit_token_cursor += num_img_tokens

                sequence_status['packed_vit_tokens'].append(vit_tokens)
                sequence_status['vit_token_seqlens'].append(num_img_tokens)
                sequence_status['packed_vit_position_ids'].append(
                    self.get_flattened_position_ids(
                        image_tensor.size(1), image_tensor.size(2),
                        self.data_config.vit_patch_size, 
                        max_num_patches_per_side=self.data_config.max_num_patch_per_side
                    )
                )
                if pending_lvr_local_vit_indexes and not lvr_targets_mapped:
                    for local_index in pending_lvr_local_vit_indexes:
                        if local_index is None:
                            continue
                        bounded_index = min(max(int(local_index), 0), num_img_tokens - 1)
                        sequence_status['lvr_target_vit_indexes'].append(
                            image_vit_offset + bounded_index
                        )
                    lvr_targets_mapped = True

                # add a <|endofimage|> token
                sequence_status['packed_text_ids'].append(self.end_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                if item['special_token_loss'] == 1: # <|endofimage|> may have loss
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1

                # update sequence status
                attn_modes.append("full")
                sequence_status['packed_position_ids'].extend([curr_rope_id] * curr_split_len)
                curr_rope_id += 1

            elif item['type'] == 'vae_image':
                image_tensor = image_tensor_list.pop(0)
                if item['enable_cfg'] == 1 and random.random() < self.data_config.vae_cond_dropout_prob:
                    # FIXME fix vae dropout in video2video setting.
                    curr_rope_id += 1
                    continue

                # add a <|startofimage|> token
                sequence_status['packed_text_ids'].append(self.start_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                curr += 1
                curr_split_len += 1

                # preprocess image
                sequence_status['vae_image_tensors'].append(image_tensor)
                sequence_status['packed_latent_position_ids'].append(
                    self.get_flattened_position_ids(
                        image_tensor.size(1), image_tensor.size(2),
                        self.data_config.vae_image_downsample, 
                        max_num_patches_per_side=self.data_config.max_latent_size
                    )
                )
                H, W = image_tensor.shape[1:]
                h = H // self.data_config.vae_image_downsample
                w = W // self.data_config.vae_image_downsample
                sequence_status['vae_latent_shapes'].append((h, w))

                num_img_tokens = w * h
                sequence_status['packed_vae_token_indexes'].extend(range(curr, curr + num_img_tokens))
                if item['loss'] == 1:
                    sequence_status['mse_loss_indexes'].extend(range(curr, curr + num_img_tokens))
                    if split_start:
                        timestep = np.random.randn()
                else:
                    timestep = float('-inf')

                sequence_status['packed_timesteps'].extend([timestep] * num_img_tokens)
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                # add a <|endofimage|> token
                sequence_status['packed_text_ids'].append(self.end_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                # <|endofimage|> may have loss
                if item['special_token_loss'] == 1:
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1

                # update sequence status
                # 注意：attn_modes只在split_start时添加，因为一个split可能包含多个items
                # 必须确保每个有split_start的item最终都有对应的split_end
                if split_start:
                    if item['loss'] == 1 and 'frame_delta' not in item.keys():
                        attn_modes.append("noise")
                    else:
                        attn_modes.append("full")
                sequence_status['packed_position_ids'].extend([curr_rope_id] * (num_img_tokens + 2))
                if 'frame_delta' in item.keys():
                    curr_rope_id += item['frame_delta']
                elif item['loss'] == 0:
                    curr_rope_id += 1

            if item.get('split_end', True):
                split_lens.append(curr_split_len)
                sample_lens += curr_split_len

        sequence_status['curr'] = curr
        sequence_status['sample_lens'].append(sample_lens)
        # prepare attention mask
        if not self.use_flex:
            sequence_status['nested_attention_masks'].append(
                prepare_attention_mask_per_sample(split_lens, attn_modes)
            )
        else:
            sequence_status['split_lens'].extend(split_lens)
            sequence_status['attn_modes'].extend(attn_modes)

        return sequence_status


class SimpleCustomBatch:
    def __init__(self, batch):
        data = batch[0]
        self.batch_data_indexes = data['batch_data_indexes']
        self.sequence_length = data["sequence_length"]
        self.sample_lens = data["sample_lens"]
        self.packed_text_ids = data["packed_text_ids"]
        self.packed_text_indexes = data["packed_text_indexes"]
        self.packed_position_ids = data["packed_position_ids"]

        self.use_flex = "nested_attention_masks" not in data.keys()

        if self.use_flex:
            self.split_lens = data["split_lens"]
            self.attn_modes = data["attn_modes"]
        else:
            self.nested_attention_masks = data["nested_attention_masks"]

        if "padded_images" in data.keys():
            self.padded_images = data["padded_images"]
            self.patchified_vae_latent_shapes = data["patchified_vae_latent_shapes"]
            self.packed_latent_position_ids = data["packed_latent_position_ids"]
            self.packed_vae_token_indexes = data["packed_vae_token_indexes"]

        if "packed_vit_tokens" in data.keys():
            self.packed_vit_tokens = data["packed_vit_tokens"]
            self.packed_vit_position_ids = data["packed_vit_position_ids"]
            self.packed_vit_token_indexes = data["packed_vit_token_indexes"]
            self.vit_token_seqlens = data["vit_token_seqlens"]
            if "lvr_target_vit_indexes" in data.keys():
                self.lvr_target_vit_indexes = data["lvr_target_vit_indexes"]
            
            # LVR mode switch training data
            if "lvr_start_positions" in data.keys():
                self.lvr_start_positions = data["lvr_start_positions"]
            if "lvr_token_positions" in data.keys():
                self.lvr_token_positions = data["lvr_token_positions"]
            if "lvr_end_positions" in data.keys():
                self.lvr_end_positions = data["lvr_end_positions"]
            if "lvr_latent_end_positions" in data.keys():
                self.lvr_latent_end_positions = data["lvr_latent_end_positions"]

        if "packed_timesteps" in data.keys():
            self.packed_timesteps = data["packed_timesteps"]
            self.mse_loss_indexes = data["mse_loss_indexes"]

        if "packed_label_ids" in data.keys():
            self.packed_label_ids = data["packed_label_ids"]
            self.ce_loss_indexes = data["ce_loss_indexes"]
            self.ce_loss_weights = data["ce_loss_weights"]

    def pin_memory(self):
        self.packed_text_ids = self.packed_text_ids.pin_memory()
        self.packed_text_indexes = self.packed_text_indexes.pin_memory()
        self.packed_position_ids = self.packed_position_ids.pin_memory()

        if not self.use_flex:
            self.nested_attention_masks = [item.pin_memory() for item in self.nested_attention_masks]

        if hasattr(self, 'padded_images'):
            self.padded_images = self.padded_images.pin_memory()
            self.packed_vae_token_indexes = self.packed_vae_token_indexes.pin_memory()
            self.packed_latent_position_ids = self.packed_latent_position_ids.pin_memory()

        if hasattr(self, 'packed_timesteps'):
            self.packed_timesteps = self.packed_timesteps.pin_memory()
            self.mse_loss_indexes = self.mse_loss_indexes.pin_memory()

        if hasattr(self, 'packed_vit_tokens'):
            self.packed_vit_tokens = self.packed_vit_tokens.pin_memory()
            self.packed_vit_position_ids = self.packed_vit_position_ids.pin_memory()
            self.packed_vit_token_indexes = self.packed_vit_token_indexes.pin_memory()
            self.vit_token_seqlens = self.vit_token_seqlens.pin_memory()
            if hasattr(self, 'lvr_target_vit_indexes'):
                self.lvr_target_vit_indexes = self.lvr_target_vit_indexes.pin_memory()
            
            # LVR mode switch training data
            if hasattr(self, 'lvr_start_positions'):
                self.lvr_start_positions = self.lvr_start_positions.pin_memory()
            if hasattr(self, 'lvr_token_positions'):
                self.lvr_token_positions = self.lvr_token_positions.pin_memory()
            if hasattr(self, 'lvr_end_positions'):
                self.lvr_end_positions = self.lvr_end_positions.pin_memory()
            if hasattr(self, 'lvr_latent_end_positions'):
                self.lvr_latent_end_positions = self.lvr_latent_end_positions.pin_memory()

        if hasattr(self, 'packed_label_ids'):
            self.packed_label_ids = self.packed_label_ids.pin_memory()
            self.ce_loss_indexes = self.ce_loss_indexes.pin_memory()
            self.ce_loss_weights = self.ce_loss_weights.pin_memory()

        return self

    def cuda(self, device):
        self.packed_text_ids = self.packed_text_ids.to(device)
        self.packed_text_indexes = self.packed_text_indexes.to(device)
        self.packed_position_ids = self.packed_position_ids.to(device)

        if not self.use_flex:
            self.nested_attention_masks = [item.to(device) for item in self.nested_attention_masks]

        if hasattr(self, 'padded_images'):
            self.padded_images = self.padded_images.to(device)
            self.packed_vae_token_indexes = self.packed_vae_token_indexes.to(device)
            self.packed_latent_position_ids = self.packed_latent_position_ids.to(device)

        if hasattr(self, 'packed_timesteps'):
            self.packed_timesteps = self.packed_timesteps.to(device)
            self.mse_loss_indexes = self.mse_loss_indexes.to(device)

        if hasattr(self, 'packed_vit_tokens'):
            self.packed_vit_tokens = self.packed_vit_tokens.to(device)
            self.packed_vit_position_ids = self.packed_vit_position_ids.to(device)
            self.packed_vit_token_indexes = self.packed_vit_token_indexes.to(device)
            self.vit_token_seqlens = self.vit_token_seqlens.to(device)
            if hasattr(self, 'lvr_target_vit_indexes'):
                self.lvr_target_vit_indexes = self.lvr_target_vit_indexes.to(device)
            
            # LVR mode switch training data
            if hasattr(self, 'lvr_start_positions'):
                self.lvr_start_positions = self.lvr_start_positions.to(device)
            if hasattr(self, 'lvr_token_positions'):
                self.lvr_token_positions = self.lvr_token_positions.to(device)
            if hasattr(self, 'lvr_end_positions'):
                self.lvr_end_positions = self.lvr_end_positions.to(device)
            if hasattr(self, 'lvr_latent_end_positions'):
                self.lvr_latent_end_positions = self.lvr_latent_end_positions.to(device)

        if hasattr(self, 'packed_label_ids'):
            self.packed_label_ids = self.packed_label_ids.to(device)
            self.ce_loss_indexes = self.ce_loss_indexes.to(device)
            self.ce_loss_weights = self.ce_loss_weights.to(device)

        return self

    def to_dict(self):
        data = dict(
            sequence_length = self.sequence_length,
            sample_lens = self.sample_lens,
            packed_text_ids = self.packed_text_ids,
            packed_text_indexes = self.packed_text_indexes,
            packed_position_ids = self.packed_position_ids,
            batch_data_indexes = self.batch_data_indexes,
        )

        if not self.use_flex:
            data['nested_attention_masks'] = self.nested_attention_masks
        else:
            data['split_lens'] = self.split_lens
            data['attn_modes'] = self.attn_modes

        if hasattr(self, 'padded_images'):
            data['padded_images'] = self.padded_images
            data['patchified_vae_latent_shapes'] = self.patchified_vae_latent_shapes
            data['packed_latent_position_ids'] = self.packed_latent_position_ids
            data['packed_vae_token_indexes'] = self.packed_vae_token_indexes

        if hasattr(self, 'packed_vit_tokens'):
            data['packed_vit_tokens'] = self.packed_vit_tokens
            data['packed_vit_position_ids'] = self.packed_vit_position_ids
            data['packed_vit_token_indexes'] = self.packed_vit_token_indexes
            data['vit_token_seqlens'] = self.vit_token_seqlens
            if hasattr(self, 'lvr_target_vit_indexes'):
                data['lvr_target_vit_indexes'] = self.lvr_target_vit_indexes
            
            # LVR mode switch training data
            if hasattr(self, 'lvr_start_positions'):
                data['lvr_start_positions'] = self.lvr_start_positions
            if hasattr(self, 'lvr_token_positions'):
                data['lvr_token_positions'] = self.lvr_token_positions
            if hasattr(self, 'lvr_end_positions'):
                data['lvr_end_positions'] = self.lvr_end_positions
            if hasattr(self, 'lvr_latent_end_positions'):
                data['lvr_latent_end_positions'] = self.lvr_latent_end_positions

        if hasattr(self, 'packed_timesteps'):
            data['packed_timesteps'] = self.packed_timesteps
            data['mse_loss_indexes'] = self.mse_loss_indexes

        if hasattr(self, 'packed_label_ids'):
            data['packed_label_ids'] = self.packed_label_ids
            data['ce_loss_indexes'] = self.ce_loss_indexes
            data['ce_loss_weights'] = self.ce_loss_weights

        return data


def collate_wrapper():
    def collate_fn(batch):
        return SimpleCustomBatch(batch)
    return collate_fn
