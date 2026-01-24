# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


import math
import random
from PIL import Image

import torch
from torch.nn.attention.flex_attention import or_masks, and_masks


def create_sparse_mask(document_lens, split_lens, attn_modes, device, block_size=128):
    """
    Create sparse attention mask for FlexAttention.
    
    IMPORTANT: FlexAttention's create_block_mask rounds up sequence length to 
    multiples of BLOCK_SIZE. The mask functions may receive indices beyond the 
    actual sequence length. We must handle this by:
    1. Padding tensors to aligned length, OR
    2. Using safe indexing with boundary checks
    
    We use approach 1 (padding) for better performance as it avoids conditional
    branching in the mask functions.
    """
    # 验证长度一致性
    split_lens_sum = sum(split_lens)
    document_lens_sum = sum(document_lens)
    if split_lens_sum != document_lens_sum:
        raise ValueError(
            f"Length mismatch in create_sparse_mask: sum(split_lens)={split_lens_sum} != sum(document_lens)={document_lens_sum}\n"
            f"split_lens={split_lens}\n"
            f"document_lens={document_lens}\n"
            f"attn_modes={attn_modes}"
        )
    
    seqlen = split_lens_sum
    # Calculate aligned length (rounded up to block_size)
    aligned_len = ((seqlen + block_size - 1) // block_size) * block_size
    pad_len = aligned_len - seqlen
    
    full_and_noise_tmp = []
    noise_tmp = []

    for i, (length, model) in enumerate(zip(split_lens, attn_modes)):
        value = i if model in ['full', 'noise'] else -1
        full_and_noise_tmp.extend([value] * length)
        value_noise = i if model == 'noise' else -1
        noise_tmp.extend([value_noise] * length)

    # Pad with -1 (invalid document) to handle out-of-bounds access from FlexAttention
    # When q_idx or kv_idx >= seqlen, they will access padded region with -1
    # This ensures:
    # - full_and_noise_mask returns False (since -1 < 0)
    # - remove_noise_mask returns True (since -1 indicates not a noise region)
    # - sample_mask returns False (since padded region has document_id=0, different from valid docs starting at 1)
    if pad_len > 0:
        full_and_noise_tmp.extend([-1] * pad_len)
        noise_tmp.extend([-1] * pad_len)

    full_and_noise_seq_id = torch.tensor(full_and_noise_tmp, dtype=torch.int64, device=device)
    noise_seq_id = torch.tensor(noise_tmp, dtype=torch.int64, device=device)

    # document_id uses 1-indexed document IDs, pad with 0 (invalid document)
    document_id_list = []
    for i, l in enumerate(document_lens, start=1):
        document_id_list.extend([i] * l)
    if pad_len > 0:
        document_id_list.extend([0] * pad_len)  # 0 means invalid/padding
    document_id = torch.tensor(document_id_list, dtype=torch.int64, device=device)

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def full_and_noise_mask(b, h, q_idx, kv_idx):
        return (full_and_noise_seq_id[q_idx] == full_and_noise_seq_id[kv_idx]) & (full_and_noise_seq_id[q_idx] >= 0)

    def remove_noise_mask(b, h, q_idx, kv_idx):
        return (~((noise_seq_id[kv_idx] >= 0) & (noise_seq_id[q_idx] != noise_seq_id[kv_idx])))

    def sample_mask(b, h, q_idx, kv_idx):
        return document_id[q_idx] == document_id[kv_idx]

    return and_masks(or_masks(causal_mask, full_and_noise_mask), remove_noise_mask, sample_mask)


def patchify(image, patch_size):
    p = patch_size
    c, h, w = image.shape
    assert h % p == 0 and w % p == 0
    image = image.reshape(c, h // p, p, w // p, p)
    image = torch.einsum("chpwq->hwpqc", image)
    image = image.reshape(-1, p**2 * c)
    return image


def get_flattened_position_ids_extrapolate(img_h, img_w, patch_size, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    coords_h = torch.arange(0, num_patches_h)
    coords_w = torch.arange(0, num_patches_w)
    pos_ids = (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()
    return pos_ids


def get_flattened_position_ids_interpolate(img_h, img_w, patch_size, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    boundaries = torch.arange(1 / max_num_patches_per_side, 1.0, 1 / max_num_patches_per_side)
    fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / num_patches_h)
    fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / num_patches_w)
    bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
    bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
    pos_ids = (bucket_coords_h[:, None] * max_num_patches_per_side + bucket_coords_w).flatten()
    return pos_ids


def prepare_attention_mask_per_sample(split_lens, attn_modes, device="cpu"):
    """
    nested_split_lens: A list of N lists of ints. Each int indicates the length of a split within 
        a sample, where each sample contains multiple splits with different attn modes.
    nested_attn_modes: whether to use full attn in each split.
    """
    sample_len = sum(split_lens)
    attention_mask = torch.zeros((sample_len, sample_len), dtype=torch.bool, device=device)

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        assert attn_mode in ['causal', 'full', 'noise']
        if attn_mode == "causal":
            attention_mask[csum:csum + s, csum:csum + s] = torch.ones((s, s), device=device).tril()
            attention_mask[csum:csum + s, :csum] = 1
        else:
            attention_mask[csum:csum + s, csum:csum + s] = torch.ones((s, s))
            attention_mask[csum:csum + s, :csum] = 1
        csum += s

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        if attn_mode == "noise":
            attention_mask[:, csum : csum + s] = torch.zeros((sample_len, s))
            attention_mask[csum : csum + s, csum : csum + s] = torch.ones((s, s))
        csum += s

    attention_mask = torch.zeros_like(attention_mask, dtype=torch.float).masked_fill_(
        ~attention_mask, float("-inf")
    )

    return attention_mask


def split_integer_exp_decay(S, ng_sample_decay=1.0):
    if ng_sample_decay == 1.0:
        N = random.randint(1, S)
    else:
        base = (1 - ng_sample_decay) / (1 - math.pow(ng_sample_decay, S))
        p = [base * math.pow(ng_sample_decay, i) for i in range(S)]
        N = random.choices(list(range(1, S + 1)), p, k=1)[0]
    cumsum = [0] + sorted(random.sample(range(1, S), N - 1)) + [S]
    result = [cumsum[i+1] - cumsum[i] for i in range(len(cumsum) - 1)]
    return result, cumsum


def pil_img2rgb(image):
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        image = image.convert("RGBA")
        white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
        white.paste(image, mask=image.split()[3])
        image = white
    else:
        image = image.convert("RGB")

    return image


def add_special_tokens(tokenizer):
    all_special_tokens = []
    for k, v in tokenizer.special_tokens_map.items():
        if isinstance(v, str):
            all_special_tokens.append(v)
        elif isinstance(v, list):
            all_special_tokens += v

    new_tokens = []

    if '<|im_start|>' not in all_special_tokens:
        new_tokens.append('<|im_start|>')

    if '<|im_end|>' not in all_special_tokens:
        new_tokens.append('<|im_end|>')

    if '<|vision_start|>' not in all_special_tokens:
        new_tokens.append('<|vision_start|>')

    if '<|vision_end|>' not in all_special_tokens:
        new_tokens.append('<|vision_end|>')

    lvr_tokens = [
        '<|lvr_start|>',
        '<|lvr|>',
        '<|lvr_latent_end|>',
        '<|lvr_end|>',
    ]
    anchor_tokens = [
        '<|anchor_start|>',
        '<|anchor_end|>',
        '<|sam_pad|>',
        '<|dino_pad|>',
        '<|depth_pad|>',
        '<|sd_pad|>',
        '<|intern_pad|>',
        '<|pidinet_pad|>',
        '<|siglip_pad|>',
        '<|metaclip_pad|>',
    ]
    for token in lvr_tokens + anchor_tokens:
        if token not in all_special_tokens:
            new_tokens.append(token)

    num_new_tokens = tokenizer.add_tokens(new_tokens)
    bos_token_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
    eos_token_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    start_of_image = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    end_of_image = tokenizer.convert_tokens_to_ids('<|vision_end|>')
    lvr_start_id = tokenizer.convert_tokens_to_ids('<|lvr_start|>')
    lvr_token_id = tokenizer.convert_tokens_to_ids('<|lvr|>')
    lvr_latent_end_id = tokenizer.convert_tokens_to_ids('<|lvr_latent_end|>')
    lvr_end_id = tokenizer.convert_tokens_to_ids('<|lvr_end|>')
    anchor_start_id = tokenizer.convert_tokens_to_ids('<|anchor_start|>')
    anchor_end_id = tokenizer.convert_tokens_to_ids('<|anchor_end|>')
    sam_token_id = tokenizer.convert_tokens_to_ids('<|sam_pad|>')
    dino_token_id = tokenizer.convert_tokens_to_ids('<|dino_pad|>')
    depth_token_id = tokenizer.convert_tokens_to_ids('<|depth_pad|>')
    sd_token_id = tokenizer.convert_tokens_to_ids('<|sd_pad|>')
    intern_token_id = tokenizer.convert_tokens_to_ids('<|intern_pad|>')
    pidinet_token_id = tokenizer.convert_tokens_to_ids('<|pidinet_pad|>')
    siglip_token_id = tokenizer.convert_tokens_to_ids('<|siglip_pad|>')
    metaclip_token_id = tokenizer.convert_tokens_to_ids('<|metaclip_pad|>')

    new_token_ids = dict(
        bos_token_id=bos_token_id, 
        eos_token_id=eos_token_id, 
        start_of_image=start_of_image, 
        end_of_image=end_of_image, 
        lvr_start_id=lvr_start_id,
        lvr_token_id=lvr_token_id,
        lvr_latent_end_id=lvr_latent_end_id,
        lvr_end_id=lvr_end_id,
        anchor_start_id=anchor_start_id,
        anchor_end_id=anchor_end_id,
        sam_token_id=sam_token_id,
        dino_token_id=dino_token_id,
        depth_token_id=depth_token_id,
        sd_token_id=sd_token_id,
        intern_token_id=intern_token_id,
        pidinet_token_id=pidinet_token_id,
        siglip_token_id=siglip_token_id,
        metaclip_token_id=metaclip_token_id,
    )

    return tokenizer, new_token_ids, num_new_tokens


def len2weight(x, loss_reduction='square'):
    if x == 0:
        return x
    if loss_reduction == 'token':
        return 1
    if loss_reduction == 'sample':
        return 1 / x
    if loss_reduction == 'square':
        return 1 / (x ** 0.5)
    raise NotImplementedError(loss_reduction)
