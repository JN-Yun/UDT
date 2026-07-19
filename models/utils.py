import os
import torch.distributed as dist
from typing import Callable, Tuple, Any
import torch
import textwrap

def build_token_schedule(
    input_size,
    patch_size,
    enc_depth,
    num_skip,
    skip_interval,
    num_seg,
):
    initial_T = (input_size // patch_size) ** 2

    r_schedule = [0] * enc_depth
    is_skip_layer = [False] * enc_depth

    T_current = initial_T
    start_standard_idx = num_skip

    # patch_size == 1 special handling
    if patch_size == 1 or initial_T == 1024:
        idx1 = start_standard_idx
        idx2 = start_standard_idx + 1

        if idx1 < enc_depth:
            is_skip_layer[idx1] = True
            r_schedule[idx1] = T_current // 2
            T_current -= r_schedule[idx1]

        if idx2 < enc_depth:
            is_skip_layer[idx2] = True
            r_schedule[idx2] = T_current // 2
            T_current -= r_schedule[idx2]

        start_standard_idx += 2

    # standard skip layers
    standard_skip_indices = []

    for i in range(start_standard_idx, enc_depth):
        if (
            skip_interval > 0
            and (i - start_standard_idx) % skip_interval == 0
        ):
            if i == enc_depth - 1:
                continue

            is_skip_layer[i] = True
            standard_skip_indices.append(i)

    num_active_skips = sum(is_skip_layer)
    skip_indices = [i for i, v in enumerate(is_skip_layer) if v]

    # distribute merge ratio
    if len(standard_skip_indices) > 0:
        total_merge = max(0, T_current - num_seg)

        n = total_merge // len(standard_skip_indices)
        rem = total_merge % len(standard_skip_indices)

        for i in standard_skip_indices:
            r_schedule[i] = n + (1 if rem > 0 else 0)

            if rem > 0:
                rem -= 1

    # token schedule
    token_schedule = []

    test_T = initial_T
    for r in r_schedule:
        test_T -= r
        token_schedule.append(test_T)

    return {
        "initial_T": initial_T,
        "r_schedule": r_schedule,
        "is_skip_layer": is_skip_layer,
        "num_active_skips": num_active_skips,
        "skip_indices": skip_indices,
        "token_schedule": token_schedule,
    }

def log_udt_config(
    *,
    schedule_info,
    patch_size,
    depth,
    enc_depth,
    dec_depth,
    num_seg,
    block_kwargs=None,
    log_dir=None,
):

    block_config = ""

    if block_kwargs:
        max_key_len = max(len(k) for k in block_kwargs.keys())

        block_lines = [
            f"• {k:<{max_key_len}} : {v}"
            for k, v in block_kwargs.items()
        ]

        block_config = "\n".join(block_lines)

    config_info = textwrap.dedent(f"""
==================================================
🚀  UDT Model Configuration Info
==================================================

[ General Settings ]
• Patch Size       : {patch_size}
• Initial Tokens   : {schedule_info["initial_T"]}

[ U-Net Depth Split ]
• Total Depth      : {depth}
• Encoder Depth    : {enc_depth} (Merge & Skip)
• Decoder Depth    : {dec_depth} (Unmerge & Connect)

[ ToMe & Skip Configuration ]
• Num Segments     : {num_seg}
• Active Skips     : {schedule_info["num_active_skips"]} layers
                    (skip connections use features before encoder indices:
                    {schedule_info["skip_indices"]})
• ToMe r Schedule  : {schedule_info["r_schedule"]}
• Token Schedule   : {schedule_info["token_schedule"]}

[ Block Configuration ]
{block_config}

==================================================
""")

    if not dist.is_initialized() or dist.get_rank() == 0:
        print(config_info)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)

        save_path = os.path.join(log_dir, "model_config.txt")

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(config_info)

    return config_info

#################################################################################
#                               Token Merging (ToME)                            #
#################################################################################
def bipartite_soft_matching(metric: torch.Tensor, r: int) -> Tuple[Callable, Callable, torch.Tensor]:
    B, T, C = metric.shape
    
    num_a = (T + 1) // 2
    num_b = T // 2
    
    r = min(r, num_b)

    if r <= 0:
        mask = torch.ones(B, T, 1, device=metric.device, dtype=metric.dtype)
        return do_nothing, do_nothing, mask

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        
        scores = a @ b.transpose(-1, -2) 
        
        node_max, node_idx = scores.max(dim=-1) 
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :]  
        src_idx = edge_idx[..., :r, :]  
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

        soft_a = 1.0 - node_max 
        node_max_b, _ = scores.max(dim=-2) 
        soft_b = 1.0 - node_max_b
        
        soft_mask = torch.empty(B, T, 1, device=metric.device, dtype=metric.dtype)
        soft_mask[..., ::2, 0] = soft_a
        soft_mask[..., 1::2, 0] = soft_b
        
        mask_min = soft_mask.min(dim=1, keepdim=True)[0]
        mask_max = soft_mask.max(dim=1, keepdim=True)[0]
        mask = (soft_mask - mask_min) / (mask_max - mask_min + 1e-5) 

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        n, t1, c = x.shape
        x_a, x_b = x[..., ::2, :], x[..., 1::2, :]
        
        curr_num_a = (t1 + 1) // 2
        
        unm = gather(x_a, dim=-2, index=unm_idx.expand(n, curr_num_a - r, c))
        src = gather(x_a, dim=-2, index=src_idx.expand(n, r, c))
        
        dst = x_b.clone()
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        
        return torch.cat([dst, unm], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        n, _, c = x.shape
        dst_len = num_b 
        
        dst, unm = x[:, :dst_len, :], x[:, dst_len:, :]
        src = gather(dst, dim=-2, index=dst_idx.expand(n, r, c))
        
        out = torch.zeros(n, T, c, device=x.device, dtype=x.dtype)
        out[..., 1::2, :] = dst
        out[..., ::2, :].scatter_(dim=-2, index=unm_idx.expand(n, num_a - r, c), src=unm)
        out[..., ::2, :].scatter_(dim=-2, index=src_idx.expand(n, r, c), src=src)
        
        return out

    return merge, unmerge, mask

def do_nothing(x: torch.Tensor, mode:str=None):
    return x

def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)

def bipartite_soft_matching_random2d(metric: torch.Tensor,
                                     w: int, h: int, sx: int, sy: int, r: int,
                                     no_rand: bool = True,
                                     generator: torch.Generator = None) -> Tuple[Callable, Callable, torch.Tensor]:
    B, N, _ = metric.shape

    if r <= 0:
        mask = torch.ones(B, N, 1, device=metric.device, dtype=metric.dtype)
        return do_nothing, do_nothing, mask

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    
    with torch.no_grad():
        hsy, wsx = h // sy, w // sx

        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(sy*sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(metric.device)
        
        idx_buffer_view = torch.zeros(hsy, wsx, sy*sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)
        del idx_buffer, idx_buffer_view

        num_dst = hsy * wsx
        num_src = N - num_dst
        
        a_idx = rand_idx[:, num_dst:, :] 
        b_idx = rand_idx[:, :num_dst, :] 

        def split(x):
            n = x.shape[0]
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(n, num_src, C))
            dst = gather(x, dim=1, index=b_idx.expand(n, num_dst, C))
            return src, dst

        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        r = min(num_src, r) 

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  
        src_idx = edge_idx[..., :r, :]  
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

        soft_a = 1.0 - node_max.unsqueeze(-1) 
        node_max_b, _ = scores.max(dim=-2) 
        soft_b = 1.0 - node_max_b.unsqueeze(-1) 
        
        soft_mask = torch.empty(B, N, 1, device=metric.device, dtype=metric.dtype)
        soft_mask.scatter_(dim=1, index=a_idx.expand(B, num_src, 1), src=soft_a)
        soft_mask.scatter_(dim=1, index=b_idx.expand(B, num_dst, 1), src=soft_b)
        
        mask_min = soft_mask.min(dim=1, keepdim=True)[0]
        mask_max = soft_mask.max(dim=1, keepdim=True)[0]
        mask = (soft_mask - mask_min) / (mask_max - mask_min + 1e-5)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, _, c = src.shape
        
        unm = gather(src, dim=-2, index=unm_idx.expand(n, num_src - r, c))
        src_merged = gather(src, dim=-2, index=src_idx.expand(n, r, c))
        
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src_merged, reduce=mode)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        n, _, c = x.shape
        unm_len = num_src - r
        
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        
        src_unmerged = gather(dst, dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(n, num_dst, c), src=dst)
        
        idx_unm = gather(a_idx.expand(n, num_src, 1), dim=1, index=unm_idx.expand(n, unm_len, 1)).expand(n, unm_len, c)
        out.scatter_(dim=-2, index=idx_unm, src=unm)
        
        idx_src = gather(a_idx.expand(n, num_src, 1), dim=1, index=src_idx.expand(n, r, 1)).expand(n, r, c)
        out.scatter_(dim=-2, index=idx_src, src=src_unmerged)

        return out

    return merge, unmerge, mask