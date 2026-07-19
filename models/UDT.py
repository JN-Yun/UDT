# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# REPA: https://github.com/sihyun-yu/REPA/tree/main
# ToMe: https://github.com/facebookresearch/ToMe
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import math
from typing import Tuple
from timm.models.vision_transformer import PatchEmbed, Mlp  

import torch
import torch.nn as nn
import torch._dynamo
torch._dynamo.config.cache_size_limit = 256

from models.utils import build_token_schedule, log_udt_config, bipartite_soft_matching, bipartite_soft_matching_random2d


@torch.compile
def compiled_rms_norm_math(hidden_states, weight, eps: float):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return (weight * hidden_states).to(input_dtype)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return compiled_rms_norm_math(hidden_states, self.weight, self.variance_epsilon)


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    # @torch.compile
    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


#################################################################################
#                  Sine/Cosine Positional Embedding Functions                   #
#################################################################################
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  
    emb = np.concatenate([emb_h, emb_w], axis=1) 
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  
    pos = pos.reshape(-1)  
    out = np.einsum('m,d->md', pos, omega)  
    emb_sin = np.sin(out) 
    emb_cos = np.cos(out) 
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  
    return emb

# ========
# 2D RoPE 
# ========
def precompute_freqs_cis_2d(dim: int, end: int, theta: float = 10000.0, scale=1.0, use_cls=False):
    H = int( end**0.5 )
    # assert  H * H == end
    flat_patch_pos = torch.arange(0 if not use_cls else -1, end) # N = end
    x_pos = flat_patch_pos % H # N
    y_pos = flat_patch_pos // H # N
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim)) # Hc/4
    x_freqs = torch.outer(x_pos, freqs).float() # N Hc/4
    y_freqs = torch.outer(y_pos, freqs).float() # N Hc/4
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1) # N,Hc/4,2
    freqs_cis = freqs_cis.reshape(end if not use_cls else end + 1, -1)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # x: B N H Hc/2
    # freqs_cis:  N, H*Hc/2 or  N Hc/2
    ndim = x.ndim
    assert 0 <= 1 < ndim

    if freqs_cis.shape[-1] == x.shape[-1]:
        shape = [1 if i == 2 or i == 0 else d for i, d in enumerate(x.shape)]  # 1, N, 1, Hc/2
    else:
        shape = [d if i != 0 else 1 for i, d in enumerate(x.shape)] # 1, N, H, Hc/2
    return freqs_cis.view(*shape)

def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq : B N H Hc
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # B N H Hc/2
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3) # B, N, H, Hc
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def modulate(x, shift, scale):
    if scale.dim() == 2:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    else:
        return x * (1 + scale) + shift

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################            
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        self.timestep_embedding = self.positional_embedding
        t_freq = self.timestep_embedding(t, dim=self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

#################################################################################
#                                Core UDT Model                                 #
#################################################################################

class ToMeAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

            
    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None, freqs_cis: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        if freqs_cis is not None:
            q_perm = q.permute(0, 2, 1, 3) 
            k_perm = k.permute(0, 2, 1, 3)
            
            q_out, k_out = apply_rotary_emb(q_perm, k_perm, freqs_cis)
            
            q = q_out.permute(0, 2, 1, 3)
            k = k_out.permute(0, 2, 1, 3)

        k_mean = k.mean(dim=1)

        attn_mask = None
        if size is not None:
            attn_mask = size.log()[:, None, None, :, 0]

        dropout_p = self.attn_drop.p if self.training else 0.0

        x = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=dropout_p if self.training else 0.
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, k_mean



class UDTBlock(nn.Module):

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        use_rmsnorm = block_kwargs.get("use_rmsnorm", False)
        use_swiglu = block_kwargs.get("use_swiglu", False)
        attn_drop = block_kwargs.get("attn_drop", 0.0)
        proj_drop = block_kwargs.get("proj_drop", 0.0)
        qk_norm = block_kwargs.get("qk_norm", False)


        # Normalization
        norm_layer = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.norm1 = norm_layer(hidden_size, eps=1e-6) if use_rmsnorm else nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = norm_layer(hidden_size, eps=1e-6) if use_rmsnorm else nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Attention
        self.attn = ToMeAttention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm, 
            attn_drop=attn_drop,  
            proj_drop=proj_drop,  
        )

        # MLP
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        else:
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, 
                           act_layer=lambda: nn.GELU(approximate="tanh"), drop=proj_drop)

        # AdaLN Modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, tome_r=0, r_type="parall", w=None, h=None, size=None, freqs_cis=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        
        # Attention 
        x_norm1 = modulate(self.norm1(x), shift_msa, scale_msa)        
        attn_out, metric = self.attn(x_norm1, size=size, freqs_cis=freqs_cis)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        u_fn_out = None 
        mask_out = None
        
        # Merge
        if tome_r > 0:
            if r_type == "random":
                m_a, u_a, mask_a = bipartite_soft_matching_random2d(metric, w=w, h=h, sx=2, sy=2, r=tome_r)
            else:
                m_a, u_a, mask_a = bipartite_soft_matching(metric, tome_r)
                
            mask_out = mask_a
            u_fn_out = u_a 
            
            if size is None:
                size = torch.ones_like(x[..., 0, None])
                
            x = m_a(x * size, mode="sum")
            size = m_a(size, mode="sum")
            x = x / size
            
        # MLP
        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x, u_fn_out, mask_out, size
    

class FinalLayer(nn.Module):
    """
    The final layer of UDT (with advanced options).
    """
    def __init__(self, hidden_size, patch_size, out_channels, **block_kwargs):
        super().__init__()
        
        use_rmsnorm = block_kwargs.get("use_rmsnorm", False)
        
        if use_rmsnorm:
            self.norm_final = RMSNorm(hidden_size, eps=1e-6)
        else:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
    @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class UDT(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        decoder_hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        num_seg=112,               
        r_type="parall",   # parall (default) | random
        skip_interval=1,   # Interval for token merging
        log_dir=None,      # Save the token merge configurations 
        num_skip=1,        # Number of initial layers where token merging is skipped
        **block_kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        
        self.depth = depth

        self.enc_depth = depth // 2
        self.dec_depth = depth - self.enc_depth                    
        self.num_seg = num_seg    
        self.r_type = r_type      

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # Define UDT configs
        schedule_info = build_token_schedule(
            input_size=input_size,
            patch_size=patch_size,
            enc_depth=self.enc_depth,
            num_skip=num_skip,
            skip_interval=skip_interval,
            num_seg=self.num_seg,  )

        self.initial_T = schedule_info["initial_T"]
        self.r_schedule = schedule_info["r_schedule"]
        self.is_skip_layer = schedule_info["is_skip_layer"]
        self.num_active_skips = schedule_info["num_active_skips"]
        self.token_schedule = schedule_info["token_schedule"]

        log_udt_config(
            schedule_info=schedule_info,
            patch_size=self.patch_size,
            depth=self.depth,
            enc_depth=self.enc_depth,
            dec_depth=self.dec_depth,
            num_seg=self.num_seg,
            block_kwargs=block_kwargs,
            log_dir=log_dir,
        )
        self.enc_blocks = nn.ModuleList([
            UDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(self.enc_depth)
        ])
        
        self.skip_projections = nn.ModuleList([
            nn.Linear(hidden_size * 2, hidden_size) for _ in range(self.num_active_skips)
        ])

        self.dec_blocks = nn.ModuleList([
            UDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(self.dec_depth)
        ])
        
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_layer = FinalLayer(decoder_hidden_size, patch_size, self.out_channels, **block_kwargs)
        self.initialize_weights()

        # Partial ROPE
        self.use_partial_rope = block_kwargs.get("use_partial_rope", False)
        if self.use_partial_rope:
            head_dim = hidden_size // num_heads
            freqs_cis = precompute_freqs_cis_2d(dim=head_dim, end=self.initial_T, use_cls=False)
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
            )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.enc_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.dec_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, patch_size=None):
        c = self.out_channels
        p = self.x_embedder.patch_size[0] if patch_size is None else patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def _get_freqs_cis(self, current_x):
        if getattr(self, "use_partial_rope", False) and current_x.shape[1] == self.initial_T:
            return self.freqs_cis
        return None
    
    def forward(self, x, t, y):
        x = self.x_embedder(x) + self.pos_embed
        B, T, D = x.shape
        w = h = math.isqrt(T) 

        t_embed = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t_embed + y
        
        skip_feats = []; unmerge_stack = []; size_stack = []          
        size = torch.ones(B, T, 1, device=x.device, dtype=x.dtype)
        
        # ==================
        # 1. Encoder Pass
        # ==================
        for i in range(self.enc_depth):
            current_r = self.r_schedule[i]
            
            if self.is_skip_layer[i]:
                skip_feats.append(x) 
            
            size_stack.append(size)
            
            x, u_fn, _, size = self.enc_blocks[i](
                x, c, tome_r=current_r, r_type=self.r_type, w=w, h=h, size=size,
                freqs_cis=self._get_freqs_cis(x)
            )            
            
            if u_fn is not None:
                unmerge_stack.append(u_fn)

        # =================
        # 2. Decoder Pass
        # =================
        skip_proj_idx = self.num_active_skips - 1 
        
        for j in range(self.dec_depth):
            enc_idx = self.enc_depth - 1 - j
            
            if self.r_schedule[enc_idx] > 0 and len(unmerge_stack) > 0:
                u_fn = unmerge_stack.pop()
                x = u_fn(x)
            
            current_size = size_stack[enc_idx] 
            
            if self.is_skip_layer[enc_idx]:
                skip_x = skip_feats.pop() 
                
                x_concat = torch.cat([x, skip_x], dim=-1)
                x = self.skip_projections[skip_proj_idx](x_concat)
                    
                skip_proj_idx -= 1
            
            x, _, _, _ = self.dec_blocks[j](x, c, tome_r=0, size=current_size, freqs_cis=self._get_freqs_cis(x))

 
        x_out = self.final_layer(x, c)
        img_out = self.unpatchify(x_out)
        
        return img_out
    
#################################################################################
#                                   UDT Configs                                 #
#################################################################################
def UDT_XL_1(**kwargs):
    return UDT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=1, num_heads=16, **kwargs)

def UDT_XL_2(**kwargs):
    return UDT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def UDT_XL_4(**kwargs):
    return UDT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def UDT_XL_8(**kwargs):
    return UDT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def UDT_L_1(**kwargs):
    return UDT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=1, num_heads=16, **kwargs)

def UDT_L_2(**kwargs):
    return UDT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def UDT_L_4(**kwargs):
    return UDT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def UDT_L_8(**kwargs):
    return UDT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def UDT_B_1(**kwargs):
    return UDT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=1, num_heads=12, **kwargs)

def UDT_B_2(**kwargs):
    return UDT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def UDT_B_4(**kwargs):
    return UDT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def UDT_B_8(**kwargs):
    return UDT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def UDT_S_2(**kwargs):  # Modified
    return UDT(depth=12, hidden_size=480, decoder_hidden_size=480, patch_size=2, num_heads=8, **kwargs)

def UDT_S_4(**kwargs):
    return UDT(depth=12, hidden_size=384, decoder_hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def UDT_S_8(**kwargs):
    return UDT(depth=12, hidden_size=384, decoder_hidden_size=384, patch_size=8, num_heads=6, **kwargs)

def UDT_M_2(**kwargs):  # Modified
    return UDT(depth=16, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, **kwargs)

UDT_models = {
    'UDT-XL/1': UDT_XL_1,  'UDT-XL/2': UDT_XL_2,  'UDT-XL/4': UDT_XL_4,  'UDT-XL/8': UDT_XL_8,
    'UDT-L/1':  UDT_L_1,   'UDT-L/2':  UDT_L_2,   'UDT-L/4':  UDT_L_4,   'UDT-L/8':  UDT_L_8,
    'UDT-B/1':  UDT_B_1,   'UDT-B/2':  UDT_B_2,   'UDT-B/4':  UDT_B_4,   'UDT-B/8':  UDT_B_8,
    'UDT-S/2':  UDT_S_2,   'UDT-S/4':  UDT_S_4,   'UDT-S/8':  UDT_S_8, 'UDT-M/2':  UDT_M_2,
}