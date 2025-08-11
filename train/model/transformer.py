import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from inspect import isfunction

from typing import Optional, Tuple

import logging
import math 
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
import einops

from .position_embeddings import *

try:
    import flash_attn
except ImportError:
    flash_attn = None

from utils.ops import pad_tensors_wgrad, gen_seq_masks

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

@torch.inference_mode()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

# RMSNorm -- Better, simpler alternative to LayerNorm
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale, self.eps = dim**-0.5, eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


# SwishGLU -- A Gated Linear Unit (GLU) with the Swish activation; always better than GELU MLP!
class SwishGLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.act, self.project = nn.SiLU(), nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected, gate = self.project(x).tensor_split(2, dim=-1)
        return projected * self.act(gate)

class Attention(nn.Module):

    def __init__(
            self, 
            n_embd: int,
            n_head: int,
            attn_pdrop: float,
            resid_pdrop: float,
            block_size: int,
            causal: bool = False,
            bias=False,
            use_rot_embed: bool = False,
            rotary_xpos: bool = False,
            rotary_emb_dim = None,
            rotary_xpos_scale_base = 512,
            rotary_interpolation_factor = 1.,
        ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.n_embd = n_embd
        self.causal = causal
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))
        self.use_rot_embed = use_rot_embed
        if self.use_rot_embed:
        # Update (12/2022): Rotary embedding has since been hugely successful, widely adopted in many large language models, including the largest in the world, PaLM. 
        # However, it has been uncovered in the ALiBi paper that rotary embeddings cannot length extrapolate well. 
        # This was recently addressed in <a href="https://arxiv.org/abs/2212.10554v1">a Microsoft research paper</a>. 
        # They propose a way to unobtrusively add the same decay as in ALiBi, and found that this resolves the extrapolation problem.
        # You can use it in this repository by setting `rotary_xpos = True`. Like ALiBi, it would enforce the attention to be local. You can set the receptive field with `rotary_xpos_scale_base` value, which defaults to `512`
            rotary_emb_dim = max(default(rotary_emb_dim, self.n_head // 2), 32)
            self.rotary_pos_emb = RotaryEmbedding(
                rotary_emb_dim, 
                use_xpos = rotary_xpos, 
                xpos_scale_base = rotary_xpos_scale_base, 
                interpolate_factor = rotary_interpolation_factor, 
            ) 

    def forward(self, x, context=None, custom_attn_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # if the context is not None we do cross-attention othberwise self=attention
        # cross attention computes the query from x and the keys and values are from the context
        if context is not None:
            k = self.key(context).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = self.value(context).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        else:
            k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # apply rotary stuff here if needed:
        if self.use_rot_embed:
            q = self.rotary_pos_emb.rotate_queries_or_keys(q)
            k = self.rotary_pos_emb.rotate_queries_or_keys(k)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=custom_attn_mask, dropout_p=self.attn_dropout.p if self.training else 0, is_causal=self.causal)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.causal:
                if custom_attn_mask is not None:
                    att = att.masked_fill(custom_attn_mask == 0, float('-inf'))
                else:
                    att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class MLP(nn.Module):

    def __init__(
            self, 
            n_embd: int,
            bias: bool,
            dropout: float = 0
        ):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):

    def __init__(
            self, 
            n_embd: int, 
            n_heads: int, 
            attn_pdrop: float, 
            resid_pdrop: float, 
            mlp_pdrop: float,
            block_size: int, 
            causal: bool,
            use_cross_attention: bool = False,
            use_rot_embed: bool=False,
            rotary_xpos: bool = False,
            bias: bool = False, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        ):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = Attention(n_embd, n_heads, attn_pdrop, resid_pdrop, block_size, causal, bias, use_rot_embed, rotary_xpos)
        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.cross_att = Attention(n_embd, n_heads, attn_pdrop, resid_pdrop, block_size, causal, bias, use_rot_embed, rotary_xpos)
            self.ln3 = nn.LayerNorm(n_embd)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, bias=bias, dropout=mlp_pdrop)

    def forward(self, x, context=None, custom_attn_mask=None):
        x = x + self.attn(self.ln_1(x), custom_attn_mask=custom_attn_mask)
        if self.use_cross_attention and context is not None:
            x = x + self.cross_att(self.ln3(x), context, custom_attn_mask=custom_attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x
    
class MiniBlock(nn.Module):
    def __init__(
            self, 
            n_embd: int, 
            n_heads: int, 
            attn_pdrop: float, 
            resid_pdrop: float, 
            mlp_pdrop: float,
            enable_flash=True, 
            use_cross_attention: bool = False,
            bias: bool = False, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        ):
        super().__init__()  
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = QuerySupportAttention(
            channels=n_embd,
            num_heads=n_heads,
            kv_channels=None,         # =channels
            attn_drop=attn_pdrop,
            proj_drop=resid_pdrop,
            enable_flash=enable_flash,
        )

        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.cross_att = QuerySupportAttention(
                channels=n_embd,
                num_heads=n_heads,
                kv_channels=None,
                attn_drop=attn_pdrop,
                proj_drop=resid_pdrop,
                enable_flash=True,
            )
            self.ln3 = nn.LayerNorm(n_embd)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, bias=bias, dropout=mlp_pdrop)

    def forward(self, x, context=None, x_offset=None, context_offset=None):
        x = x + self.attn(
                        self.ln_1(x), 
                        context=None,
                        x_offset=x_offset,
                        context_offset=None,
                        cross=False
                        )
        if self.use_cross_attention and context is not None:
            x = x + self.cross_att(
                                self.ln3(x), 
                                context=context,
                                x_offset=x_offset,
                                context_offset=context_offset,
                                cross=self.use_cross_attention
                                )#第一个cross
        x = x + self.mlp(self.ln_2(x))
        return x

class ConditionedBlock(Block):
    """
    Block with AdaLN-Zero conditioning.
    """
    def __init__(
            self, 
            n_embd, 
            n_heads, 
            attn_pdrop, 
            resid_pdrop, 
            mlp_pdrop, 
            block_size, 
            causal, 
            film_cond_dim,
            use_cross_attention=False, 
            use_rot_embed=False, 
            rotary_xpos=False, 
            bias=False # and any other arguments from the Block class
        ):
        super().__init__(n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_pdrop, block_size, causal,
                         use_cross_attention=use_cross_attention, 
                         use_rot_embed=use_rot_embed, 
                         rotary_xpos=rotary_xpos, 
                         bias=bias)
        self.adaLN_zero = AdaLNZero(film_cond_dim)#[B, 1, 6*hidden_size] -> 6 * [B, 1, hidden_size]

    def forward(self, x, c, context=None, custom_attn_mask=None):
        #c作为条件进行adaln-zero操作
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_zero(c)
        
        # Attention with modulation
        x_attn = self.ln_1(x)
        x_attn = modulate(x_attn, shift_msa, scale_msa)
        x = x + gate_msa * self.attn(x_attn, custom_attn_mask=custom_attn_mask)
        
        # Cross attention if used
        if self.use_cross_attention and context is not None:
            x = x + self.cross_att(self.ln3(x), context, custom_attn_mask=custom_attn_mask)
        #condition作为kv
        # MLP with modulation
        x_mlp = self.ln_2(x)
        x_mlp = modulate(x_mlp, shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(x_mlp)
        
        return x
    
class MiniConditionedBlock(MiniBlock):
    """
    Block with AdaLN-Zero conditioning.
    """
    def __init__(
        self, 
        n_embd: int, 
        n_heads: int, 
        attn_pdrop: float, 
        resid_pdrop: float, 
        mlp_pdrop: float, 
        film_cond_dim: int,
        enable_flash=True, 
        use_cross_attention: bool = False, 
        bias: bool = False,
    ):
        # 直接对齐你最新的 MiniBlock 构造函数
        super().__init__(
            n_embd=n_embd,
            n_heads=n_heads,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            mlp_pdrop=mlp_pdrop,
            enable_flash=enable_flash, 
            use_cross_attention=use_cross_attention,
            bias=bias,
        )
        self.adaLN_zero = AdaLNZero(film_cond_dim)#[B, 1, 6*hidden_size] -> 6 * [B, 1, hidden_size]

    def forward(self, embed_t, x, context=None, x_offset=None, context_offset=None):
        #c作为条件进行adaln-zero操作
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_zero(embed_t)
        
        # Attention with modulation
        x_attn_in = self.ln_1(x)
        x_attn_in = modulate(x_attn_in, shift_msa, scale_msa)
        attn_out  = self.attn(
            x_attn_in,
            context=None,
            x_offset=x_offset,
            context_offset=None,
            cross=False
        )
        x = x + gate_msa * attn_out
        
        # Cross attention if used
        if self.use_cross_attention and context is not None:
            x_cross_in = self.ln3(x)
            x = x + self.cross_att(
                x_cross_in,
                context=context,
                x_offset=x_offset,
                context_offset=context_offset,
                cross=True
            )
        #condition作为kv
        # MLP with modulation
        x_mlp = self.ln_2(x)
        x_mlp = modulate(x_mlp, shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(x_mlp)
        
        return x

class NoiseBlock(Block):
    """
    Block with AdaLN-Zero conditioning.
    """
    def __init__(
            self, 
            n_embd, 
            n_heads, 
            attn_pdrop, 
            resid_pdrop, 
            mlp_pdrop, 
            block_size, 
            causal, 
            use_cross_attention=False, 
            use_rot_embed=False, 
            rotary_xpos=False, 
            bias=False # and any other arguments from the Block class
        ):
        super().__init__(n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_pdrop, block_size, causal,
                         use_cross_attention=use_cross_attention, 
                         use_rot_embed=use_rot_embed, 
                         rotary_xpos=rotary_xpos, 
                         bias=bias)

    def forward(self, x, c, context=None, custom_attn_mask=None):
        
        x = x + self.attn(self.ln_1(x) + c, custom_attn_mask=custom_attn_mask)
        if self.use_cross_attention and context is not None:
            x = x + self.cross_att(self.ln3(x) + c, context, custom_attn_mask=custom_attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class AdaLNZero(nn.Module):
    """
    AdaLN-Zero modulation for conditioning.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        # Initialize weights and biases to zero
        # nn.init.zeros_(self.modulation[1].weight)
        # nn.init.zeros_(self.modulation[1].bias)

    def forward(self, c):
        return self.modulation(c).chunk(6, dim=-1)
    #输出两组shift / scale ，两个gate
    #[B, 6*hidden_size] -> 6 * [B, hidden_size]

def modulate(x, shift, scale):
    return shift + (x * (scale))

class TransformerEncoder(nn.Module):
    def __init__(
            self, 
            embed_dim: int, 
            n_heads: int, 
            attn_pdrop: float,  
            resid_pdrop: float, 
            n_layers: int, 
            block_size: int,
            bias: bool = False,
            use_rot_embed: bool = False,
            rotary_xpos: bool = False,
            mlp_pdrop: float = 0,
            use_cross_attention=False,
        ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[Block(
            embed_dim, 
            n_heads, 
            attn_pdrop, 
            resid_pdrop, 
            mlp_pdrop,
            block_size,
            causal=False, 
            use_cross_attention=use_cross_attention,
            use_rot_embed=use_rot_embed,
            rotary_xpos=rotary_xpos,
            bias=bias
            ) 
            for _ in range(n_layers)]
        )
        self.ln = LayerNorm(embed_dim, bias)

    def forward(self, x, context=None, custom_attn_mask=None):
        for layer in self.blocks:
            if context is not None:
                x = layer(x, context=context, custom_attn_mask=custom_attn_mask)
            else:
                x = layer(x, custom_attn_mask=custom_attn_mask)
        x = self.ln(x)
        return x
    
class TransformerFiLMEncoder(nn.Module):
    def __init__(
            self, 
            embed_dim: int, 
            n_heads: int, 
            attn_pdrop: float,  
            resid_pdrop: float, 
            n_layers: int, 
            block_size: int,
            film_cond_dim: int,
            bias: bool = False,
            use_rot_embed: bool = False,
            rotary_xpos: bool = False,
            mlp_pdrop: float = 0,
        ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ConditionedBlock(
            embed_dim, 
            n_heads, 
            attn_pdrop, 
            resid_pdrop, 
            mlp_pdrop,
            block_size,
            causal=False, 
            use_rot_embed=use_rot_embed,
            rotary_xpos=rotary_xpos,
            bias=bias,
            film_cond_dim=film_cond_dim
            ) 
            for _ in range(n_layers)]
        )
        self.ln = LayerNorm(embed_dim, bias)

    def forward(self, x, c):
        for layer in self.blocks:
            x = layer(x, c)
        x = self.ln(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(
            self, 
            embed_dim: int, 
            n_heads: int, 
            attn_pdrop: float,  
            resid_pdrop: float, 
            n_layers: int, 
            block_size: int,
            bias: bool = False,
            use_rot_embed: bool = False,
            rotary_xpos: bool = False,
            mlp_pdrop: float = 0,
            use_cross_attention: bool = True,
        ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[Block(
            embed_dim, 
            n_heads, 
            attn_pdrop, 
            resid_pdrop, 
            mlp_pdrop,
            block_size,
            causal=True, 
            use_cross_attention=use_cross_attention,
            use_rot_embed=use_rot_embed,
            rotary_xpos=rotary_xpos,
            bias=bias
            ) 
            for _ in range(n_layers)]
        )
        self.ln = LayerNorm(embed_dim, bias)

    def forward(self, x, cond=None, custom_attn_mask=None):
        for layer in self.blocks:
            x = layer(x, cond, custom_attn_mask=custom_attn_mask)
        x = self.ln(x)
        return x

class TransformerFiLMDecoder(nn.Module):
    def __init__(
            self, 
            embed_dim: int, 
            n_heads: int, 
            attn_pdrop: float,  
            resid_pdrop: float, 
            n_layers: int, 
            block_size: int,
            film_cond_dim: int,
            bias: bool = False,
            use_rot_embed: bool = False,
            rotary_xpos: bool = False,
            mlp_pdrop: float = 0,
            use_cross_attention: bool = True,
            use_noise_encoder: bool = False,
            enable_flash=True,
            kwargs: Optional[DictConfig] = None,
        ):
        super().__init__()
        if use_noise_encoder:
            self.blocks = nn.Sequential(
                *[NoiseBlock(
                embed_dim, 
                n_heads, 
                attn_pdrop, 
                resid_pdrop, 
                mlp_pdrop,
                block_size,
                causal=True, 
                use_cross_attention=use_cross_attention,
                use_rot_embed=use_rot_embed,
                rotary_xpos=rotary_xpos,
                bias=bias,
                ) 
                for _ in range(n_layers)]
            )
        else:#1
            self.blocks = nn.Sequential(
                *[MiniConditionedBlock(
                    n_embd=embed_dim, 
                    n_heads=n_heads, 
                    attn_pdrop=attn_pdrop, 
                    resid_pdrop=resid_pdrop, 
                    mlp_pdrop=mlp_pdrop,
                    film_cond_dim=film_cond_dim,
                    enable_flash=enable_flash,
                    use_cross_attention=use_cross_attention, 
                    bias=bias,
                ) for _ in range(n_layers)]
            )
        self.ln = LayerNorm(embed_dim, bias)

    def forward(self, emb_t, x, context=None, x_offset=None, context_offset=None):
        for layer in self.blocks:
            x = layer(emb_t, x, context=context, x_offset=x_offset, context_offset=context_offset)
        x = self.ln(x)
        return x
class QuerySupportAttention(nn.Module):
    def __init__(
        self, 
        channels, 
        num_heads, 
        kv_channels=None, 
        attn_drop=0, 
        proj_drop=0, 
        enable_flash=True, 

        use_rot_embed: bool = False,
        rotary_xpos: bool = False,
        rotary_emb_dim: int = None,
        rotary_xpos_scale_base: int = 512,
        rotary_interpolation_factor: float = 1.0,
    ):#
        super().__init__()
        if kv_channels is None:
            kv_channels = channels
        
        self.q = nn.Linear(channels, channels, bias=True)
        self.kv = nn.Linear(kv_channels, channels * 2, bias=True)
        self.attn_drop = attn_drop

        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.scale = self.head_dim ** -0.5
        # self.qk_norm = qk_norm
        self.enable_flash = enable_flash

        # TODO: eps should be 1 / 65530 if using fp16 (eps=1e-6)
        # self.q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-6) if self.qk_norm else nn.Identity() # TODO: why not use LayerNorm
        # self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-6) if self.qk_norm else nn.Identity()

        self.use_rot_embed = use_rot_embed
        if self.use_rot_embed:#先不做
            # 常见做法：按每个 head 的维度决定旋转维度
            if rotary_emb_dim is None:
                rotary_emb_dim = self.head_dim
            rotary_emb_dim = min(rotary_emb_dim, self.head_dim)
            self.rotary_pos_emb = RotaryEmbedding(
                rotary_emb_dim,
                use_xpos=rotary_xpos,
                xpos_scale_base=rotary_xpos_scale_base,
                interpolate_factor=rotary_interpolation_factor,
            )
    def forward(self, x, context=None, x_offset=None, context_offset=None, *, cross: bool = True):
        device = x.device

        kv_source       = context if cross else x
        kv_offset_src   = context_offset if cross else x_offset

        q = self.q(x).view(-1, self.num_heads, self.head_dim)
        kv = self.kv(kv_source).view(-1, 2, self.num_heads, self.head_dim)

        k  = kv[:, 0]
        v  = kv[:, 1]
            
        # q = self.q_norm(q)
        # k = self.k_norm(k)
        kv = torch.stack([k, v], dim=1)

        if self.enable_flash:
            cu_seqlens_q = torch.cat([torch.zeros(1, dtype=torch.int32, device=device),
                                      x_offset.to(torch.int32)], dim=0)
            max_seqlen_q = offset2bincount(x_offset).max() # TODO: known
            if cross:
                cu_seqlens_k = torch.cat([torch.zeros(1, dtype=torch.int32, device=device),
                                          kv_offset_src.to(torch.int32)], dim=0)
                
                max_seqlen_k = offset2bincount(kv_offset_src).max()

                feat = flash_attn.flash_attn_varlen_kvpacked_func( #实现了一个变长注意力
                    q.half(), kv.half(), cu_seqlens_q, cu_seqlens_k, int(max_seqlen_q.item()), int(max_seqlen_k.item()),
                    dropout_p=self.attn_drop if self.training else 0,
                    softmax_scale=self.scale
                ).reshape(-1, self.channels).to(q.dtype)
            else:
                qkv = torch.stack([q, k, v], dim=1)  # (N, 3, H, D)
                feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                    qkv.half(), cu_seqlens_q, int(max_seqlen_q.item()),
                    dropout_p=self.attn_drop if self.training else 0.0,
                    softmax_scale=self.scale,
                    causal=False  
                ).reshape(-1, self.channels).to(q.dtype)
        else:
            # raise NotImplementedError
            # q: (#all points, #heads, #dim)
            # kv: (#all words, k/v, #heads, #dim)
            # print(q.size(), kv.size())
            n_q_bins  = offset2bincount(x_offset).data.cpu().numpy().tolist()            # [L1, ...]
            n_kv_bins = offset2bincount(kv_offset_src).data.cpu().numpy().tolist()

            kv_padded_mask = torch.from_numpy(
                gen_seq_masks(n_kv_bins)#返回bool型二维矩阵，表示每个位置是否是有效点
            ).to(device).logical_not() #(B, lk_max)
            #标记哪些是padding的点，以确保注意力机制只作用在有效点

            q_pad = pad_tensors_wgrad(torch.split(q, n_q_bins, dim=0), n_q_bins)# (B, Lq_max, H, D)
            kv_pad = pad_tensors_wgrad(torch.split(kv, n_kv_bins, dim=0), n_kv_bins)# (B, Lk_max, 2, H, D)
            # q_pad: (batch_size, #points, #heads, #dim)
            # kv_pad: (batch_size, #words, k/v, #heads, #dim)
            # print(q_pad.size(), kv_pad.size())
            logits = torch.einsum('bqhd,bkhd->bqkh', q_pad, kv_pad[:, :, 0]) * self.scale #计算每个Q和K的点积注意力得分
            logits.masked_fill_(kv_padded_mask.unsqueeze(1).unsqueeze(-1), -1e4) #给无效的key的logits分数设置一个极小值
            attn_probs = torch.softmax(logits, dim=2)#对key维度执行softmax
            out = torch.einsum('bqkh,bkhd->bqhd', attn_probs, kv_pad[:, :, 1]) #(B, Lq, H, D)
            out = torch.cat([ft[:n_q_bins[i]] for i, ft in enumerate(out)], dim=0)#(Nx, H, D)
            feat = out.reshape(-1, self.channels).float()#(Nx, C)

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        return feat
    
class CrossAttention(nn.Module):
    def __init__(
        self, 
        channels, 
        num_heads, 
        kv_channels=None, 
        attn_drop=0, 
        proj_drop=0, 
        qk_norm=False, 
        enable_flash=True
    ):
        super().__init__()
        if kv_channels is None:
            kv_channels = channels
        self.q = nn.Linear(channels, channels)
        self.kv = nn.Linear(kv_channels, channels * 2)
        self.attn_drop = attn_drop

        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qk_norm = qk_norm
        self.enable_flash = enable_flash

        # TODO: eps should be 1 / 65530 if using fp16 (eps=1e-6)
        self.q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-6) if self.qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-6) if self.qk_norm else nn.Identity()

    def forward(self, x, context=None):
        device = x.device

        q = self.q(x).view(-1, self.num_heads, self.head_dim)#(N, num_heads, head_dim)
        kv = self.kv(context).view(-1, 2, self.num_heads, self.head_dim)#(M, 2, num_heads, head_dim)

        q = self.q_norm(q)
        k = self.k_norm(kv[:, 0])#(M, 1, num_heads, head_dim)
        kv = torch.stack([k, kv[:, 1]], dim=1)

        if self.enable_flash:
            cu_seqlens_q = torch.cat([torch.zeros(1).int().to(device), point.offset.int()], dim=0)
            cu_seqlens_k = torch.cat([torch.zeros(1).int().to(device), point.context_offset.int()], dim=0)
            max_seqlen_q = offset2bincount(point.offset).max()
            max_seqlen_k = offset2bincount(point.context_offset).max()

            feat = flash_attn.flash_attn_varlen_kvpacked_func(
                q.half(), kv.half(), cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale
            ).reshape(-1, self.channels)
            feat = feat.to(q.dtype)
        else:
            # q: (#all points, #heads, #dim)
            # kv: (#all words, k/v, #heads, #dim)
            # print(q.size(), kv.size())
            npoints_in_batch = offset2bincount(point.offset).data.cpu().numpy().tolist()
            nwords_in_batch = offset2bincount(point.context_offset).data.cpu().numpy().tolist()
            word_padded_masks = torch.from_numpy(
                gen_seq_masks(nwords_in_batch)
            ).to(q.device).logical_not()
            # print(word_padded_masks)

            q_pad = pad_tensors_wgrad(
                torch.split(q, npoints_in_batch, dim=0), npoints_in_batch
            )
            kv_pad = pad_tensors_wgrad(
                torch.split(kv, nwords_in_batch), nwords_in_batch
            )
            # q_pad: (batch_size, #points, #heads, #dim)
            # kv_pad: (batch_size, #words, k/v, #heads, #dim)
            # print(q_pad.size(), kv_pad.size())
            logits = torch.einsum('bphd,bwhd->bpwh', q_pad, kv_pad[:, :, 0]) * self.scale
            logits.masked_fill_(word_padded_masks.unsqueeze(1).unsqueeze(-1), -1e4)
            attn_probs = torch.softmax(logits, dim=2)
            # print(attn_probs.size())
            feat = torch.einsum('bpwh,bwhd->bphd', attn_probs, kv_pad[:, :, 1])
            feat = torch.cat([ft[:npoints_in_batch[i]] for i, ft in enumerate(feat)], 0)
            feat = feat.reshape(-1, self.channels).float()
            # print(feat.size())

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point