import logging
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
import einops
from einops import rearrange, repeat, reduce
from .transformer import *
from timm.models.layers import trunc_normal_
class BaseModel(nn.Module):
    @property
    def num_parameters(self):
        nweights, nparams = 0, 0
        for k, v in self.named_parameters():
            nweights += np.prod(v.size())
            nparams += 1
        return nweights, nparams

    @property
    def num_trainable_parameters(self):
        nweights, nparams = 0, 0
        for k, v in self.named_parameters():
            if v.requires_grad:
                nweights += np.prod(v.size())
                nparams += 1
        return nweights, nparams

    def prepare_batch(self, batch):
        device = next(self.parameters()).device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        return batch
    
    def _init_weights(self, m):#初始化权重参数
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            trunc_normal_(m.weight, std=.02)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class Model_Transformer(nn.Module):
    def __init__(
            self, 
            embed_dim: int,
            n_heads: int,
            attn_pdrop: int,
            resid_pdrop: int,
            n_enc_layers: int,
            n_dec_layers: int,
            block_size: int,#zhuyi
            action_dim: int,
            mlp_pdrop: float,
            embed_pdrob: float,
            bias=False,
            use_rot_embed: bool = False,
            rotary_xpos: bool = False,
            use_noise_encoder: bool = False,
            linear_output: bool = True,
            encoder_use_cross_attention = True,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim, #512
            n_heads=n_heads, #8
            attn_pdrop=attn_pdrop, #0.3
            resid_pdrop=resid_pdrop, #0.1
            n_layers=n_enc_layers, #4
            block_size=block_size, #1+10+1+1
            bias=bias,
            use_rot_embed=use_rot_embed,
            rotary_xpos=rotary_xpos,
            mlp_pdrop=mlp_pdrop,
            use_cross_attention = encoder_use_cross_attention,
        )
        self.decoder = TransformerFiLMDecoder(
            embed_dim=embed_dim,
            n_heads=n_heads,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            n_layers=n_dec_layers,
            film_cond_dim=embed_dim,
            block_size=block_size,
            bias=bias,
            use_rot_embed=use_rot_embed,
            rotary_xpos=rotary_xpos,
            mlp_pdrop=mlp_pdrop,
            use_cross_attention=True,
            use_noise_encoder=use_noise_encoder,
        )
        self.drop = nn.Dropout(embed_pdrob)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))#位置编码
        self.action_emb = nn.Linear(action_dim, embed_dim)
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, self.action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, self.action_dim)
            )
        self.sigma_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim), #b embed_dim
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.spa_emb = nn.Sequential(
            #MLP

        )
        self.lang_emb = nn.Sequential(

        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, Model_Transformer):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def process_sigma_embeddings(self, sigma):
        sigmas = sigma.log() / 4
        sigmas = einops.rearrange(sigmas, 'b -> b 1')#b 1
        emb_t = self.sigma_emb(sigmas) #b d
        if len(emb_t.shape) == 2:
            emb_t = einops.rearrange(emb_t, 'b d -> b 1 d')
        return emb_t
    
    def apply_position_embeddings(self, obs_embeds, language_embeds):
        position_embeddings = self.pos_emb
        obs_x = self.drop(obs_embeds + position_embeddings)
        language_x = self.drop(language_embeds + position_embeddings)
        return obs_x, language_x
    
    def enc_only_forward(self, obs_embeds, language_embeds):
        obs_embeds = self.spa_emb(obs_embeds)
        lang_embeds = self.lang_emb(language_embeds)
        obs_x, language_x = self.apply_position_embeddings(obs_embeds, language_embeds)
        #这里修改，通过开关控制是否cross一下
        context = self.encoder(obs_x, language_x)#x(q), context(k,v)
        return context

    def dec_only_forward(self, context, actions, sigma):
        emb_t = self.process_sigma_embeddings(sigma)
        action_embed = self.action_emb(actions)
        action_x = self.drop(action_embed)
        x = self.decoder(action_x, emb_t, context)
        pred_actions = self.action_pred(x)
        return pred_actions
        
    def forward(self, actions, obs_embeds, language_embeds, sigma):
        context = self.enc_only_forward(obs_embeds, language_embeds)
        pred_actions = self.dec_only_forward(context, actions, sigma)
        return pred_actions