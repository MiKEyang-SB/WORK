import logging
import math
from functools import partial
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
import einops
from einops import rearrange, repeat, reduce
from .transformer import *
from timm.models.layers import trunc_normal_
from .edm_diffusion import utils
class BaseModel(nn.Module):
    @property
    def num_parameters(self):
        nweights = sum(p.numel() for p in self.parameters())
        nparams  = sum(1 for _ in self.parameters())
        return nweights, nparams


    @property
    def num_trainable_parameters(self):
        nweights = 0
        nparams = 0
        for p in self.parameters():          # 不用名字就不必 named_parameters
            if p.requires_grad:
                nweights += p.numel()        # 等价于 np.prod(p.size())
                nparams  += 1
        return int(nweights), int(nparams)


    def prepare_batch(self, batch):
        device = next(self.parameters()).device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        return batch, device
    
    # def _init_weights(self, m):#初始化权重参数
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv1d):
    #         trunc_normal_(m.weight, std=.02)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.Embedding):
    #         trunc_normal_(m.weight, std=.02)

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
            repeat_num: int,
            embed_dim: int,
            n_heads: int,
            attn_pdrop: int,
            resid_pdrop: int,
            n_enc_layers: int,
            n_dec_layers: int,
            block_size: int,#zhuyi
            action_dim: int,
            txt_ft_size: int,
            mlp_pdrop: float,
            embed_pdrob: float,
            bias=False,
            use_rot_embed: bool = False,
            rotary_xpos: bool = False,
            use_noise_encoder: bool = False,
            linear_output: bool = False,
            encoder_use_cross_attention = False,
            enable_flash: bool = True,
            use_midi: bool = True,
    ):
        super().__init__()
        self.repeat_num = repeat_num if use_midi else 1
        self.action_dim = action_dim
        self.txt_ft_size = txt_ft_size
        self.enable_flash = enable_flash
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim, #512
            n_heads=n_heads, #8
            attn_pdrop=attn_pdrop, #0.3
            resid_pdrop=resid_pdrop, #0.1
            n_layers=n_enc_layers, #4
            block_size=block_size, 
            bias=bias,
            use_rot_embed=use_rot_embed,
            rotary_xpos=rotary_xpos,
            mlp_pdrop=mlp_pdrop,
            use_cross_attention = encoder_use_cross_attention,
            enable_flash=self.enable_flash,
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
            enable_flash=self.enable_flash,
            use_midi = use_midi,
            use_noise_encoder=use_noise_encoder,
        )
        self.drop = nn.Dropout(embed_pdrob)
        self.pos_emb = nn.Parameter(torch.zeros(1, 4, embed_dim))#位置编码
        
        #spa
        self.obs_emb = nn.Sequential(
            nn.Linear(1024, embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim), 
        )

        #txt
        self.txt_emb = nn.Sequential(
            nn.Linear(512, embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.txt_attn_fn = nn.Linear(self.txt_ft_size, 1)#text


        #action
        self.action_emb = nn.Linear(action_dim, embed_dim)
        # self.action_emb = nn.Sequential(
        #     nn.Linear(action_dim, embed_dim * 2),
        #     nn.GELU(),
        #     nn.LayerNorm(embed_dim * 2),
        #     nn.Linear(embed_dim * 2, embed_dim),
        #     # 可选：nn.Dropout(p=0.1),
        # )
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, self.action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, self.action_dim)
            )
        
        #sigma
        self.sigma_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim), #b embed_dim
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.embed_ln = nn.LayerNorm(embed_dim)
        self.latent_encoder_emb = None
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
        sigma = sigma.clamp_min(1e-12)
        sigmas = sigma.log() / 4
        sigmas = einops.rearrange(sigmas, 'b -> b 1')#b 1
        emb_t = self.sigma_emb(sigmas) #b d
        if emb_t.ndim == 2:
            emb_t = einops.rearrange(emb_t, 'b d -> b 1 d')
        return emb_t
    def prepare_txt_embeds(self, txt_embeds, txt_lens):
        txt_weights = torch.split(self.txt_attn_fn(txt_embeds), txt_lens)
        cxt_embeds = torch.split(txt_embeds, txt_lens)
        ctx = []
        for weight, embed in zip(txt_weights, cxt_embeds):
            weight = torch.softmax(weight, dim=0)
            ctx.append(torch.sum(weight * embed, dim=0))
        #ctx:[(dim,), (dim,), ...]
        ctx_embeds = torch.stack(ctx, 0)[:, None, :] #(b, 1, dim)
        return ctx_embeds
    
    def apply_position_embeddings(self, obs_embeds, language_embeds):
        position_embeddings = self.pos_emb
        B, obs_len, D = obs_embeds.shape
        _, lang_len, _ = language_embeds.shape
        obs_x = self.drop(obs_embeds + position_embeddings[:,:obs_len,:])
        language_x = self.drop(language_embeds + position_embeddings[:,obs_len:obs_len+lang_len,:])
        return obs_x, language_x
    
    def enc_only_forward(self, obs_embeds, language_embeds, txt_len):
        obs_embeds = self.obs_emb(obs_embeds) #(b,3,d)
        language_embeds = self.prepare_txt_embeds(language_embeds, txt_len)#(bs, 1, dim)
        
        lang_embeds = self.txt_emb(language_embeds)#(b,1,d)
        obs_x, language_x = self.apply_position_embeddings(obs_embeds, lang_embeds)
        #这里修改，通过开关控制是否cross一下
        input_seq = torch.cat([obs_x, language_x], dim = 1)#(b,4,d)
        # context = self.encoder(obs_x, language_x)#x(q), context(k,v)
        input_seq = self.embed_ln(input_seq)
        context = self.encoder(input_seq)
        self.latent_encoder_emb = context
        return context
    
    
    def dec_only_forward(self, context, actions, sigma):
        #eval:context:(1, 4, 512), actions:(1, 8), sigma:(1,) 
        #train:context:(bs, 4, d) actions:(bs*repeat, 8) sigma:(bs*repeat,)
        emb_t = self.process_sigma_embeddings(sigma) #eval:(1,1,512) train:(bs*repeat,1,d) else (bs, 1, d)
        action_embed = self.action_emb(actions)#(b*repeat, d)
        action_x = self.drop(action_embed)[:, None, :]#(b*repeat, 1, d)
        # print("ACTION_X:", action_x.shape)#eval:(1 1 512)

        b_r, _ , _= action_x.shape #
       
        if not self.training:
            # action_in_batch = torch.full((1,), 1, dtype=torch.long)
            action_in_batch = torch.ones(b_r, dtype=torch.long)
        else:
            assert b_r % self.repeat_num == 0, \
            f"B'={b_r} is not divisible by repeat_num={self.repeat_num}"
            action_in_batch = torch.full((b_r // self.repeat_num,), self.repeat_num, dtype=torch.long)
        # action_offset = self._lengths_to_offsets(action_in_batch)
        action_offset = torch.cumsum(torch.LongTensor(action_in_batch), dim=0).to(action_x.device)
        b, n, _ = context.shape
        context_in_batch= torch.full((b,), n, dtype=torch.long)
        context_offset = torch.cumsum(torch.LongTensor(context_in_batch), dim=0).to(action_x.device)

        x = self.decoder(emb_t, action_x, context, action_offset, context_offset)
        pred_actions = self.action_pred(x)
        return pred_actions

    def forward(self, actions, obs_embeds, language_embeds, language_lens, sigma):
        #eval:actions:(1,8), obs_embeds:(1, 3, 1024), language_embeds:(len, 512), language_lens:(len,), sigma:tensor:(len)
        context = self.enc_only_forward(obs_embeds, language_embeds, language_lens)
        # level2_context = einops.repeat(context, 'b n d -> (b k) n d', k = self.repeat_num)
        pred_actions = self.dec_only_forward(context, actions, sigma)
        #context:(bs, 4, d) actions:(bs*repeat, 8) sigma:(bs*repeat,)
        return pred_actions