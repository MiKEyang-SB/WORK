import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from train.model.model import BaseModel, Model_Transformer
from .edm_diffusion.score_wrappers import GCDenoiser
from train.model.edm_diffusion import utils
from train.model.edm_diffusion.gc_sampling import *
from functools import partial
import math
from typing import Any, Dict, NamedTuple, Optional, Tuple

class DiffuseAgent(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.repeat_num = self.config.repeat_num
        self.sigma_sample_density_type = config.sigma_sample_density_type #loglogistic
        self.sampler_type = config.sampler_type #ddim
        self.noise_scheduler = config.noise_scheduler #exponential
        self.inner_model = Model_Transformer(**config.INNER_MODEL)
        self.sigma_data = config.sigma_data #0.5
        self.sigma_min = config.sigma_min
        self.sigma_max = config.sigma_max
        self.sampling_steps = config.sampling_steps #ddim steps
        self.model = GCDenoiser(self.inner_model, self.sigma_data)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total trainable parameters in model backbone: {total_params}") #50M
    def make_sample_density(self):
        """ 
        Generate a sample density function based on the desired type for training the model
        We mostly use log-logistic as it has no additional hyperparameters to tune.
        """
        sd_config = []
        if self.sigma_sample_density_type == 'lognormal':
            loc = self.sigma_sample_density_mean  # if 'mean' in sd_config else sd_config['loc']
            scale = self.sigma_sample_density_std  # if 'std' in sd_config else sd_config['scale']
            return partial(utils.rand_log_normal, loc=loc, scale=scale)
        
        if self.sigma_sample_density_type == 'loglogistic':
            loc = sd_config['loc'] if 'loc' in sd_config else math.log(self.sigma_data)
            scale = sd_config['scale'] if 'scale' in sd_config else 0.5
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_log_logistic, loc=loc, scale=scale, min_value=min_value, max_value=max_value)
        
        if self.sigma_sample_density_type == 'loguniform':
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_log_uniform, min_value=min_value, max_value=max_value)
        
        if self.sigma_sample_density_type == 'uniform':
            return partial(utils.rand_uniform, min_value=self.sigma_min, max_value=self.sigma_max)
        
        if self.sigma_sample_density_type == 'v-diffusion':
            min_value = self.min_value if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_v_diffusion, sigma_data=self.sigma_data, min_value=min_value, max_value=max_value)
        if self.sigma_sample_density_type == 'discrete':
            sigmas = self.get_noise_schedule(self.num_sampling_steps*1e5, 'exponential')
            return partial(utils.rand_discrete, values=sigmas)
        if self.sigma_sample_density_type == 'split-lognormal':
            loc = sd_config['mean'] if 'mean' in sd_config else sd_config['loc']
            scale_1 = sd_config['std_1'] if 'std_1' in sd_config else sd_config['scale_1']
            scale_2 = sd_config['std_2'] if 'std_2' in sd_config else sd_config['scale_2']
            return partial(utils.rand_split_log_normal, loc=loc, scale_1=scale_1, scale_2=scale_2)
        else:
            raise ValueError('Unknown sample density type')
    def sample_loop(
        self, 
        sigmas, 
        x_t: torch.Tensor,
        obs: torch.Tensor, 
        txt_embeds: torch.Tensor, 
        txt_lens: list,
        sampler_type: str,
        extra_args={}, 
        ):
        """
        Main method to generate samples depending on the chosen sampler type. DDIM is the default as it works well in all settings.
        """
        s_churn = extra_args['s_churn'] if 's_churn' in extra_args else 0
        s_min = extra_args['s_min'] if 's_min' in extra_args else 0
        use_scaler = extra_args['use_scaler'] if 'use_scaler' in extra_args else False
        keys = ['s_churn', 'keep_last_actions']
        if bool(extra_args):
            reduced_args = {x:extra_args[x] for x in keys}
        else:
            reduced_args = {}
        if use_scaler:
            scaler = self.scaler
        else:
            scaler=None
        # ODE deterministic
        if sampler_type == 'lms':
            x_0 = sample_lms(self.model, obs, x_t, txt_embeds, txt_lens, scaler=scaler, disable=True, extra_args=reduced_args)
        # ODE deterministic can be made stochastic by S_churn != 0
        elif sampler_type == 'heun':
            x_0 = sample_heun(self.model, obs, x_t, txt_embeds, txt_lens, sigmas, scaler=scaler, s_churn=s_churn, s_tmin=s_min, disable=True)
        # ODE deterministic 
        elif sampler_type == 'euler':
            x_0 = sample_euler(self.model, obs, x_t, txt_embeds, txt_lens, sigmas, scaler=scaler, disable=True)
        # SDE stochastic
        elif sampler_type == 'ancestral':
            x_0 = sample_dpm_2_ancestral(self.model, obs, x_t, txt_embeds, txt_lens, sigmas, scaler=scaler, disable=True) 
        # SDE stochastic: combines an ODE euler step with an stochastic noise correcting step
        elif sampler_type == 'euler_ancestral':
            x_0 = sample_euler_ancestral(self.model, obs, x_t, txt_embeds, txt_lens, sigmas, scaler=scaler, disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm':
            x_0 = sample_dpm_2(self.model, obs, x_t, txt_embeds, txt_lens, sigmas, disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm_adaptive':
            x_0 = sample_dpm_adaptive(self.model, obs, x_t, txt_embeds, txt_lens, sigmas[-2].item(), sigmas[0].item(), disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm_fast':
            x_0 = sample_dpm_fast(self.model, obs, x_t, txt_embeds, txt_lens, sigmas[-2].item(), sigmas[0].item(), len(sigmas), disable=True)
        # 2nd order solver
        elif sampler_type == 'dpmpp_2s_ancestral':
            x_0 = sample_dpmpp_2s_ancestral(self.model, obs, x_t, txt_embeds, txt_lens, sigmas, scaler=scaler, disable=True)
        # 2nd order solver
        elif sampler_type == 'dpmpp_2m':
            x_0 = sample_dpmpp_2m(self.model, obs, x_t, txt_embeds, txt_lens, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2m_sde':
            x_0 = sample_dpmpp_sde(self.model, obs, x_t, txt_embeds, txt_lens, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'ddim':
            x_0 = sample_ddim(self.model, obs, x_t, txt_embeds, txt_lens, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2s':
            x_0 = sample_dpmpp_2s(self.model, obs, x_t, txt_embeds, txt_lens, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2_with_lms':
            x_0 = sample_dpmpp_2_with_lms(self.model, obs, x_t, txt_embeds, txt_lens, sigmas, scaler=scaler, disable=True)
        else:
            raise ValueError('desired sampler type not found!')
        return x_0    
    def get_noise_schedule(self, n_sampling_steps, noise_schedule_type, device):
        """
        Get the noise schedule for the sampling steps. Describes the distribution over the noise levels from sigma_min to sigma_max.
        """
        if noise_schedule_type == 'karras':
            return get_sigmas_karras(n_sampling_steps, self.sigma_min, self.sigma_max, 7, device) # rho=7 is the default from EDM karras
        elif noise_schedule_type == 'exponential':#1
            return get_sigmas_exponential(n_sampling_steps, self.sigma_min, self.sigma_max, device)
        elif noise_schedule_type == 'vp':
            return get_sigmas_vp(n_sampling_steps, device=self.device)
        elif noise_schedule_type == 'linear':
            return get_sigmas_linear(n_sampling_steps, self.sigma_min, self.sigma_max, device=device)
        elif noise_schedule_type == 'cosine_beta':
            return cosine_beta_schedule(n_sampling_steps, device=self.device)
        elif noise_schedule_type == 've':
            return get_sigmas_ve(n_sampling_steps, self.sigma_min, self.sigma_max, device=device)
        elif noise_schedule_type == 'iddpm':
            return get_iddpm_sigmas(n_sampling_steps, self.sigma_min, self.sigma_max, device=device)
        raise ValueError('Unknown noise schedule type')

    
    # def _log_training_metrics(self, total_loss, total_bs):
    #     """
    #     Log the training metrics.
    #     """
    #     # self.log("train/action_loss", action_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
    #     self.log("train/total_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True,batch_size=total_bs)
    #     # self.log("train/cont_loss", cont_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
    #     # self.log("train/img_gen_loss", img_gen_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)

    def forward(self, batch):
        #prepare batch

        batch, device = self.prepare_batch(batch) 
        if not self.training:
            return self.denoise_actions(batch)
        #这里处理数据和噪声

        actions = batch['gt_action'] #(b, 8)
        actions = einops.repeat(actions, 'b c -> (b k) c', k = self.repeat_num).to(device) #(bs*repeat_num, 8)

        sigmas = self.make_sample_density()(shape=(len(actions),), device=device).to(device)#(bs*repeat_num,)
        noise = torch.randn_like(actions).to(device)#(bs*repeat_num, 8)
        
        loss, pred_actions = self.model.loss(actions, batch['spa_featuremap'], batch['txt_embeds'], batch['txt_lens'], noise, sigmas)
        # pred_actions: (bs * repeat_num, 1, 8)
        return pred_actions, {'total_loss': loss}
    
    @torch.no_grad()
    def denoise_actions(self, batch):
        batch, device = self.prepare_batch(batch) 
        #batch:
        #step_id:(1,)
        #txt_embeds:(len, 512)
        #txt_lens:(len,),list
        #spa_featuremap:(3, 1024)
        spa_featuremap = batch['spa_featuremap'][None, :, :] #(1,3,1024)
        txt_embeds = batch['txt_embeds']#(len,512)
        txt_lens = batch['txt_lens']#List:[len,]
        self.model.eval()
        sigmas = self.get_noise_schedule(self.sampling_steps, self.noise_scheduler, device) #(11,)
        x_t = torch.randn((len(spa_featuremap), 8), device=device) * self.sigma_max #(1,8)
        actions = self.sample_loop(sigmas, x_t, spa_featuremap, txt_embeds, txt_lens, self.sampler_type)
        return actions


