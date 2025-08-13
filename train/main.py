import time
import os
import uuid
from train.utils.distributed import set_cuda, wrap_model
from train.utils.logger import LOGGER, add_log_to_file
from train.utils.misc import set_random_seed
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import  loader
from datasets.dataset import (DPDataset, midi_collate_fn)
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp
import numpy as np
from train.model.Agent import DiffuseAgent
import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm

from train.optim import get_lr_sched, get_lr_sched_decay_rate
from train.optim.misc import build_optimizer
DATASET_FACTORY = {
    'DP': (DPDataset, midi_collate_fn),
}
MODEL_FACTORY = {
    'DP': DiffuseAgent,
}

def is_main_process() -> bool:
    return os.environ.get("RANK", "0") == "0"

def main(config):
    OmegaConf.set_readonly(config, False)
    OmegaConf.set_struct(config, False)
    default_gpu, n_gpu, device = set_cuda(config)

    if default_gpu:
        LOGGER.info(
            'device: {} n_gpu: {}, distributed training: {}'.format(
                device, n_gpu, bool(config.local_rank != -1)
            )
        )
    seed = config.SEED
    if config.local_rank != -1:
        seed += config.rank
    set_random_seed(seed)

    dataset_class, dataset_collate_fn = DATASET_FACTORY[config.MODEL.model_class]
    train_dataset = dataset_class(**config.TRAIN_DATASET)
    LOGGER.info(f'#num_train: {len(train_dataset)}')
    train_dataloader, pre_epoch = loader.build_dataloader(
        train_dataset, dataset_collate_fn, True, config
    )
    print('len:', len(train_dataset))

    #此处加入评估集
    if config.TRAIN.num_train_steps is None:
        config.TRAIN.num_train_steps = len(train_dataloader) * config.TRAIN.num_epochs #计算训练总步数
    else:
        config.TRAIN.num_epochs = int(np.ceil(config.TRAIN.num_train_steps / len(train_dataloader))) #计算需要多少个epoch

    # setup loggers
    if default_gpu:
        save_training_meta(config)
        # TB_LOGGER.create(os.path.join(config.output_dir, 'logs'))
        # if config.tfboard_log_dir is None:
        #     output_dir_tokens = config.output_dir.split('/')
        #     config.tfboard_log_dir = os.path.join(output_dir_tokens[0], 'TFBoard', *output_dir_tokens[1:])
        # TB_LOGGER.create(config.tfboard_log_dir)
        model_saver = ModelSaver(os.path.join(config.output_dir, 'ckpts'))
        add_log_to_file(os.path.join(config.output_dir, 'logs', 'log.txt'))
    else:
        LOGGER.disabled = True
        model_saver = NoOp()

    #prepare model
    model_class = MODEL_FACTORY[config.MODEL.model_class]
    model = model_class(config.MODEL) #实例化模型

    if config.world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if config.wandb_enable:
        wandb_dict = {}
    
    LOGGER.info("Model: nweights %d nparams %d" % (model.num_parameters))#17,649,632个参数
    LOGGER.info("Model: trainable nweights %d nparams %d" % (model.num_trainable_parameters))

    OmegaConf.set_readonly(config, True)

    # Load from checkpoint
    model_checkpoint_file = config.checkpoint
    optimizer_checkpoint_file = os.path.join(
        config.output_dir, 'ckpts', 'train_state_latest.pt'
    )
    if os.path.exists(optimizer_checkpoint_file) and config.TRAIN.resume_training: #检查是否恢复训练
        LOGGER.info('Load the optimizer checkpoint from %s' % optimizer_checkpoint_file)
        optimizer_checkpoint = torch.load(
            optimizer_checkpoint_file, map_location=lambda storage, loc: storage
        )
        lastest_model_checkpoint_file = os.path.join(
            config.output_dir, 'ckpts', 'model_step_%d.pt' % optimizer_checkpoint['step']
        )
        if os.path.exists(lastest_model_checkpoint_file):
            LOGGER.info('Load the model checkpoint from %s' % lastest_model_checkpoint_file)
            model_checkpoint_file = lastest_model_checkpoint_file
        global_step = optimizer_checkpoint['step']#设置step
        restart_epoch = global_step // len(train_dataloader)#设置epoch
    else:
        optimizer_checkpoint = None
        # to compute training statistics
        restart_epoch = 0
        global_step = restart_epoch * len(train_dataloader) #训练重新开始

    if model_checkpoint_file is not None:
        checkpoint = torch.load(
            model_checkpoint_file, map_location=lambda storage, loc: storage)
        LOGGER.info('Load the model checkpoint (%d params)' % len(checkpoint))
        new_checkpoint = {}
        state_dict = model.state_dict()
        for k, v in checkpoint.items():
            if k in state_dict:
                # TODO: mae_encoder.encoder.first_conv.0.weight
                if k == 'mae_encoder.encoder.first_conv.0.weight':
                    if v.size(1) != state_dict[k].size(1):
                        new_checkpoint[k] = torch.zeros_like(state_dict[k])
                        min_v_size = min(v.size(1), state_dict[k].size(1))
                        new_checkpoint[k][:, :min_v_size] = v[:, :min_v_size] #有通道不匹配就用0填充，复制可用的部分
                if v.size() == state_dict[k].size():
                    if config.TRAIN.resume_encoder_only and (k.startswith('mae_decoder') or 'decoder_block' in k):
                        continue
                    new_checkpoint[k] = v # 正常匹配就加载
        LOGGER.info('Resumed the model checkpoint (%d params)' % len(new_checkpoint))
        model.load_state_dict(new_checkpoint, strict=config.checkpoint_strict_load)

    model.train()
    model = wrap_model(model, device, config.local_rank, find_unused_parameters=False)

    # Prepare optimizer
    optimizer, init_lrs = build_optimizer(model, config.TRAIN)
    if optimizer_checkpoint is not None:
        optimizer.load_state_dict(optimizer_checkpoint['optimizer'])

    if default_gpu:
        pbar = tqdm(initial=global_step, total=config.TRAIN.num_train_steps)
    else:
        pbar = NoOp()

    LOGGER.info(f"***** Running training with {config.world_size} GPUs *****")
    LOGGER.info("  Batch size = %d", config.TRAIN.train_batch_size if config.local_rank == -1 
                else config.TRAIN.train_batch_size * config.world_size)
    LOGGER.info("  Accumulate steps = %d", config.TRAIN.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", config.TRAIN.num_train_steps)

    optimizer.zero_grad()
    optimizer.step()

    running_metrics = {}
    for epoch_id in range(restart_epoch, config.TRAIN.num_epochs):
        pre_epoch(epoch_id)
        for step, batch in enumerate(train_dataloader):
            _, losses = model(batch)
            if config.TRAIN.gradient_accumulation_steps > 1:  # average loss
                losses['total_loss']= losses['total_loss'] / config.TRAIN.gradient_accumulation_steps
            losses['total_loss'].backward()
            for key, value in losses.items():
                if config.wandb_enable:
                    print('loss', losses.items())
                    wandb_dict.update({f'total_loss': value.item()})
            if (step + 1) % config.TRAIN.gradient_accumulation_steps == 0:
                global_step += 1
                # learning rate scheduling
                lr_decay_rate = get_lr_sched_decay_rate(global_step, config.TRAIN)#学习率衰减
                for kp, param_group in enumerate(optimizer.param_groups):
                    param_group['lr'] = lr_this_step = max(init_lrs[kp] * lr_decay_rate, 1e-5)
                    #为每类参数组设置相应的学习率
                # TB_LOGGER.add_scalar('lr', lr_this_step, global_step)
                if config.wandb_enable:
                    wandb_dict.update({'lr': lr_this_step, 'global_step': global_step})

                if config.TRAIN.grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.TRAIN.grad_norm
                    )#梯度裁剪，防止所有梯度L2范数超过一定值
                    # TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
                    if config.wandb_enable:
                        wandb_dict.update({'grad_norm': grad_norm})
                optimizer.step()
                optimizer.zero_grad()
            if global_step % config.TRAIN.bar_steps == 0:
                pbar.update(config.TRAIN.bar_steps)#更新进度条

            if global_step % config.TRAIN.log_steps == 0:
                # monitor training throughput
                LOGGER.info(
                    f'==============Epoch {epoch_id} Step {global_step}===============')
                LOGGER.info(', '.join(['%s:%.4f' % (lk, lv.val) for lk, lv in running_metrics.items()]))
                LOGGER.info('===============================================')
                if config.wandb_enable:
                    wandb.log(wandb_dict) 
            if global_step % config.TRAIN.save_steps == 0:
                model_saver.save(model, global_step, optimizer=optimizer, rewrite_optimizer=True)
            if global_step >= config.TRAIN.num_train_steps:
                break
    if global_step % config.TRAIN.save_steps != 0:
        LOGGER.info(
            f'==============Epoch {epoch_id} Step {global_step}===============')
        LOGGER.info(', '.join(['%s:%.4f' % (lk, lv.val) for lk, lv in running_metrics.items()]))
        LOGGER.info('===============================================')
        model_saver.save(model, global_step, optimizer=optimizer, rewrite_optimizer=True)

        # val_metrics = validate(model, val_loader)
        # LOGGER.info(f'=================Validation=================')
        # metric_str = ', '.join(['%s: %.4f' % (lk, lv) for lk, lv in val_metrics.items()])
        # LOGGER.info(metric_str)
        # LOGGER.info('===============================================')

@torch.no_grad()
def validate(model, val_iter, num_batches_per_step=5):
    model.eval()

    per_task_metrics = {}
    for _ in range(num_batches_per_step):
        try:
            batch = val_iter.next_batch()
        except StopIteration:
            raise ValueError("Validation iterator exhausted. This Should not happend.")
        
        pred_action = model(batch)
        pred_action = pred_action.cpu()
        batch_size = pred_action.size(0)
        total_samples += batch_size

        # pred_open = torch.sigmoid(pred_action[..., -1]) > 0.5
        # gt_open = batch["gt_actions"][..., -1].cpu()
        # batch_open_acc = (pred_open == gt_open).float().sum().item()
        # total_open_acc += batch_open_acc

        pred_loss = torch.nn.functional.mse_loss(pred_action, batch["gt_action"])
        val_total_act_loss_pp += pred_loss

        tasks = batch["data_ids"]
        task_names = [task.split("_peract")[0] for task in tasks]

        for task_name in np.unique(task_names):
            mask = (np.array(task_names) == task_name)
            task_count = mask.sum()
            loss_sum = pred_loss[mask].sum().item()
            if task_name not in per_task_metrics:
                per_task_metrics[task_name] = {
                    "act_loss": 0,
                    "count": 0
                }

            per_task_metrics[task_name]["count"] += task_count
            per_task_metrics[task_name]["act_loss"] += loss_sum

    total_samples = max(total_samples, 1)
    metrics = {
        "total/act_loss": val_total_act_loss_pp / total_samples
    }

    for task_name, task_data in per_task_metrics.items():#计算每个任务的误差
        count = max(task_data["count"], 1)
        metrics[f"per_task/{task_name}_act_loss"] = task_data["act_loss"] / count






@hydra.main(version_base=None, config_path="./", config_name="config")
def hydra_main(config: DictConfig):
    if config.wandb_enable:
        # gerenate a id including date and time
        # time_id = f"{config.MODEL.model_class}_{time.strftime('%m%d-%H')}"
        time_id = f"{time.strftime('%m%d-%H')}"
        # gnerate a UUID incase of same time_id
        wandb.init(project='diffuse', 
                   name=config.wandb_name + f"{time_id}_{str(uuid.uuid4())[:8]}", 
                   config=OmegaConf.to_container(config, resolve=True),
                #    mode="offline"
                   )
        main(config)
        wandb.finish()

    else:
        print(OmegaConf.to_container(config, resolve=True)) 
        main(config)
if __name__ == '__main__':
    hydra_main()