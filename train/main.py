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

class InfIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iter_obj = iter(dataloader)

    def next_batch(self):
        try:
            return next(self.iter_obj)
        except StopIteration:
            self.iter_obj = iter(self.dataloader)  # Reset iterator
            return next(self.iter_obj)
        
def is_main_process() -> bool:
    return os.environ.get("RANK", "0") == "0"

def _setup_stable_env():
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")     # 出错时立刻报，避免卡死
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("WANDB_START_METHOD", "thread")  # 避免多进程 fork 造成的卡顿
    os.environ.setdefault("OMP_NUM_THREADS", "1")        # 避免 dataloader 过度占用 CPU
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def main(config):
    _setup_stable_env() 
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
    LOGGER.info(f'#num_train: {len(train_dataset)}')#11851
    train_dataloader, pre_epoch = loader.build_dataloader(
        train_dataset, dataset_collate_fn, True, config
    )

    if config.VAL_DATASET.use_val:
        val_dataset = dataset_class(**config.VAL_DATASET)
        LOGGER.info(f"#num_val: {len(val_dataset)}")
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=config.TRAIN.val_batch_size,
            num_workers=config.TRAIN.n_workers, 
            pin_memory=True, 
            collate_fn=dataset_collate_fn, 
            sampler=torch.utils.data.RandomSampler(val_dataset, replacement=True)
        )
    else:
        val_dataloader = None
    val_loader = InfIterator(val_dataloader) 

    if config.TRAIN.num_train_steps is None:
        config.TRAIN.num_train_steps = len(train_dataloader) * config.TRAIN.num_epochs #计算训练总步数
    else:
        config.TRAIN.num_epochs = int(np.ceil(config.TRAIN.num_train_steps / len(train_dataloader))) #计算需要多少个epoch

    # setup loggers
    if default_gpu:
        save_training_meta(config)
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

    use_wandb = bool(config.wandb_enable and default_gpu)
    if use_wandb:
        wandb_dict = {}
    
    LOGGER.info("Model: nweights %d nparams %d" % (model.num_parameters))#17,649,632个参数
    LOGGER.info("Model: trainable nweights %d nparams %d" % (model.num_trainable_parameters))

    OmegaConf.set_readonly(config, True)

    # Load from checkpoint
    model_checkpoint_file = config.checkpoint
    optimizer_checkpoint_file = os.path.join(config.TRAIN.resume_dir, 'ckpts', 'train_state_latest.pt')

    optimizer_checkpoint = None
    restart_epoch = 0
    global_step = 0

    if os.path.exists(optimizer_checkpoint_file) and config.TRAIN.resume_training: #检查是否恢复训练
        LOGGER.info('Load the optimizer checkpoint from %s' % optimizer_checkpoint_file)
        optimizer_checkpoint = torch.load(
            optimizer_checkpoint_file, map_location=lambda storage, loc: storage
        )
        lastest_model_checkpoint_file = os.path.join(
            config.TRAIN.resume_dir, 'ckpts', 'model_step_%d.pt' % optimizer_checkpoint['step']
        )
        if os.path.exists(lastest_model_checkpoint_file):
            LOGGER.info('Load the model checkpoint from %s' % lastest_model_checkpoint_file)
            model_checkpoint_file = lastest_model_checkpoint_file
        global_step = optimizer_checkpoint['step']#设置step
        restart_epoch = global_step // len(train_dataloader)#设置epoch

    if model_checkpoint_file is not None:
        checkpoint = torch.load(model_checkpoint_file, map_location=lambda storage, loc: storage)
        LOGGER.info('Load the model checkpoint (%d params)' % len(checkpoint))
        new_checkpoint = {}
        state_dict = model.state_dict()
        # for k, v in checkpoint.items():
        #     if k in state_dict:
        #         # TODO: mae_encoder.encoder.first_conv.0.weight
        #         if k == 'mae_encoder.encoder.first_conv.0.weight':
        #             if v.size(1) != state_dict[k].size(1):
        #                 new_checkpoint[k] = torch.zeros_like(state_dict[k])
        #                 min_v_size = min(v.size(1), state_dict[k].size(1))
        #                 new_checkpoint[k][:, :min_v_size] = v[:, :min_v_size] #有通道不匹配就用0填充，复制可用的部分
        #         if v.size() == state_dict[k].size():
        #             if config.TRAIN.resume_encoder_only and (k.startswith('mae_decoder') or 'decoder_block' in k):
        #                 continue
        #             new_checkpoint[k] = v # 正常匹配就加载
        LOGGER.info('Resumed the model checkpoint (%d params)' % len(new_checkpoint))
        model.load_state_dict(state_dict, strict=config.checkpoint_strict_load)

    model.train()
    model = wrap_model(model, device, config.local_rank, find_unused_parameters=False)

    # Prepare optimizer
    optimizer, init_lrs = build_optimizer(model, config.TRAIN)
    if optimizer_checkpoint is not None:
        optimizer.load_state_dict(optimizer_checkpoint['optimizer'])

    pbar = tqdm(initial=global_step, total=config.TRAIN.num_train_steps, dynamic_ncols=True, desc="train") if default_gpu else NoOp()

    LOGGER.info(f"***** Running training with {config.world_size} GPUs *****")
    LOGGER.info("  Batch size = %d", config.TRAIN.train_batch_size if config.local_rank == -1 
                else config.TRAIN.train_batch_size * config.world_size)
    LOGGER.info("  Accumulate steps = %d", config.TRAIN.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", config.TRAIN.num_train_steps)

    optimizer.zero_grad(set_to_none=True)

    running_metrics = {}
    accum = int(config.TRAIN.gradient_accumulation_steps)
    for epoch_id in range(restart_epoch, config.TRAIN.num_epochs):
        pre_epoch(epoch_id)
        for step, batch in enumerate(train_dataloader):
            _, losses = model(batch)
            loss = losses['total_loss'] / accum
            loss.backward()
            if use_wandb:
                wandb_dict.update({'train/total_loss_step': float(losses['total_loss'].item())})
            if (step + 1) % accum == 0:
                global_step += 1
                # learning rate scheduling
                lr_decay_rate = get_lr_sched_decay_rate(global_step, config.TRAIN)#学习率衰减
                for kp, param_group in enumerate(optimizer.param_groups):
                    param_group['lr'] = lr_this_step = max(init_lrs[kp] * lr_decay_rate, 1e-5)
                lr_this_step = optimizer.param_groups[0]['lr']

                # if config.wandb_enable:
                #     wandb_dict.update({'lr': lr_this_step, 'global_step': global_step})
                grad_norm = None
                if config.TRAIN.grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.TRAIN.grad_norm
                    )
                    # TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
                    # if config.wandb_enable:
                    #     wandb_dict.update({'grad_norm': grad_norm})
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                val_dict= None
                if (val_dataloader is not None) and (global_step % config.TRAIN.val_steps == 0):
                    val_dict = validate(model, val_loader, config.TRAIN.val_num_batches_per_step)
                    
                if default_gpu:
                    pbar.set_postfix({"loss": float(loss.detach().item()) * accum, "lr": lr_this_step})
                    pbar.update(1)
                
                if use_wandb:
                    log_items = {
                        'train/total_loss': float(loss.detach().item()) * accum,
                        'train/lr': lr_this_step,
                        'train/global_step': global_step
                    }
                    if grad_norm is not None and isinstance(grad_norm, torch.Tensor):
                        log_items['train/grad_norm'] = float(grad_norm.item())
                    if val_dict is not None:   # 只有在验证过时才加
                        log_items.update(val_dict)
                    wandb.log(log_items)

                if global_step % config.TRAIN.save_steps == 0 and default_gpu:
                    model_saver.save(model, global_step, optimizer=optimizer, rewrite_optimizer=True)
                if global_step >= config.TRAIN.num_train_steps:
                    break
   
        if global_step >= config.TRAIN.num_train_steps:
            break  
    if (global_step % config.TRAIN.save_steps != 0) and default_gpu:
        model_saver.save(model, global_step, optimizer=optimizer, rewrite_optimizer=True)

@torch.no_grad()
def validate(model, val_iter, val_num_batches_per_step=5):
    model.eval()

    total_sqerr = 0.0 
    total_elems = 0 
    total_trans_mae = 0.0
    total_rot_mae = 0.0
    total_open_mae = 0.0
    n_samples = 0
    for _ in range(val_num_batches_per_step):

        batch = val_iter.next_batch()
        pred_action = model(batch) 
        gt = batch['gt_action'].to(pred_action.device)

        total_sqerr += F.mse_loss(pred_action, gt, reduction='sum').item()
        total_elems += gt.numel()
        total_trans_mae += torch.mean(torch.abs(pred_action[:, :3] - gt[:, :3])).item()
        total_rot_mae   += torch.mean(torch.abs(pred_action[:, 3:7] - gt[:, 3:7])).item()
        total_open_mae  += torch.mean(torch.abs(pred_action[:, 7] - gt[:, 7])).item()
        n_samples += 1

    avg_mse = total_sqerr / max(total_elems, 1)
    avg_trans_mae = total_trans_mae / max(n_samples, 1)
    avg_rot_mae   = total_rot_mae / max(n_samples, 1)
    avg_open_mae  = total_open_mae / max(n_samples, 1)

    model.train()
    return {
        "val/mse": avg_mse,
        "val/trans_mae": avg_trans_mae,
        "val/rot_mae": avg_rot_mae,
        "val/open_mae": avg_open_mae,
    }




@hydra.main(version_base=None, config_path=".", config_name="config")
def hydra_main(config: DictConfig):

    main_process = is_main_process()
    if not main_process:
        os.environ["WANDB_SILENT"]   = "true"
        os.environ["WANDB_DISABLED"] = "true"

    if config.wandb_enable and main_process:
        time_id = f"{time.strftime('%m%d-%H')}"
        run = wandb.init(
            project='diffuse',
            name=config.wandb_name + f"{time_id}_{str(uuid.uuid4())[:8]}",
            config=OmegaConf.to_container(config, resolve=True),
            settings=wandb.Settings(start_method="thread")  # 避免多进程 fork 卡住
        )
        try:
            main(config)
        finally:
            run.finish()
    else:
        print(OmegaConf.to_container(config, resolve=True))
        main(config)
if __name__ == '__main__':
    hydra_main()
    '''
    CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29501 train/main.py
    '''