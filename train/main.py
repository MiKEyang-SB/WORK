import time
import os
from train.utils.distributed import set_cuda
from train.utils.logger import LOGGER, add_log_to_file
from train.utils.misc import set_random_seed
import wandb
# wandb.init(project="pt3-diff", mode="offline")
import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import (DPDataset, midi_collate_fn, loader)
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp
import numpy as np
from train.model.Agent import DiffuseAgent
DATASET_FACTORY = {
    'DP': (DPDataset, midi_collate_fn),
}
MODEL_FACTORY = {
    'DP': DiffuseAgent,
}

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
    if config.locak_rank != -1:
        seed += config.rank
    set_random_seed(seed)

    dataset_class, dataset_collate_fn = DATASET_FACTORY[config.MODEL.model_class]
    train_dataset = dataset_class(**config.TRAIN_DATASET)
    LOGGER.info(f'#num_train: {len(train_dataset)}')
    train_dataloader, pre_epoch = loader.build_dataloader(
        train_dataset, dataset_collate_fn, True, config
    )

    if config.TRAIN.num_train_steps is None:
        config.TRAIN.num_train_steps = len(train_dataloader) * config.TRAIN.num_epochs #计算训练总步数
    else:
        # assert config.TRAIN.num_epochs is None, 'cannot set num_train_steps and num_epochs at the same time.'
        config.TRAIN.num_epochs = int(np.ceil(config.TRAIN.num_train_steps / len(train_dataloader))) #计算需要多少个epoch
    # setup loggers
    if default_gpu:
        save_training_meta(config)
        # TB_LOGGER.create(os.path.join(config.output_dir, 'logs'))
        if config.tfboard_log_dir is None:
            output_dir_tokens = config.output_dir.split('/')
            config.tfboard_log_dir = os.path.join(output_dir_tokens[0], 'TFBoard', *output_dir_tokens[1:])
        # TB_LOGGER.create(config.tfboard_log_dir)
        model_saver = ModelSaver(os.path.join(config.output_dir, 'ckpts'))
        add_log_to_file(os.path.join(config.output_dir, 'logs', 'log.txt'))
    else:
        LOGGER.disabled = True
        model_saver = NoOp()
    



@hydra.main(config_path="./", config_name="config")
def hydra_main(config: DictConfig):
    if config.wandb_enable:
        # gerenate a id including date and time
        # time_id = f"{config.MODEL.model_class}_{time.strftime('%m%d-%H')}"
        time_id = f"{time.strftime('%m%d-%H')}"
        # gnerate a UUID incase of same time_id
        wandb.init(project='mini-diff', name=config.wandb_name + f"{time_id}_{str(uuid.uuid4())[:8]}", config=OmegaConf.to_container(config, resolve=True))
        main(config)
        wandb.finish()

    else:
        print(OmegaConf.to_container(config, resolve=True)) 
        #如果每开始wandb，则打印出完整的参数，这种方式是将config转换为标准的dict格式
        main(config)
if __name__ == '__main__':
    hydra_main()