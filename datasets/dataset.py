
import os
import numpy as np
import json
import copy
import random
import hydra
import lmdb
import msgpack
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch
def get_buffer_data(data, key):
    a = data[key]
    original_dtype = a[b'type']
    original_shape = a[b'shape']
    b_flat = np.frombuffer(a[b'data'], dtype=original_dtype)
    b = b_flat.reshape(original_shape)
    return b
class DPDataset(Dataset):
    def __init__(self,
                 spa_dir,
                 task_instr_file,
                 task_instr_embeds_file,
                 instr_embed_type,
                 tasks_file,
                 taskvars_filter,
                 include_last_step,
                 **kwargs
                 ):
        super().__init__()
        self.task_instr = json.load(open(task_instr_file))
        self.task_instr_embeds = np.load(task_instr_embeds_file, allow_pickle=True).item()
        if instr_embed_type == 'last':
            self.task_instr_embeds = {instr: embeds[-1:] for instr, embeds in self.task_instr_embeds.items()}
        # print("task_instr_embeds.shape: ", self.task_instr_embeds.keys())
        # test_instr = 'slide the ring onto the teal colored spoke'
        # print("task_instr_embeds.shape: ", self.task_instr_embeds.get(test_instr).shape)
        if tasks_file is not None:
            self.taskvars = json.load(open(tasks_file))
        else:
            self.taskvars = os.listdir(spa_dir)
        # if kwargs.get('taskvars_filter', None):
        if taskvars_filter is not None:
            self.taskvars = [t for t in self.taskvars if t.split("_peract+")[0] in taskvars_filter]
            # print('taskvars_after_filter:', self.taskvars)#所有的任务名称
        
        # self.lmdb_envs, self.lmdb_txns = {}, {}
        self.episode_paths = {}
        self.data_ids = []
        for task in self.taskvars:
            episode_path = os.path.join(spa_dir, task)
            if not os.path.exists(episode_path):
                # print(f'{task} not found in {spa_dir}')
                continue
            for episode in os.listdir(episode_path):
                full_episode_path = os.path.join(episode_path, episode)
                self.episode_paths[task+episode] = full_episode_path

                env = lmdb.open(full_episode_path, readonly=True, lock=False)
                txn = env.begin()
                value_bytes = txn.get('00000000'.encode())
                if value_bytes is None:
                    continue
                data = msgpack.unpackb(value_bytes)
                key_frame = get_buffer_data(data, 'key_frame')
                n_steps = len(key_frame)
                env.close()

                if include_last_step:
                    self.data_ids.extend([(task, episode, t) for t in range(n_steps)])
                else:
                    self.data_ids.extend([(task, episode, t) for t in range(n_steps - 1)])
        self.local_envs = {}
        self.local_txns = {}

    def __len__(self):
        return len(self.data_ids)

    def _get_txn(self, key):
        if key not in self.local_txns:
            env = lmdb.open(self.episode_paths[key], readonly=True, lock=False, readahead=False, max_readers=256)
            self.local_envs[key] = env
            self.local_txns[key] = env.begin()
        return self.local_txns[key]
            
    def __getitem__(self, index): #(task, episode, t)
        taskvar, episode, data_step = self.data_ids[index]
        key = taskvar + episode
        txn = self._get_txn(key)        
        # task, variation = taskvar.split('+')
        # data = msgpack.unpackb(self.lmdb_txns.get((taskvar + episode).encode()))#解码
        value_bytes = txn.get('00000000'.encode())
        data = msgpack.unpackb(value_bytes)
        outs = {
            'data_ids': [],
            'txt_embeds': [],
            'gt_action': [],
            'spa_featuremap': [],
            'step_ids': []
        }
        
        # for t in num_steps:
        #这里需要把tensor拆开，根据t的值来拆分
        key_frame = get_buffer_data(data, 'key_frame') #[7,]
        keyframe_SPA_featureMap = get_buffer_data(data, 'keyframe_SPA_featureMap') #[7,3,1024,14,14]
        keyframe_action = get_buffer_data(data, 'keyframe_action') #[7,8]
        num_steps = len(key_frame)
        # for t in range(num_steps):
        #     if t != data_step: 
        #         continue
        #因为这里data_step根本就不会key_frame,所以
        keyframe_SPA_featureMap_t = keyframe_SPA_featureMap[data_step] #(3, 1024, 14, 14)
        keyframe_action_t = keyframe_action[data_step + 1]#下一个时刻的动作
        instr = random.choice(self.task_instr[taskvar])
        instr_embed = self.task_instr_embeds[instr]

        outs['data_ids'].append(f'{taskvar}-{episode}-t{data_step}')
        outs['step_ids'].append(data_step)
        outs['spa_featuremap'].append(torch.tensor(keyframe_SPA_featureMap_t))
        outs['txt_embeds'].append(torch.tensor(instr_embed))
        outs['gt_action'].append(torch.tensor(keyframe_action_t))
        
        return outs

def midi_collate_fn(data):#展开
    batch = {}
    for key in data[0].keys():
        batch[key] = sum([x[key] for x in data], [])
    batch['step_ids'] = torch.LongTensor(batch['step_ids'])
    batch['txt_lens'] = [x.size(0) for x in batch['txt_embeds']]
    
    for key in ['gt_action', 'spa_featuremap']:
        batch[key] = torch.stack(batch[key], 0)
    batch['txt_embeds'] = torch.cat(batch['txt_embeds'], 0)#(all_txt, 512)
    return batch


    



@hydra.main(version_base=None, config_path="/home/mike/MyWork/train", config_name="config")
def hydra_main(config:DictConfig):
    dataset = DPDataset(**config.TRAIN_DATASET)
    # sampler: Union[
    #     RandomSampler, SequentialSampler, DistributedSampler
    # ] = RandomSampler(dataset)#打乱数据，适合训练
    batch_size = 128
    loader = DataLoader(
        dataset,
        sampler=RandomSampler,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=midi_collate_fn,
        drop_last=False,
        # prefetch_factor=2 if opts.TRAIN.n_workers > 0 else None,
    )
    sampler = RandomSampler(dataset)
    indices = [next(iter(sampler)) for _ in range(batch_size)]
    samples = [dataset[i] for i in indices]
    batch = midi_collate_fn(samples)
    # loader.setup(stage="fit")
    # first_dataset_key = next(iter(loader.dataset))
    # sample = next(iter(loader.dataset))[0]

if __name__ == '__main__':
    hydra_main()