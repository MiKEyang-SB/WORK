
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
import functools
from collections import OrderedDict
from datasets.rotation_transform import (
    RotationMatrixTransform, quaternion_to_discrete_euler
)
from datasets.aug import *
def get_buffer_data(data, key):
    a = data[key]
    original_dtype = a[b'type']
    original_shape = a[b'shape']
    b_flat = np.frombuffer(a[b'data'], dtype=original_dtype)
    b = b_flat.reshape(original_shape)
    return b
class LRUEnvCache:
    def __init__(self, max_size=16):
        self.max_size = max_size
        self.cache = OrderedDict()

    def get(self, key, path):
        if key in self.cache:
            env, txn = self.cache.pop(key)
            self.cache[key] = (env, txn)  # 放到队尾
            return env, txn

        # 打开新的 env
        env = lmdb.open(path, readonly=True, lock=False, readahead=False, max_readers=256)
        txn = env.begin()
        self.cache[key] = (env, txn)

        # 超出最大数量，关掉最旧的
        if len(self.cache) > self.max_size:
            old_key, (old_env, _) = self.cache.popitem(last=False)
            old_env.close()

        return env, txn
    
class DPDataset(Dataset):
    def __init__(self,
                 spa_dir,
                 task_instr_file,
                 task_instr_embeds_file,
                 instr_embed_type,
                 tasks_file,
                 taskvars_filter,
                 include_last_step,
                 max_open_envs=16,
                 rot_type = 'euler',
                 load_entire_img = 'false',
                 use_aug = 'false',
                 **kwargs
                 ):
        super().__init__()
        self.task_instr = json.load(open(task_instr_file))
        self.task_instr_embeds = np.load(task_instr_embeds_file, allow_pickle=True).item()
        if instr_embed_type == 'last':
            self.task_instr_embeds = {instr: embeds[-1:] for instr, embeds in self.task_instr_embeds.items()}
        if tasks_file is not None:
            self.taskvars = json.load(open(tasks_file))
        else:
            self.taskvars = os.listdir(spa_dir)
        if taskvars_filter is not None:
            self.taskvars = [t for t in self.taskvars if t.split("_peract+")[0] in taskvars_filter]
            # print('taskvars_after_filter:', self.taskvars)#所有的任务名称
        
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
                    env.close()
                    continue
                data = msgpack.unpackb(value_bytes)
                key_frame = get_buffer_data(data, 'key_frame')
                n_steps = len(key_frame)
                env.close()

                if include_last_step:
                    self.data_ids.extend([(task, episode, t) for t in range(n_steps)])
                else:
                    self.data_ids.extend([(task, episode, t) for t in range(n_steps - 1)])

        self.env_cache = LRUEnvCache(max_size=max_open_envs)
        self.rot_type = rot_type
        self.rotation_transform = RotationMatrixTransform()
        self.AddGaussianNoise = AddGaussianNoise(mean=0.0, std=0.01)
        self.RandomShiftsAug = RandomShiftsAug(pad=4)
        self.use_aug = use_aug
    def __len__(self):
        return len(self.data_ids)
    
    def _open_env_impl(self, key):
        env = lmdb.open(self.episode_paths[key], readonly=True, lock=False,
                        readahead=False, max_readers=256)
        self.local_envs[key] = env  # 保存引用，避免被GC回收
        txn = env.begin()
        return txn
    
    def _augment(self, lang, feature, action, img=None):
        device = torch.device("cpu") 

        lang_t = torch.from_numpy(lang.copy()).to(device).float()
        lang_t = jitter(lang_t, sigma=0.02)
        lang = lang_t.cpu().numpy()

        # -------- feature 增强 --------
        feature_t = torch.from_numpy(feature.copy()).to(device).float()
        feature_t, perm, lam = feature_mixup(feature_t, alpha=0.4)
        feature = feature_t.cpu().numpy()
        # perm, lam 如果训练时需要 (比如混合标签)，你也可以 return 出去

        # -------- action 增强 --------
        action_t = torch.from_numpy(action.copy()).to(device).float()
        action_t = add_gaussian_noise_gt(action_t, sigma_trans=0.03, sigma_rot=0.02)
        action = action_t.cpu().numpy()

    def get_groundtruth_rotations(self, ee_poses):
        gt_rots = torch.from_numpy(ee_poses.copy())   # quaternions
        if self.rot_type == 'euler':    # [-1, 1]
            gt_rots = self.rotation_transform.quaternion_to_euler(gt_rots) / 180. #四元数转欧拉角

        elif self.rot_type == 'euler_disc': 
            gt_rots = [quaternion_to_discrete_euler(x, self.euler_resolution) for x in gt_rots[1:]]#5

        elif self.rot_type == 'euler_delta':
            gt_eulers = self.rotation_transform.quaternion_to_euler(gt_rots)
            gt_rots = (gt_eulers[1:] - gt_eulers[:-1]) % 360
            gt_rots[gt_rots > 180] -= 360
            gt_rots = gt_rots / 180.
            gt_rots = torch.cat([gt_rots, torch.zeros(1, 3)], 0)
        elif self.rot_type == 'rot6d':
            gt_rots = self.rotation_transform.quaternion_to_ortho6d(gt_rots)
            gt_rots = torch.cat([gt_rots, gt_rots[-1:]], 0)
        else:
            gt_rots = torch.cat([gt_rots, gt_rots[-1:]], 0)
        # gt_rots = gt_rots.numpy()
        return gt_rots
    
    def __getitem__(self, index): #(task, episode, t)
        taskvar, episode, data_step = self.data_ids[index]
        key = taskvar + episode
        _, txn = self.env_cache.get(key, self.episode_paths[key])      
        value_bytes = txn.get('00000000'.encode())
        data = msgpack.unpackb(value_bytes)
        
        # for t in num_steps:
        #这里需要把tensor拆开，根据t的值来拆分
        key_frame = get_buffer_data(data, 'key_frame') #[7,]
        keyframe_SPA_featureMap = get_buffer_data(data, 'keyframe_SPA_featureMap') #[7,3,1024,14,14]
        keyframe_action = get_buffer_data(data, 'keyframe_action') #[7,8]
        # num_steps = len(key_frame)

        keyframe_SPA_featureMap_t = keyframe_SPA_featureMap[data_step] #(3, 1024)
        keyframe_action_t = keyframe_action[data_step + 1]#array:(8,)
        
        keyframe_rot_t = self.get_groundtruth_rotations(keyframe_action_t[3:7])
        gt_action = np.concatenate([
            keyframe_action_t[0:3].astype(np.float32),   # (3,)
            keyframe_rot_t.astype(np.float32),           # (3,)
            np.array([keyframe_action_t[-1]], dtype=np.float32)  # (1,)
        ], axis=0)

        instr = random.choice(self.task_instr[taskvar])
        
        instr_embed = self.task_instr_embeds[instr]
        if self.use_aug:
            self._augment(instr_embed, keyframe_SPA_featureMap_t, gt_action)

        outs = {
            'data_ids': [f'{taskvar}-{episode}-t{data_step}'],
            'step_ids': [data_step],
            'spa_featuremap': [torch.tensor(keyframe_SPA_featureMap_t)],
            'txt_embeds': [torch.tensor(instr_embed)],
            'gt_action': [torch.tensor(gt_action)],
        }
        
        return outs

def midi_collate_fn(data):
    batch = {}
    for key in data[0].keys():
        batch[key] = sum([x[key] for x in data], [])
    batch['step_ids'] = torch.LongTensor(batch['step_ids'])
    batch['txt_lens'] = [x.size(0) for x in batch['txt_embeds']]
    
    # keys = ['gt_action', 'spa_featuremap']
    for key in ['gt_action', 'spa_featuremap', 'txt_embeds']:
        batch[key] = torch.stack(batch[key], 0)
    # batch['txt_embeds'] = torch.cat(batch['txt_embeds'], 0)#(all_txt, 512)
    return batch

    



@hydra.main(version_base=None, config_path="/home/server/ysz/WORK/train", config_name="config")
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