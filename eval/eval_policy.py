from typing import Tuple, Dict, List

import os
import json
import jsonlines
import tap
import copy
from pathlib import Path
from filelock import FileLock

import torch
import numpy as np
from scipy.special import softmax
from omegaconf import OmegaConf
from train.utils.misc import set_random_seed
from train.model.Agent import DiffuseAgent
# from train.main import DATASET_FACTORY, MODEL_FACTORY
from spa.models import spa_vit_base_patch16, spa_vit_large_patch16
from env import _RLBenchEnv
from datasets.rotation_transform import RotationMatrixTransform
MODEL_FACTORY = {
    'DP': DiffuseAgent,
}
def write_to_file(filepath, data):
    lock = FileLock(filepath+'.lock')
    with lock:
        with jsonlines.open(filepath, 'a', flush=True) as outf:
            outf.write(data)

class Arguments(tap.Tap):
    exp_config: str
    device: str = 'cuda'
    microstep_data_dir: str = ''
    seed: int = 100
    num_demos: int = 20
    taskvar: str = 'push_button+0'

    checkpoint: str = None
    headless: bool = False
    max_tries: int = 10
    max_steps: int = 25
    cam_rand_factor: float = 0.0
    image_size: List[int] = [128, 128] #这里要改成从spa嵌入

    save_image: bool = False
    save_obs_outs_dir: str = None
    record_video: bool = False
    not_include_robot_cameras: bool = False
    video_rotate_cam: bool = False
    video_resolution: int = 480
    SPA_img_size: int = 224
    num_ensembles: int = 1

    _feature_map: bool=False #True:1024, False:1
    cat_cls: bool=False
    spa_ckpt_path = '/home/mike/ysz/WORK/libs/SPA/checkpoints'

class Actioner(object):
    def __init__(self, args) -> None:
        self.args = args
        self.rot_type = 'euler'
        config = OmegaConf.load(args.exp_config)
        self.config = config
        self.device = args.device
        self.spa_ckpt_path = args.spa_ckpt_path
        self._feature_map = args._feature_map
        self.cat_cls = args.cat_cls
        self.SPA_img_size = args.SPA_img_size
        self.rot_transform = RotationMatrixTransform()
        if args.checkpoint is not None:
            config.checkpoint = args.checkpoint

        model_class = MODEL_FACTORY[config.MODEL.model_class] 
        self.model = model_class(config.MODEL)#dp
        if config.checkpoint:
            checkpoint = torch.load(
                config.checkpoint, map_location=lambda storage, loc: storage
            )
            self.model.load_state_dict(checkpoint, strict=True)

        self.model.to(self.device)
        self.model.eval()

        self.spa_model = spa_vit_large_patch16(self.spa_ckpt_path, pretrained=True).to(self.device)
        self.spa_model.eval()
        self.spa_model.freeze()

        OmegaConf.set_readonly(self.config, True)

        data_cfg = self.config.TRAIN_DATASET
        self.data_cfg = data_cfg
        self.instr_embeds = np.load(data_cfg.task_instr_embeds_file, allow_pickle=True).item()
        if data_cfg.instr_embed_type == 'last':
            self.instr_embeds = {instr: embeds[-1:] for instr, embeds in self.instr_embeds.items()}
        self.taskvar_instrs = json.load(open(data_cfg.task_instr_file))

    def CompileImgBySPA(self, state, feature_map, cat_cls):
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # model = spa_vit_large_patch16(pretrained = True)
        # model.eval()
        # model.freeze()
        
        # model = model.to(self.device)
        images = torch.nn.functional.interpolate(
            state, size=(self.SPA_img_size, self.SPA_img_size), mode="bilinear"
        ).to(self.device) / 255.0
        #n c h w [3 3 224 224]
        with torch.inference_mode():
            feature_map = self.spa_model(images, feature_map=feature_map, cat_cls=cat_cls)
        return feature_map
    
    def preprocess_obs(self, taskvar, step_id, obs):
        rgb = np.stack(obs['rgb'], 0)  # (N, H, W, C)
        state = torch.tensor(rgb).permute(0, 3, 1, 2)#(3,3,128,128)
        # select one instruction
        instr = self.taskvar_instrs[taskvar][0] #选第一个语言
        instr_embed = self.instr_embeds[instr]#编码
        
        #spa SPA_img_size
        feature_map = self.CompileImgBySPA(state, self._feature_map, self.cat_cls)
        
        batch = {
            'step_ids': torch.LongTensor([step_id]),
            'txt_embeds': torch.from_numpy(instr_embed).float(),
            'txt_lens': [instr_embed.shape[0]],
            'spa_featuremap': feature_map[None, :, :] #(1, 3, 1024)
        }

        return batch
    
    def predict(self, 
                task_str=None,
                variation=None,
                step_id=None,
                obs_state_dict=None,
                episode_id=None,
                instructions=None,):
        taskvar = f'{task_str}+{variation}'
        batch = self.preprocess_obs(
            taskvar, step_id, obs_state_dict,
        )
        with torch.no_grad():
            actions = []
            for _ in range(self.args.num_ensembles): #1
                action = self.model(batch)[0].data.cpu()
                actions.append(action)
            if len(actions) > 1:
                action = torch.stack(actions, 0).mean(0)
            else:
                action = actions[0]
        #这里输出的是(1,8)的动作，
        if len(action.shape) == 2 and action.shape[0] == 1:
            action = action.squeeze(0)   # (1,8)→(8,)
        #euler转换
        if self.rot_type == 'euler':
            pred_rot = action[3:6]
            pred_rot = pred_rot * 180
            pred_rot = self.rot_transform.euler_to_quaternion(pred_rot.cpu()).float()

        action[-1] = torch.sigmoid(action[-1]) > 0.5 
        new_action = torch.cat([action[0:3], pred_rot, action[-1:].unsqueeze(0)], dim=0)
        out = {
            'action': new_action
        }
        return out


def evaluate_actioner(args):   
    set_random_seed(args.seed)
    actioner = Actioner(args)

    pred_dir = os.path.join(actioner.config.output_dir, 'preds', f'seed{args.seed}')
    if args.cam_rand_factor > 0:
        pred_dir = '%s-cam_rand_factor%.1f' % (pred_dir, args.cam_rand_factor)
    os.makedirs(pred_dir, exist_ok=True)

    if len(args.image_size) == 1:
        args.image_size = [args.image_size[0], args.image_size[0]]    # (height, width)

    outfile = os.path.join(pred_dir, 'results.jsonl')

    existed_data = set()
    if os.path.exists(outfile):
        with jsonlines.open(outfile, 'r') as f:
            for item in f:
                existed_data.add((item['checkpoint'], '%s+%d'%(item['task'], item['variation'])))

    if (args.checkpoint, args.taskvar) in existed_data:
        return
    
    env = _RLBenchEnv( #注意这个env和package的env做对比
        data_path=args.microstep_data_dir,
        apply_rgb=True,
        # apply_pc=True,
        # apply_mask=True,
        headless=args.headless,
        image_size=args.image_size,
        cam_rand_factor=args.cam_rand_factor,
    )
    task_str, variation = args.taskvar.split('+')#注意这里是否要用到peract
    variation = int(variation)

    if args.microstep_data_dir != '':
        episodes_dir = os.path.join(args.microstep_data_dir, task_str, f"variation{variation}", "episodes")
        demo_keys, demos = [], []
        if os.path.exists(str(episodes_dir)):
            episode_ids = os.listdir(episodes_dir)
            episode_ids.sort(key=lambda ep: int(ep[7:]))
            for idx, ep in enumerate(episode_ids):
                # episode_id = int(ep[7:])
                try:
                    demo = env.get_demo(task_str, variation, idx, load_images=False)
                    demo_keys.append(f'episode{idx}')
                    demos.append(demo)
                except Exception as e:
                    print('\tProblem to load demo_id:', idx, ep)
                    print(e)
    else:
        demo_keys = None
        demos = None

    success_rate = env.evaluate(
        task_str, variation,
        actioner=actioner,
        max_episodes=args.max_steps,
        num_demos=len(demos) if demos is not None else args.num_demos,
        demos=demos,
        demo_keys=demo_keys,
        log_dir=Path(pred_dir),
        max_tries=args.max_tries,
        save_image=args.save_image,
        record_video=args.record_video,
        include_robot_cameras=(not args.not_include_robot_cameras),
        video_rotate_cam=args.video_rotate_cam,
        video_resolution=args.video_resolution,
    )

    print("Testing Success Rate {}: {:.04f}".format(task_str, success_rate))
    write_to_file(
        outfile,
        {
            'checkpoint': args.checkpoint,
            'task': task_str, 
            'variation': variation,
            'num_demos': args.num_demos, 
            'sr': success_rate
        }
    )

if __name__ == '__main__':
    args = Arguments().parse_args(known_only=True)
    args.remained_args = args.extra_args
    evaluate_actioner(args)