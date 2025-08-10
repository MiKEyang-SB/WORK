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

from train.main import DATASET_FACTORY, MODEL_FACTORY

from env import _RLBenchEnv

def write_to_file(filepath, data):
    lock = FileLock(filepath+'.lock')
    with lock:
        with jsonlines.open(filepath, 'a', flush=True) as outf:
            outf.write(data)

class Arguments(tap.Tap):

    pass

class Actioner(object):
    def __init__(self, args) -> None:
        config = OmegaConf.load(args.exp_config)
        self.config = config
        model_class = MODEL_FACTORY[config.MODEL.model_class] #SimplePolicyPTV3CA
        self.model = model_class(config.MODEL)#dp
        if config.checkpoint:
            checkpoint = torch.load(
                config.checkpoint, map_location=lambda storage, loc: storage
            )
            self.model.load_state_dict(checkpoint, strict=True)

        self.model.to(self.device)
        self.model.eval()

        OmegaConf.set_readonly(self.config, True)

        data_cfg = self.config.TRAIN_DATASET
        self.data_cfg = data_cfg
        self.instr_embeds = np.load(data_cfg.instr_embed_file, allow_pickle=True).item()
        if data_cfg.instr_embed_type == 'last':
            self.instr_embeds = {instr: embeds[-1:] for instr, embeds in self.instr_embeds.items()}
        self.taskvar_instrs = json.load(open(data_cfg.taskvar_instr_file))

    def predict(self, 
                task_str=None,
                variation=None,
                step_id=None,
                obs_state_dict=None,
                episode_id=None,
                instrctions=None,):
        taskvar = f'{task_str}+{variation}'
        batch = self.preprocess_obs(
            taskvar, step_id, obs_state_dict,
        )
        with torch.no_grad():
            pass

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
    
    env = _RLBenchEnv(
        data_path=args.microstep_data_dir,
        apply_rgb=True,
        # apply_pc=True,
        # apply_mask=True,
        headless=args.headless,
        image_size=args.image_size,
        cam_rand_factor=args.cam_rand_factor,
    )
    task_str, variation = args.taskvar.split('+')
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
            'task': task_str, 'variation': variation,
            'num_demos': args.num_demos, 'sr': success_rate
        }
    )

if __name__ == '__main__':
    args = Arguments().parse_args(known_only=True)
    args.remained_args = args.extra_args
    evaluate_actioner(args)