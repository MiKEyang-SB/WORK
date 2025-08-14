import torch.utils
from torch.utils.data import Dataset
import torch
import argparse 
from pathlib import Path
from collections import defaultdict, Counter
# from .utils import TrajectoryInterpolator
from typing import Tuple, Dict, List
import itertools

import torch.utils.data
from datasets.utils_with_rlbench import RLBenchEnv, get_observation, get_attn_indices_from_demo, obs_to_attn, keypoint_discovery
import tap
from spa.models import spa_vit_base_patch16, spa_vit_large_patch16
import imageio.v3 as iio
import random
import numpy as np
import tqdm
import blosc
import pickle
import msgpack
import lmdb
import msgpack_numpy as m
import torch.multiprocessing as mp
m.patch()#numpy
SPA_img_size = 224
class Arguments(tap.Tap):
    # data_dir: Path = Path(__file__).parent / "c2farm"
    # seed: int = 2
    # tasks: Tuple[str, ...] = ("stack_wine",)
    # cameras: Tuple[str, ...] = ("left_shoulder", "right_shoulder", "wrist", "front")
    # image_size: str = "256,256"
    output: Path = Path("/home/mike/data/package_SPA_cls")
    max_variations: int = 60
    offset: int = 0
    # num_workers: int = 0
    store_intermediate_actions: int = 1
    data_dir: Path = Path("/home/mike/data/RLBench_dataset/train")
    tasks: Tuple[str, ...] =(
        "close_jar",
        "insert_onto_square_peg",
        "light_bulb_in",
        "meat_off_grill",
        "open_drawer",
        "place_cups",
        "place_shape_in_shape_sorter",
        "place_wine_at_rack_location",
        "push_buttons",
        "put_groceries_in_cupboard",
        "put_item_in_drawer",
        "put_money_in_safe",
        "reach_and_drag",
        "slide_block_to_color_target",
        "stack_blocks",
        "stack_cups",
        "sweep_to_dustpan_of_size",
        "turn_tap",
        )
    # tasks: Tuple[str, ...] = ("stack_blocks", "push_buttons")
    cameras: Tuple[str, ...] = ("front", "left_shoulder", "overhead")
    image_size: str = "128,128"
    seed: int = 2
    _feature_map: bool=False #True:1024, False:1
    cat_cls: bool=False

def save_lmdb(data_dict, lmdb_path: Path, key: str = "00000000"):
    map_size = 1 << 30
    env = lmdb.open(str(lmdb_path), map_size = map_size)

    with env.begin(write = True) as txn:
        # txn.put(key.encode("ascii"), pickle.dumps(data_dict))
        txn.put(key.encode("ascii"), msgpack.packb(data_dict))
    env.close()






class CompileRLBenchDataset(Dataset):
    def __init__(
            self,
            args: Arguments
            # root_path,
            # data_dir, #./pdata/train
            # tasks,
            # interpolation_length = 100,
            # return_low_lvl_trajectory=False,
            ):
        super().__init__()
        # self.root = root_path
        # if isinstance(root, (Path, str)):
        #     root = [Path(root)]
        # if return_low_lvl_trajectory:
        #     self._interpolate_traj = TrajectoryInterpolator(
        #         use=return_low_lvl_trajectory,
        #         interpolation_length=interpolation_length
        #     )#插值机器人轨迹
        # self._instructions = defaultdict(dict)
        # self._num_vars = Counter()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = spa_vit_large_patch16(pretrained=True).to(self.device)
        self.model.eval()
        self.model.freeze()
        self.env = RLBenchEnv(
            data_path = self.args.data_dir,
            image_size=[int(x) for x in self.args.image_size.split(",")],
            apply_rgb = True,
            apply_cameras = self.args.cameras
        )
        tasks = self.args.tasks
        variations = range(self.args.offset, self.args.max_variations)
        self.items = []
        for task_str, variation in itertools.product(tasks, variations):
            # print(task_str, variation)
            episodes_dir = self.args.data_dir / task_str / f'variation{variation}' / 'episodes'
            episodes = [
                (task_str, variation, int(ep.stem[7:]))
                for ep in episodes_dir.glob("episode*")
            ]
            self.items += episodes
        #[('close_jar', 0, 1), ('close_jar', 0, 0), ('close_jar', 0, 3), ....]
        self.num_items = len(self.items)
    def CompileImgBySPA(self, state, feature_map, cat_cls):
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # model = spa_vit_large_patch16(pretrained = True)
        # model.eval()
        # model.freeze()
        
        # model = model.to(self.device)
        images = torch.nn.functional.interpolate(
            state, size=(SPA_img_size, SPA_img_size), mode="bilinear"
        ).to(self.device) / 255.0
        #n c h w [3 3 224 224]
        feature_map = self.model(images, feature_map=feature_map, cat_cls=cat_cls)
        return feature_map
    
    def get_observation(self, 
                        task_str: str, 
                        variation: int, 
                        episode: int, 
                        env: RLBenchEnv, 
                        store_intermediate_actions: bool, 
                        _feature_map: bool,
                        cat_cls: bool):
        demos = env.get_demo(task_str, variation, episode)
        demo = demos[0]
        key_frame = keypoint_discovery(demo)
        key_frame.insert(0, 0)
        keyframe_state_ls = []
        keyframe_SPA_featureMap_ls = []
        keyframe_action_ls = []
        intermediate_action_ls = []
        for i in range(len(key_frame)):
            state, action = env.get_obs_action(demo._observations[key_frame[i]])
            state = np.array(state['rgb'])  # 将列表中的 numpy.ndarray 合并为一个 numpy.ndarray
            state = torch.tensor(state).permute(0, 3, 1, 2) #[3,3,128,128] cam, rgb, h, w
            keyframe_state_ls.append(state.unsqueeze(0)) #state:[1,3,128,128,3] 
            keyframe_action_ls.append(action.unsqueeze(0)) #action:[1, 8]
            feature_map = self.CompileImgBySPA(state, _feature_map, cat_cls)#[3,1024,14,14] else cls:[3,1024]
            keyframe_SPA_featureMap_ls.append(feature_map)
            # print(feature_map.shape)
            if store_intermediate_actions and i < len(key_frame) - 1:
                intermediate_actions = []
                for j in range(key_frame[i], key_frame[i + 1] + 1):
                    _, action = env.get_obs_action(demo._observations[j])
                    intermediate_actions.append(action.unsqueeze(0)) #intermediate_actions:[x, 8]
                intermediate_action_ls.append(torch.cat(intermediate_actions))
        keyframe_SPA_featureMap = torch.stack(keyframe_SPA_featureMap_ls, dim = 0) #[7,3,1024,14,14]
        keyframe_state = torch.cat(keyframe_state_ls, dim=0) #keyframe_state:[7,3,128,128,3]
        keyframe_action = torch.cat(keyframe_action_ls, dim=0)#keyframe_action:[7,8]
        keyframe_SPA_featureMap = keyframe_SPA_featureMap.to(self.device) 
        # print(key_frame)
        # print(keyframe_state.shape)
        # print(keyframe_action.shape)
        # print(len(intermediate_action_ls[0]))
        return demo, key_frame, keyframe_SPA_featureMap, keyframe_state, keyframe_action, intermediate_action_ls
    def __len__(self) -> int:
        return self.num_items
    
    def __getitem__(self, index: int) -> None:
        task, variation, episode = self.items[index]
        # taskvar_dir = args.output / f"{task} + {variation}"
        # taskvar_dir.mkdir(parents = True, exist_ok = True)
        print(task, variation, episode)
        (demo,
         key_frame,#list:[x1, x2, x3...]
         keyframe_SPA_featureMap,#list:[[3,1024,14,14], ...]
         keyframe_state_ls,#keyframe_state:[7,3,128,128,3]
         keyframe_action,#keyframe_action:tensor[7,8]
         intermediate_action_ls) = self.get_observation(
             task, variation, episode, self.env, 
             bool(self.args.store_intermediate_actions),
             self.args._feature_map, self.args.cat_cls)



        # attn_indices = get_attn_indices_from_demo(task, demo, args.cameras)
        # state_dict: List = [[] for _ in range(7)]
        # # print("Demo {}".format(episode))
        # # frame_ids = list(range(len(keyframe_SPA_featureMap_ls) - 1))
        # state_dict[0].extend(key_frame)
        # state_dict[1].extend(keyframe_SPA_featureMap)
        # # state_dict[2].extend(keyframe_state_ls)
        # state_dict[2].extend(attn_indices)
        # state_dict[3].extend(intermediate_action_ls)
        # state_dict = []
        # state_dict.append(key_frame)#list:[]
        # state_dict.append(keyframe_SPA_featureMap)#tensor[7,3,1024,14,14]
        # state_dict.append(keyframe_action)#tensor[7,8]
        state_dict = {}
        state_dict['key_frame'] = np.array(key_frame, dtype=np.int32)
        state_dict['keyframe_SPA_featureMap'] = keyframe_SPA_featureMap.cpu().numpy()
        state_dict['keyframe_action'] = keyframe_action.cpu().numpy()
        lmdb_path = Path(f"/home/mike/data/package_SPA_cls/train/{task}_peract+{variation}/episode{episode}")
        lmdb_path.mkdir(parents=True, exist_ok=True)
        save_lmdb(state_dict, lmdb_path)

        # print(keyframe_SPA_featureMap.device)#cuda:0
        # print(state_dict[0])
        # print(state_dict[1].shape)
        # print(state_dict[2].shape)

        # with open(taskvar_dir / f"ep{episode}.dat", "wb") as f:
        #     f.write(blosc.compress(pickle.dumps(state_dict)))
def collate_fn_custom(batch):
    return batch

if __name__ == '__main__':
    # task, variation, episode = ('close_jar', 0, 1)
    # args = Arguments().parse_args()
    # env = RLBenchEnv(
    #     data_path = args.data_dir,
    #     image_size=[int(x) for x in args.image_size.split(",")],
    #     apply_rgb = True,
    #     apply_cameras = args.cameras
    # )
    # get_observation(
    #         task, variation, episode, env,
    #         bool(args.store_intermediate_actions)
    #     )
    mp.set_start_method('spawn', force=True) 
    args = Arguments().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    dataset = CompileRLBenchDataset(args)
    print(dataset.env)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        num_workers=4,
        collate_fn=collate_fn_custom,
        pin_memory=True
    )
    # print("qq")
    for _ in tqdm.tqdm(dataloader):
        continue
