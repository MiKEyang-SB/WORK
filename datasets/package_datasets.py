from torch.utils.data import Dataset
import torch
from pathlib import Path
from collections import defaultdict, Counter
from .utils import TrajectoryInterpolator
from typing import Tuple, Dict, List
import tap
import itertools
from utils.utils_with_rlbench import RLBenchEnv, get_observation, get_attn_indices_from_demo, obs_to_attn

from spa.models import spa_vit_base_patch16, spa_vit_large_patch16
import imageio.v3 as iio

class Arguments(tap.Tap):
    # data_dir: Path = Path(__file__).parent / "c2farm"
    # seed: int = 2
    # tasks: Tuple[str, ...] = ("stack_wine",)
    # cameras: Tuple[str, ...] = ("left_shoulder", "right_shoulder", "wrist", "front")
    # image_size: str = "256,256"
    # output: Path = Path(__file__).parent / "datasets"
    # max_variations: int = 199
    # offset: int = 0
    # num_workers: int = 0
    # store_intermediate_actions: int = 1
    data_dir: Path = Path("/media/mike/7e954b64-dd7d-4cbf-a706-58871eeaaae3/pdata/train")
    tasks: Tuple[str, ...] = ("close_jar",)
    cameras: Tuple[str, ...] = ("front", "left_shoulder", "overhead")
    image_size: Tuple[int, ...] = (256, 256)
    output:Path = Path("")


def CompileImgBySPA():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = spa_vit_large_patch16(pretrained = True)
    model.eval()
    model.freeze()
    model = model.to(device)

    images = [
        iio.imread
    ]

def get_observation(task_str: str, variation: int, episode: int, env: RLBenchEnv, store_intermediate_actions: bool)
    demos = env.get_demo(task_str, variation, episode)
    demo = demos[0]

class CompileRLBenchDataset(Dataset):
    def __init__(
            self,
            args: Arguments,
            root_path,
            data_dir, #./pdata/train
            tasks,
            interpolation_length = 100,
            return_low_lvl_trajectory=False,
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

        self.env = RLBenchEnv(
            data_path = args.data_dir,
            image_size=[int(x) for x in args.image_size.split(",")],
            apply_rgb = True,
            apply_cameras = args.cameras
        )
        tasks = args.tasks
        variations = range(args.offset, args.max_variations)
        self.items = []
        for task_str, variation in itertools.product(tasks, variations):
            # print(task_str, variation)
            episodes_dir = data_dir / task_str / f'variation{variation}' / 'episodes'
            episodes = [
                (task_str, variation, int(ep.stem[7:]))
                for ep in episodes_dir.glob("episode*")
            ]
            self.items += episodes
        #[('close_jar', 0, 1), ('close_jar', 0, 0), ('close_jar', 0, 3), ....]
        self.num_items = len(self.items)

    def __len__(self) -> int:
        return self.num_items
    
    def __getitem__(self, index: int) -> None:
        task, variation, episode = self.items[index]
        taskvar_dir = args.output / f"{task} + {variation}"
        taskvar_dir.mkdir(parents = True, exist_ok = True)


        # (demo, 
        # keyframe_rgb, 
        # keyframe_depth, 
        # keyframe_pc, 
        # intermediate_action_ls, 
        # keyframe_action_ls) = get_observation(
        #     task, variation, episode, self.env, True
        # )#len = 7


        attn_indices = get_attn_indices_from_demo(task, demo, args.cameras)
        state_dict: List = [[] for _ in range(7)]
        print("Demo {}".format(episode))

        state_dict[0].extend()
        state_dict[1].extend(keyframe_rgb)
        state_dict[2].extend(keyframe_depth)
        state_dict[3].extend(keyframe_pc)
        state_dict[4].extend(attn_indices)
        state_dict[5].extend(intermediate_action_ls)
        state_dict[6].extend(keyframe_action_ls)



if __name__ == '__main__':
    pass