import sys
import os
from pathlib import Path
import itertools
sys.path.append(os.path.abspath(os.path.join(__file__, "../../libs")))
from RLBench.rlbench.observation_config import ObservationConfig, CameraConfig
from RLBench.rlbench.environment import Environment
from RLBench.rlbench.task_environment import TaskEnvironment
from RLBench.rlbench.action_modes.action_mode import MoveArmThenGripper
from RLBench.rlbench.action_modes.gripper_action_modes import Discrete
from RLBench.rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from RLBench.rlbench.backend.exceptions import InvalidActionError
from RLBench.rlbench.demo import Demo
from PyRep.pyrep.errors import IKError, ConfigurationPathError
from PyRep.pyrep.const import RenderMode

from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import tap
import matplotlib.pyplot as plt
import torch.nn.functional as F

class Arguments(tap.Tap):
    data_dir: Path = Path("/media/mike/7e954b64-dd7d-4cbf-a706-58871eeaaae3/pdata/train")
    tasks: Tuple[str, ...] = ("close_jar",)
    cameras: Tuple[str, ...] = ("front", "left_shoulder", "right_shoulder")
    image_size: Tuple[int, ...] = (128,128)


class RLBenchEnv:
    def __init__(
            self,
            data_path, 
            image_size = (128, 128),
            apply_rgb = False,
            apply_depth = False,
            apply_pc = False,
            headless = False,
            apply_cameras = ("left_shoulder", "right_shoulder", "wrist", "front"),
            fine_sampling_ball_diameter = None,
            collision_checking = False
            ):
        self.data_path = data_path
        self.apply_rgb = apply_rgb
        self.apply_depth = apply_depth
        self.apply_pc = apply_pc
        self.apply_cameras = apply_cameras
        self.fine_sampling_ball_diameter = fine_sampling_ball_diameter
        self.obs_config = self.create_obs_config(
            image_size, apply_rgb, apply_depth, apply_pc, apply_cameras
        )
        self.action_mode = MoveArmThenGripper(
            arm_action_mode = EndEffectorPoseViaPlanning(collision_checking=collision_checking),
            # absolute_mode = True
            # frame: RelativeFrame = RelativeFrame.WORLD
            gripper_action_mode = Discrete()
        )
        self.env = Environment(
            self.action_mode, 
            str(data_path),
            self.obs_config,
            headless=headless
        )
        self.image_size = image_size

    def get_demo(self, task_name, variation, episode_index):
        """
        Fetch a demo from the saved environment.
            :param task_name: fetch task name
            :param variation: fetch variation id
            :param episode_index: fetch episode index: 0 ~ 99
            :return: desired demo
        """
        demos = self.env.get_demos(
            task_name=task_name,
            variation_number=variation,
            amount=1,
            from_episode_number=episode_index,
            random_selection=False
        )
        return demos
    
    def get_obs_action(self, obs):
        state_dict = {"rgb": [], "depth": [], "pc": []}
        for cam in self.apply_cameras:
            if self.apply_rgb:
                rgb = getattr(obs, "{}_rgb".format(cam))
                #obs.---_rgb
                state_dict["rgb"] += [rgb]

            if self.apply_depth:
                depth = getattr(obs, "{}_depth".format(cam))
                state_dict["depth"] += [depth]

            if self.apply_pc:
                pc = getattr(obs, "{}_point_cloud".format(cam))
                state_dict["pc"] += [pc]
        action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
        return state_dict, torch.from_numpy(action).float()

    def create_obs_config(
             self, image_size, apply_rgb, apply_depth, apply_pc, apply_cameras, **kwargs
            ):
        unused_cams = CameraConfig()
        unused_cams.set_all(False)
        used_cams = CameraConfig(
            rgb = apply_rgb,
            point_cloud = apply_pc,
            depth = apply_depth,
            mask = False,
            image_size = image_size,
            render_mode = RenderMode.OPENGL,
            **kwargs,
        )
        camera_names = apply_cameras
        kwargs = {}
        for n in camera_names:
            kwargs[n] = used_cams
        obs_config = ObservationConfig(
            front_camera=kwargs.get("front", unused_cams),
            left_shoulder_camera=kwargs.get("left_shoulder", unused_cams),
            right_shoulder_camera=kwargs.get("right_shoulder", unused_cams),
            wrist_camera=kwargs.get("wrist", unused_cams),
            overhead_camera=kwargs.get("overhead", unused_cams),

            joint_forces=False,
            joint_positions=False,
            joint_velocities=True, #用于判断关键帧
            task_low_dim_state=False,
            gripper_touch_forces=False,

            gripper_pose=True,
            gripper_open=True,
            gripper_matrix=True,
            gripper_joint_positions=True,                       
        )
        return obs_config
        

def _is_stopped(demo, i, obs, stopped_buffer, delta):
    next_is_not_final = i == (len(demo) - 2) #not final id
    # gripper_state_no_change = i < (len(demo) - 2) and (
    #     obs.gripper_open == demo[i + 1].gripper_open
    #     and obs.gripper_open == demo[i - 1].gripper_open
    #     and demo[i - 2].gripper_open == demo[i - 1].gripper_open
    # )
    gripper_state_no_change = i < (len(demo) - 2) and (
        obs.gripper_open == demo[i + 1].gripper_open
        and obs.gripper_open == demo[max(0, i - 1)].gripper_open
        and demo[max(0, i - 2)].gripper_open == demo[max(0, i - 1)].gripper_open
    )
    #当前帧、下一帧、上一帧、上两帧的夹爪开合状态都一致
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    #判断机械臂是否几乎没动
    stopped = (
        stopped_buffer <= 0
        and small_delta #关节速度为0
        and (not next_is_not_final) #不是结束一帧
        and gripper_state_no_change #夹爪状态稳定
    )
    return stopped

def keypoint_discovery(demo: Demo, stopping_delta=0.1) -> List[int]:
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0

    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
        #第i帧是否是关键帧
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # If change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open

    if (
        len(episode_keypoints) > 1
        and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
    ):
        episode_keypoints.pop(-2)

    return episode_keypoints

def get_observation(task_str : str, 
                    variation: int, 
                    episode: int,
                    env: RLBenchEnv,
                    store_intermediate_actions: bool):
    demos = env.get_demo(task_str, variation, episode)
    demo = demos[0]
    key_frame = keypoint_discovery(demo)
    key_frame.insert(0, 0)

    # keyframe_state_ls = []
    keyframe_rgb = []
    keyframe_depth =[]
    keyframe_pc = []
    keyframe_action_ls = []
    intermediate_action_ls = []

    for i in range(len(key_frame)):
        state, action = env.get_obs_action(demo._observations[key_frame[i]])
        #state = {"rgb": [], "depth": [], "pc": []}
        #action:[8,]
        obs_rgb, obs_depth, obs_pc = transform_obs_dict(state) 
        print('-----obs_rgb shape:', obs_rgb.shape, '--------')
        print('-----obs_depth shape:', obs_depth.shape, '--------')
        print('-----obs_pc shape:', obs_pc.shape, '--------')
        keyframe_rgb.append(obs_rgb)
        keyframe_depth.append(obs_depth)
        keyframe_pc.append(obs_pc)
        keyframe_action_ls.append(action.unsqueeze(0))
        #[cameras * shape(128,128,3)], [cameras * shape(128, 128)], [cameras * shape(128,128,3)]
        if store_intermediate_actions and i < len(key_frame) - 1: 
            intermediate_actions = []
            for j in range(key_frame[i], key_frame[i + 1] + 1):
                _, action = env.get_obs_action(demo._observations[j])
                intermediate_actions.append(action.unsqueeze (0))
            intermediate_action_ls.append(torch.cat(intermediate_actions))
            #长度比关键帧的数量小1，中间有连续的
    return demo, keyframe_rgb, keyframe_depth, keyframe_pc, intermediate_action_ls, keyframe_action_ls


def _is_stopped(demo, i, obs, stopped_buffer, delta):
    next_is_not_final = i == (len(demo) - 2)
    # gripper_state_no_change = i < (len(demo) - 2) and (
    #     obs.gripper_open == demo[i + 1].gripper_open
    #     and obs.gripper_open == demo[i - 1].gripper_open
    #     and demo[i - 2].gripper_open == demo[i - 1].gripper_open
    # )
    gripper_state_no_change = i < (len(demo) - 2) and (
        obs.gripper_open == demo[i + 1].gripper_open
        and obs.gripper_open == demo[max(0, i - 1)].gripper_open
        and demo[max(0, i - 2)].gripper_open == demo[max(0, i - 1)].gripper_open
    )
    #当前帧、前一阵、后一帧、前两帧夹爪状态都一样
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (
        stopped_buffer <= 0 #连续多帧都是静止的
        and small_delta
        and (not next_is_not_final)
        and gripper_state_no_change
    )
    return stopped #好几帧夹爪状态都一样，


def transform_obs_dict(obs_dict, augmentation = False):
    apply_depth = len(obs_dict.get("depth", [])) > 0
    apply_pc = len(obs_dict["pc"]) > 0
    num_cams = len(obs_dict["rgb"])
    def merge_obs_lists_to_tensor(obs_rgb, obs_depth, obs_pc):
        rgb_tensor = torch.stack([torch.tensor(x) if not torch.is_tensor(x) else x for x in obs_rgb], dim=0)
        depth_tensor = torch.stack([torch.tensor(x) if not torch.is_tensor(x) else x for x in obs_depth], dim=0) if obs_depth else None
        pc_tensor = torch.stack([torch.tensor(x) if not torch.is_tensor(x) else x for x in obs_pc], dim=0) if obs_pc else None
        return rgb_tensor, depth_tensor, pc_tensor
    obs_rgb = []
    obs_depth = []
    obs_pc = []
    for i in range(num_cams):
        rgb = torch.tensor(obs_dict["rgb"][i]).float().permute(2, 0, 1)
        depth = (
            torch.tensor(obs_dict["depth"][i]).float()
            if apply_depth else None
        )
        pc = (
            torch.tensor(obs_dict["pc"][i]).float().permute(2, 0, 1) 
            if apply_pc else None
        )
        rgb_shape = rgb.shape
        pc_shape = pc.shape
        if augmentation:
            raise NotImplementedError()
        rgb = rgb / 255.0
        rgb = 2 * (rgb - 0.5)
        obs_rgb += [rgb.float()]
        if depth is not None:
            obs_depth += [depth.float()]
        if pc is not None:
            obs_pc += [pc.float()]
    # obs = obs_rgb + obs_depth + obs_pc
    # return torch.cat(obs, dim = 0)
    return merge_obs_lists_to_tensor(obs_rgb, obs_depth, obs_pc) #需要转换为torch张量


def get_attn_indices_from_demo(
    task_str: str, demo: Demo, cameras: Tuple[str, ...]
) -> List[Dict[str, Tuple[int, int]]]:
    frames = keypoint_discovery(demo)

    frames.insert(0, 0)
    return [{cam: obs_to_attn(demo[f], cam) for cam in cameras} for f in frames]
def obs_to_attn(obs, camera):
    extrinsics_44 = torch.from_numpy(
        obs.misc[f"{camera}_camera_extrinsics"]
    ).float() #相机外参
    extrinsics_44 = torch.linalg.inv(extrinsics_44)
    intrinsics_33 = torch.from_numpy(
        obs.misc[f"{camera}_camera_intrinsics"]
    ).float()#相机内参
    intrinsics_34 = F.pad(intrinsics_33, (0, 1, 0, 0))
    gripper_pos_3 = torch.from_numpy(obs.gripper_pose[:3]).float()
    #夹爪位置xyz
    gripper_pos_41 = F.pad(gripper_pos_3, (0, 1), value=1).unsqueeze(1)
    #[x,y,z,1]
    points_cam_41 = extrinsics_44 @ gripper_pos_41

    proj_31 = intrinsics_34 @ points_cam_41 #相机到像素平面
    proj_3 = proj_31.float().squeeze(1) #得到[u*z, v*z, 1]
    u = int((proj_3[0] / proj_3[2]).round())
    v = int((proj_3[1] / proj_3[2]).round())

    return u, v
    #这个函数是把夹爪位置投影到图像平面的函数
if __name__ == "__main__":
    #for debug
    args = Arguments().parse_args()
    env = RLBenchEnv(
        data_path = args.data_dir,
        image_size=args.image_size,
        apply_rgb=True,
        apply_pc=True,
        apply_depth=True,
        apply_cameras=args.cameras
    )
    variations = range(0, 199)
    items = []
    for task_str, variation in itertools.product(args.tasks, variations):
        # print(task_str, variation)
        episodes_dir = args.data_dir / task_str / f'variation{variation}' / 'episodes'
        episodes = [
            (task_str, variation, int(ep.stem[7:]))
            for ep in episodes_dir.glob("episode*")
        ]
        items += episodes

    task, variation, episode = items[0]
    (demo, 
     keyframe_rgb, 
     keyframe_depth, 
     keyframe_pc, 
     intermediate_action_ls, 
     keyframe_action_ls) = get_observation(
         task, variation, episode, env, True
    )#len = 7
    # pass
    #显示出第一张图片
    rgb_save_dir = "./save_dir/img/"
    for i, img_tensor in enumerate(keyframe_rgb):
        img_tensor = img_tensor[1]
        img = (img_tensor.clone().detach().cpu() + 1.0) / 2.0
        img = img.clamp(0, 1)
        img_np = img.permute(1, 2, 0).numpy()
        plt.imshow(img_np)
        plt.savefig(os.path.join(rgb_save_dir, f"frame_{i}.jpg"), bbox_inches = 'tight')
        plt.close()
    # pc_save_dir = "./save_dir/pc/"
    # for i, pc_tensor in enumerate(keyframe_rgb):
    #     img_tensor = img_tensor[1]
    #     img = (img_tensor.clone().detach().cpu() + 1.0) / 2.0
    #     img = img.clamp(0, 1)
    #     img_np = img.permute(1, 2, 0).numpy()
    #     plt.imshow(img_np)
    #     plt.savefig(os.path.join(pc_save_dir, f"frame_{i}.jpg"), bbox_inches = 'tight')
    #     plt.close()    
    