import numpy as np
import random
import torch
import copy
from torch import nn
import torch.nn.functional as F

def pad_tensors(tensors, lens=None, pad=0, max_len=None):
    """B x [T, ...] torch tensors"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens) if max_len is None else max_len
    bs = len(tensors)
    hid = list(tensors[0].size()[1:])
    size = [bs, max_len] + hid

    dtype = tensors[0].dtype
    output = torch.zeros(*size, dtype=dtype)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output

def gen_seq_masks(seq_lens, max_len=None):
    """
    Args:
        seq_lens: list or nparray int, shape=(N, )
    Returns:
        masks: nparray, shape=(N, L), padded=0
    """
    seq_lens = np.array(seq_lens)
    if max_len is None:
        max_len = max(seq_lens)
    if max_len == 0:
        return np.zeros((len(seq_lens), 0), dtype=bool)
    batch_size = len(seq_lens)
    masks = np.arange(max_len).reshape(-1, max_len).repeat(batch_size, 0)
    masks = masks < seq_lens.reshape(-1, 1)
    return masks


def normalize_pc(pc, centroid=None, return_params=False):
    # Normalize the point cloud to [-1, 1]
    if centroid is None:
        centroid = np.mean(pc, axis=0)
    else:
        centroid = copy.deepcopy(centroid)
    
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    if m < 1e-6:
        pc = np.zeros_like(pc)
    else:
        pc = pc / m
    if return_params:
        return pc, (centroid, m)
    return pc

def random_scale_pc(pc, scale_low=0.8, scale_high=1.25):
    # Randomly scale the point cloud.
    scale = np.random.uniform(scale_low, scale_high)
    pc = pc * scale
    return pc

def shift_pc(pc, shift_range=0.1):
    # Randomly shift point cloud.
    shift = np.random.uniform(-shift_range, shift_range, size=[3])
    pc = pc + shift
    return pc

def rotate_perturbation_pc(pc, angle_sigma=0.06, angle_clip=0.18):
    # Randomly perturb the point cloud by small rotations (unit: radius)
    angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
    cosval, sinval = np.cos(angles), np.sin(angles)
    Rx = np.array([[1, 0, 0], [0, cosval[0], -sinval[0]], [0, sinval[0], cosval[0]]])
    Ry = np.array([[cosval[1], 0, sinval[1]], [0, 1, 0], [-sinval[1], 0, cosval[1]]])
    Rz = np.array([[cosval[2], -sinval[2], 0], [sinval[2], cosval[2], 0], [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    pc = np.dot(pc, np.transpose(R))
    return pc

def random_rotate_z(pc, angle=None):
    # Randomly rotate around z-axis
    if angle is None:
        angle = np.random.uniform() * 2 * np.pi
    cosval, sinval = np.cos(angle), np.sin(angle)
    R = np.array([[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]])
    return np.dot(pc, np.transpose(R))

def random_rotate_xyz(pc):
    # Randomly rotate around x, y, z axis
    angles = np.random.uniform(size=[3]) * 2 * np.pi
    cosval, sinval = np.cos(angles), np.sin(angles)
    Rx = np.array([[1, 0, 0], [0, cosval[0], -sinval[0]], [0, sinval[0], cosval[0]]])
    Ry = np.array([[cosval[1], 0, sinval[1]], [0, 1, 0], [-sinval[1], 0, cosval[1]]])
    Rz = np.array([[cosval[2], -sinval[2], 0], [sinval[2], cosval[2], 0], [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    pc = np.dot(pc, np.transpose(R))
    return pc

def augment_pc(pc):
    pc = random_scale_pc(pc)
    pc = shift_pc(pc)
    # pc = rotate_perturbation_pc(pc)
    pc = random_rotate_z(pc)
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = torch.tensor(std)
        self.mean = torch.tensor(mean)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)
    
class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
    
def feature_mixup(h, alpha=0.4):  # h: [N, D]
    N = h.size(0)
    perm = torch.randperm(N, device=h.device)
    lam = torch.distributions.Beta(alpha, alpha).sample((N,1)).to(h.device)
    h_mix = lam * h + (1-lam) * h[perm]
    return h_mix, perm, lam.squeeze(1)

def jitter(h, sigma=0.01):
    return h + torch.randn_like(h) * sigma


def add_gaussian_noise_gt(gt_action, 
                          sigma_trans=0.03,   # 平移
                          sigma_rot=0.02,     # 旋转
                          clip_min=-1.0, clip_max=1.0):
    """
    gt_action: [N, 7] or [7] in [-1, 1]
    顺序: [tx, ty, tz, rx, ry, rz, gripper]
    """
    x = gt_action.clone()
    is_batched = x.dim() == 2
    if not is_batched:
        x = x.unsqueeze(0)

    x[:, 0:3] = x[:, 0:3] + torch.randn_like(x[:, 0:3]) * sigma_trans
    x[:, 3:6] = x[:, 3:6] + torch.randn_like(x[:, 3:6]) * sigma_rot

    # if grip_is_binary:
    #     # 方案A：label smoothing（推荐，连续监督最稳）
    #     x[:, 6] = torch.where(x[:, 6] > 0, torch.full_like(x[:, 6], 0.95), torch.full_like(x[:, 6], -0.95))
    # else:
    #     # 连续夹爪：小噪声
    #     x[:, 6] = x[:, 6] + torch.randn_like(x[:, 6]) * sigma_grip

    x = torch.clamp(x, clip_min, clip_max)
    return x if is_batched else x.squeeze(0)
