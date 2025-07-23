# import numpy as np
# from diffuser.utils import normalise_quat
# import torch
# from scipy.interpolate import CubicSpline, interp1d

# class TrajectoryInterpolator:
#     def __init__(self, use = False, interpolation_length = 50):
#         self._use = use
#         self._interpolation_length = interpolation_length
#     def __call__(self, trajectory):
#         if not self._use:
#             return trajectory
#         trajectory = trajectory.numpy()
#         old_num_steps = len(trajectory)

#         old_steps = np.linspace(0, 1, old_num_steps)
#         new_steps = np.linspace(0, 1, self._interpolation_length)

#         resampled = np.empty((self._interpolation_length, trajectory.shape[1]))
#         #创建一个a*b的数组
#         for i in range(trajectory.shape[1]):
#             if i == (trajectory.shape[1] - 1):  # gripper opening
#                 interpolator = interp1d(old_steps, trajectory[:, i])
#             else:
#                 interpolator = CubicSpline(old_steps, trajectory[:, i]) #三次样条插值

#             resampled[:, i] = interpolator(new_steps) #计算插值

#         resampled = torch.tensor(resampled)
#         if trajectory.shape[1] == 8:
#             resampled[:, 3:7] = normalise_quat(resampled[:, 3:7])
#         return resampled

