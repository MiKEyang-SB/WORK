import torch
from torch import nn
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor,
)
class BoxRenderer:
    