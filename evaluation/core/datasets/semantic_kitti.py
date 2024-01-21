import os
import numpy as np
import pickle
import torch
import torchsparse
from ..utils import classproperty
from .detection_dataset import DetectionDataset
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.utils import make_ntuple

__all__ = ['SemanticKITTI']


class SemanticKITTI(DetectionDataset):
    def __init__(self, 
            root_dir='data/semantic-kitti',
            max_points=60000,
            subset=False,
            **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.fns = sorted(os.listdir(os.path.join(root_dir, '08', 'velodyne')))
        if subset: self.fns = self.fns[::10]
        self.voxel_size = (0.05, 0.05, 0.05)


    def __len__(self):
        return len(self.fns)
        

    def __getitem__(self, index):

        # index = 2110

        fn = os.path.join(self.root_dir, '08', 'velodyne', self.fns[index])
        points = np.fromfile(fn, dtype=np.float32).reshape(-1, 4)
        points[:, :3] -= points[:, :3].min(0)
        coords, inds = sparse_quantize(points[:, :3], self.voxel_size, return_index=True)
        # subsample
        return {'pts_input': torchsparse.SparseTensor(points[inds], coords)}


