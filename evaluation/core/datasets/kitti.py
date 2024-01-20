import os
import pickle
import random
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchsparse import SparseTensor
from tqdm import tqdm

from .transform import CompositeTransform, RandomRotation, \
    RandomFlip, RandomScale, RandomGTAugment, RandomSample, Crop

from ..utils import calibration, kitti
from ..utils.roipool3d import roipool3d_utils
from .detection_dataset import DetectionDataset

from ..utils import classproperty

__all__ = ['KITTI']


class KITTIDataset(DetectionDataset):
    def __init__(self,
                 root: str,
                 split: str,
                 max_points: Optional[int] = None) -> None:
        super().__init__()
        self.root = root
        self.split = split
        if max_points is None or max_points < 0:
            max_points = None
        self.max_points = max_points

        self.imageset_dir = os.path.join(
            root, 'object', 'testing' if split == 'test' else 'training')

        self.split_dir = os.path.join(root, 'ImageSets', split + '.txt')
        self.image_idx_list = [
            x.strip() for x in open(self.split_dir).readlines()
        ]

        self.image_dir = os.path.join(self.imageset_dir, 'image_2')
        self.lidar_dir = os.path.join(self.imageset_dir, 'velodyne')
        self.calib_dir = os.path.join(self.imageset_dir, 'calib')
        self.label_dir = os.path.join(self.imageset_dir, 'label_2')
        self.plane_dir = os.path.join(self.imageset_dir, 'planes')

        self.gt_database_pkl = os.path.join(
            self.imageset_dir, 'gt_database',
            '{}_[{}].pkl'.format(split, ','.join(self.classes)))

        if 'train' in self.split and self.gt_aug_enabled:
            self.generate_gt_database()

        self.sample_id_list = [
            int(sample_id) for sample_id in self.image_idx_list
        ]

        self.aug_rot_range = 4

        self.augmentor = CompositeTransform(([
            RandomGTAugment(self.gt_database, self.gt_aug_num_range, self.
                            gt_aug_hard_ratio, self.pc_area_scope, 1.0)
        ] if self.gt_aug_enabled and split == 'train' else []) + [
            RandomRotation(
                np.array(
                    [-np.pi / self.aug_rot_range, np.pi /
                     self.aug_rot_range]), 1.0, -1.0),
            RandomScale(np.array([0.95, 1.05]), 1.0),
            RandomFlip(0.5)
        ])

        self.subsampler = CompositeTransform(
            [Crop(self.loc_range),
             RandomSample(self.max_points)])

        self.include_similar_type = True

    @classproperty
    def classes(self):
        return ['Car', 'Cyclist', 'Pedestrian']

    @classproperty
    def pc_area_scope(self):
        return np.array([[-40, 40], [-1, 3], [0, 70.4]], 'float32')

    @classproperty
    def gt_area_scope(self):
        return {
            'Car': np.array([[-40, 40], [-1, 3], [0, 70.4]], 'float32'),
            'Cyclist': np.array([[-20, 20], [-1, 3], [0, 48]], 'float32'),
            'Pedestrian': np.array([[-20, 20], [-1, 3], [0, 48]], 'float32')
        }

    @classproperty
    def mean_size(self):
        return {
            'Car': np.array([1.56, 1.6, 3.9], 'float32'),
            'Cyclist': np.array([1.73, 0.6, 1.76], 'float32'),
            'Pedestrian': np.array([1.73, 0.6, 0.8], 'float32')
        }

    @classproperty
    def mean_center_z(self):
        return {'Car': 1.0, 'Cyclist': 0.6, 'Pedestrian': 0.6}

    @classproperty
    def gt_aug_enabled(self):
        return True

    @classproperty
    def gt_aug_num_range(self):
        return {'Car': [10, 15], 'Cyclist': [5, 15], 'Pedestrian': [5, 15]}

    @classproperty
    def gt_aug_hard_ratio(self):
        return {'Car': None, 'Cyclist': None, 'Pedestrian': None}

    def get_image_shape(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        im = Image.open(img_file)
        width, height = im.size
        return height, width, 3

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_dir, '%06d.bin' % idx)
        assert os.path.exists(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return calibration.Calibration(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return kitti.get_objects_from_label(label_file)

    def get_road_plane(self, idx):
        plane_file = os.path.join(self.plane_dir, '%06d.txt' % idx)
        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    def filtrate_objects(self, obj_list):
        valid_obj_list = dict([(c, []) for c in self.classes])
        for obj in obj_list:
            if 'train' in self.split and self.include_similar_type:
                if obj.cls_type == 'Van':
                    obj.cls_type = 'Car'
                if obj.cls_type == 'Person_sitting':
                    obj.cls_type = 'Pedestrian'
            if obj.cls_type not in self.classes:
                continue
            if 'train' in self.split and self.reduce_by_range and (
                    not kitti.get_camera_fov_points(
                        pts_rect=np.array([obj.pos]),
                        reduce_by_range=self.reduce_by_range,
                        area_scope=self.pc_area_scope)[0]):
                continue
            valid_obj_list[obj.cls_type].append(obj)
        return valid_obj_list

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, index):
        sample_id = int(self.sample_id_list[index])
        calib = self.get_calib(sample_id)
        img_shape = self.get_image_shape(sample_id)
        pts_lidar = self.get_lidar(sample_id)

        # get valid point (projected points should be in image)
        pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])
        pts_intensity = pts_lidar[:, 3]
        pts_valid_flag = kitti.get_camera_fov_points(
            pts_rect=pts_rect,
            reduce_by_range=self.reduce_by_range,
            area_scope=self.pc_area_scope,
            calib=calib,
            img_shape=img_shape)
        pts_rect = pts_rect[pts_valid_flag][:, 0:3]
        pts_intensity = pts_intensity[pts_valid_flag]

        if 'test' not in self.split:
            gt_obj_list = self.filtrate_objects(self.get_label(sample_id))
            gt_boxes3d = kitti.objs_to_boxes3d_with_classes(
                gt_obj_list, self.classes)

            if 'train' in self.split:
                # augment
                road_plane = self.get_road_plane(sample_id)
                input_dict = dict(road_plane=road_plane,
                                  pts_rect=pts_rect,
                                  pts_intensity=pts_intensity,
                                  gt_boxes3d=gt_boxes3d)
                updated_input_dict = self.augmentor(input_dict)
                pts_rect = updated_input_dict['pts_rect']
                pts_intensity = updated_input_dict['pts_intensity']
                gt_boxes3d = updated_input_dict['gt_boxes3d']

            # ignore orientation pi
            for c in self.classes:
                gt_boxes3d[c][:, 6] %= np.pi

        pts_features = np.reshape(pts_intensity - 0.5, [-1, 1])
        feats = np.concatenate((pts_rect, pts_features), axis=1)
        locs = np.floor((pts_rect - self.pc_area_scope[:, 0]) /
                        self.voxel_size).astype('int32')
        # subsample
        input_dict = dict(feats=feats, locs=locs, pts_rect=pts_rect)

        updated_input_dict = self.subsampler(input_dict)
        feats = updated_input_dict['feats']
        locs = updated_input_dict['locs']
        pts_rect = updated_input_dict['pts_rect']

        # reduce average feats by locs
        if self.max_points is not None:
            feats_with_cnt = np.concatenate(
                [np.ones_like(feats[:, :1]), feats],
                axis=-1
            )
            flatten = torch.sparse.FloatTensor(
                torch.from_numpy(locs).t().long(),
                torch.from_numpy(feats_with_cnt)).coalesce()
            locs = flatten.indices().t().int().numpy()
            feats_with_cnt = flatten.values().numpy()
            feats = feats_with_cnt[:, 1:] / feats_with_cnt[:, :1]

        sample_info = dict()
        sample_info['sample_id'] = sample_id
        sample_info['calib'] = calib
        sample_info['img_shape'] = img_shape
        sample_info['pts_rect'] = pts_rect
        if 'test' not in self.split:
            sample_info['gt_boxes3d'] = gt_boxes3d

        sample_info['pts_input'] = SparseTensor(feats, locs)
        return sample_info

    def generate_gt_database(self):
        if not os.path.exists(self.gt_database_pkl):
            gt_database = dict([(c, []) for c in self.classes])
            for sample_id in tqdm(self.image_idx_list):
                sample_id = int(sample_id)

                pts_lidar = self.get_lidar(sample_id)
                calib = self.get_calib(sample_id)
                pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])
                pts_intensity = pts_lidar[:, 3]

                obj_list = self.filtrate_objects(self.get_label(sample_id))
                gt_boxes3d_dict = kitti.objs_to_boxes3d_with_classes(
                    obj_list, self.classes)

                for c, gt_boxes3d in gt_boxes3d_dict.items():
                    if gt_boxes3d.__len__() == 0:
                        continue
                    boxes_pts_mask_list = roipool3d_utils.pts_in_boxes3d_cpu(
                        torch.from_numpy(pts_rect),
                        torch.from_numpy(gt_boxes3d))

                    for k in range(boxes_pts_mask_list.__len__()):
                        pt_mask_flag = (boxes_pts_mask_list[k].numpy() == 1)
                        cur_pts = pts_rect[pt_mask_flag].astype(np.float32)
                        cur_pts_intensity = pts_intensity[pt_mask_flag].astype(
                            np.float32)
                        sample_dict = {
                            'sample_id': sample_id,
                            'gt_box3d': gt_boxes3d[k],
                            'points': cur_pts,
                            'intensity': cur_pts_intensity
                        }
                        gt_database[c].append(sample_dict)

            with open(self.gt_database_pkl, 'wb') as f:
                pickle.dump(gt_database, f)
            #logger.info('Save GT database into {}'.format(
            #    self.gt_database_pkl))

        self.gt_database = pickle.load(open(self.gt_database_pkl, 'rb'))
        for c, gt_database in self.gt_database.items():
            if self.gt_aug_hard_ratio[c] is not None:
                easy_list, hard_list = [], []
                for obj in gt_database:
                    if obj['points'].shape[0] > 100:
                        easy_list.append(obj)
                    else:
                        hard_list.append(obj)
                self.gt_database[c] = [easy_list, hard_list]
                #logger.info(
                #    'Loading gt_database {} (easy (pt_num > 100): {}, hard (pt_num <= 100): {}) from {}'
                #    .format(c, len(easy_list), len(hard_list),
                #            self.gt_database_pkl))


class KITTI(dict):
    def __init__(self, root: str, max_points: Optional[int] = None) -> None:
        super().__init__({
            split: KITTIDataset(root, split=split, max_points=max_points)
            for split in ['train', 'val']
        })
