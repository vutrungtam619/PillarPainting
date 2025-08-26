import numpy as np
import os
from config.config import config
from torch.utils.data import Dataset
from utils.io import read_pickle, read_points
from utils.process import bbox_camera2lidar
from dataset.data_aug import point_range_filter

project_root = os.path.dirname(os.path.dirname(__file__))

class Kitti(Dataset): 
    def __init__(self, data_root, split):
        assert split in ['train', 'val', 'trainval', 'test']
        self.data_root = data_root
        self.split = split
        self.data_infos = read_pickle(os.path.join(project_root, 'dataset', f'kitti_infos_{split}.pkl'))
        self.sorted_ids = list(self.data_infos.keys())
        self.CLASS = config['DATASET']['CLASS']
        
    def remove_dontcare(self, annos_info):
        keep_ids = [i for i, name in enumerate(annos_info['name']) if name != 'DontCare']
        for k, v in annos_info.items():
            annos_info[k] = v[keep_ids]
        return annos_info   
    
    def __getitem__(self, index):
        data_info = self.data_infos[self.sorted_ids[index]]
        image_info, calib_info, annos_info = data_info['image'], data_info['calib'], data_info['annos']
        
        velodyne_path = data_info['velodyne_path']
        pts = read_points(velodyne_path) # np.float32
        
        Tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
        R0_rect = calib_info['R0_rect'].astype(np.float32)
        
        annos_info = self.remove_dontcare(annos_info)
        annos_name = annos_info['name']
        annos_location = annos_info['location']
        annos_dimension = annos_info['dimensions']
        rotation_y = annos_info['rotation_y']
        gt_bboxes = np.concatenate([annos_location, annos_dimension, rotation_y[:, None]], axis=1).astype(np.float32)
        gt_bboxes_3d = bbox_camera2lidar(gt_bboxes, Tr_velo_to_cam, R0_rect)
        gt_labels = [self.CLASS.get(name, -1) for name in annos_name]
        
        image_path = os.path.join(self.data_root, image_info['image_path'])
        
        data_dict = {
            'pts': pts,
            'gt_bboxes_3d': gt_bboxes_3d,
            'gt_labels': np.array(gt_labels),
            'gt_names': annos_name,
            'difficulty': annos_info['difficulty'],
            'image_info': image_info,
            'image_path': image_path,
            'calib_info': calib_info
        }
        
        data_dict = point_range_filter(data_dict=data_dict, point_range=config['PILLARPAINTING']['pc_range'])
        
        return data_dict
    
    def __len__(self):
        return len(self.data_infos)
        