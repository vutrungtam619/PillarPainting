import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from utils.io import read_calib, read_points, read_label, write_points, write_pickle
from utils.process import remove_points_out_image, get_points_num_in_bbox

project_root = os.path.dirname(os.path.dirname(__file__))

def judge_difficulty(annotation_dict):
    """ classify difficulty level of each object base on KITTI bench mark
    Args: Dict with the following key
        name [np.ndarray string, (1, )]: name of the object category in image, include Car, Pedestrian, Cyclist, Dontcare
        truncated [np.ndarray float32, (1, )]: how much the object extend outside the image, from 0.0 -> 1.0
        occluded [np.ndarray float32, (1, )]: how much the objet is block by others, from 0 -> 3 (fully visible -> unknow)
        alpha [np.ndarray float32, (1, )]: observation agle of the object in camera coordinate (radian)
        bbox [np.ndarray float32, (4, )]: 2d bounding box in x_min, y_min, x_max, y_max
        dimensions [np.ndarray float32, (3, )]: 3d dimension in legnth, height, width
        location [np.ndarray float32, (3, )]: 3d location of the object center in camera coordinate, include x, y, z (right, down, forward)
        rotation_y [np.ndarray float32, (1, )]: rotation of the object around y-axis (radian)

    Returns:
        difficultys [np.ndarray float32, (O, )]: O is the number of object in sample
    """
    truncated = annotation_dict['truncated']
    occluded = annotation_dict['occluded']
    bbox = annotation_dict['bbox']
    height = bbox[:, 3] - bbox[:, 1]

    MIN_HEIGHTS = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.30, 0.50]
    difficultys = []
    for h, o, t in zip(height, occluded, truncated):
        difficulty = -1
        for i in range(2, -1, -1):
            if h > MIN_HEIGHTS[i] and o <= MAX_OCCLUSION[i] and t <= MAX_TRUNCATION[i]:
                difficulty = i
        difficultys.append(difficulty)
    return np.array(difficultys, dtype=np.int32)

def create_data_info_pkl(data_root, data_type, label):
    print(f"Processing {data_type} data into pkl file....")
    
    sep = os.path.sep
    
    ids_file = os.path.join(project_root, 'dataset', 'index', f'{data_type}.txt') # Path to txt file include id
    with open(ids_file, 'r') as f:
        ids = [id.strip() for id in f.readlines()] # List of id in string 
        
    split = 'training' if label else 'testing'
    
    # create folder to save velodyne_reduced
    velodyne_reduced_folder = os.path.join(project_root, 'dataset', 'velodyne_reduced', split)
    os.makedirs(velodyne_reduced_folder, exist_ok=True)
            
    kitti_infos_dict = {}
    for id in tqdm(ids): 
        cur_info_dict = {}
        image_path = os.path.join(data_root, split, 'image_2', f'{id}.png')
        lidar_path = os.path.join(data_root, split, 'velodyne', f'{id}.bin')
        calib_path = os.path.join(data_root, split, 'calib', f'{id}.txt')
        
        image = cv2.imread(image_path)
        image_shape = image.shape[:2]
        cur_info_dict['image'] = {
            'image_shape': image_shape,
            'image_path': sep.join(image_path.split(sep)[-3:]), # Example training/image_2/000001.png
            'image_idx': int(id),
        }
        
        calib_dict = read_calib(calib_path)
        cur_info_dict['calib'] = calib_dict
        
        # read lidar point and filter the point outside of image frustum
        lidar_points = read_points(lidar_path)
        reduced_points = remove_points_out_image(lidar_points, calib_dict['R0_rect'], calib_dict['Tr_velo_to_cam'], calib_dict['P2'], image_shape)
        
        # write the reduced_points to bin file
        velodyne_reduced_file = os.path.join(velodyne_reduced_folder, f'{id}.bin')
        write_points(velodyne_reduced_file, reduced_points)
        cur_info_dict['velodyne_path'] = velodyne_reduced_file
        
        if label:
            label_path = os.path.join(data_root, split, 'label_2', f'{id}.txt')
            annotation_dict = read_label(label_path)
            annotation_dict['difficulty'] = judge_difficulty(annotation_dict)
            annotation_dict['num_points_in_gt'] = get_points_num_in_bbox(points=reduced_points, r0_rect=calib_dict['R0_rect'], tr_velo_to_cam=calib_dict['Tr_velo_to_cam'], dimensions=annotation_dict['dimensions'], location=annotation_dict['location'], rotation_y=annotation_dict['rotation_y'], name=annotation_dict['name'])
            cur_info_dict['annos'] = annotation_dict
            
        kitti_infos_dict[int(id)] = cur_info_dict
    
    save_pkl_path = os.path.join(project_root, 'dataset', f'kitti_infos_{data_type}.pkl')
    write_pickle(save_pkl_path, kitti_infos_dict)       
    
    return kitti_infos_dict
    
def main(args):
    data_root = args.data_root
    
    ## 1. train: create data infomation pkl file && create reduced point clouds && create database (points in gt bbox) for data aumentation
    kitti_train_infos_dict = create_data_info_pkl(data_root, data_type='train', label=True)

    ## 2. val: create data infomation pkl file && create reduced point clouds
    kitti_val_infos_dict = create_data_info_pkl(data_root, data_type='val', label=True)
    
    ## 3. trainval: create data infomation pkl file
    kitti_trainval_infos_dict = {**kitti_train_infos_dict, **kitti_val_infos_dict}
    
    saved_path = os.path.join(project_root, 'dataset', f'kitti_infos_trainval.pkl')
    
    write_pickle(saved_path, kitti_trainval_infos_dict)
    
    print("......Processing finished!!!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset infomation')
    parser.add_argument('--data_root', default='', help='your data root for kitti')
    args = parser.parse_args()

    main(args)