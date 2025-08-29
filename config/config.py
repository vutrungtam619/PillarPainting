import os

project_root = os.path.dirname(os.path.dirname(__file__))

config = {
    'DATASET' : {
        'CLASS': {
            'Pedestrian': 0,
            'Cyclist': 1,
            'Car': 2,
        },
        'data_root': ''
    },

    'PILLARPAINTING' : {
        'num_classes': 3,
        'pc_range': [0, -39.68, -3, 69.12, 39.68, 1],
        'voxel_size': [0.16, 0.16, 4],
        'max_voxels': (16000, 40000),
        'max_points': 32
    },

    'BISENET' : {
        'new_size': (384, 1248),
        'num_classes': 19,
        'aux_mode': 'eval',
        'bisenet_ckpt': os.path.join(project_root, 'bisenet_ckpt', 'model_final_v2_city.pth')
    },

    'ANCHOR' : {
        'ranges': [[0, -39.68, -0.6, 69.12, 39.68, -0.6], [0, -39.68, -0.6, 69.12, 39.68, -0.6], [0, -39.68, -1.78, 69.12, 39.68, -1.78]],
        'sizes': [[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]],
        'rotations': [0, 1.57],
    },

    'TRAINING' : {
        'data_root': '',
        'checkpoint_dir': os.path.join(project_root, 'checkpoints'),
        'batch_size': 8,
        'num_workers': 4,
        'init_lr': 0.00025,
        'epoch': 160,
        'ckpt_freq': 2,
        'log_freq': 10,
        'log_dir': os.path.join(project_root, 'log')
    }    
}