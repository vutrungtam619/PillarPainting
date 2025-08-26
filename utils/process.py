import copy
import numba
import numpy as np
import random
import torch
import torch.nn.functional as F

def setup_seed(seed=0, deterministic = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def bbox_camera2lidar(bboxes, tr_velo_to_cam, r0_rect):
    """ Convert bouding box from camera coordinate to lidar coordinate
    Args:
        bboxes [np.ndarray float32, (n, 7)]: bouding box in camera coordinate
        Tr_velo_to_cam [np.ndarray float32, (4, 4)]: convert lidar point coordinate to 3D camera coordinate
        R0_rect [np.ndarray float32, (4, 4)]: rectification matrix, convert camera coordinate to camera rectified coordinate

    Returns:
        bboxes_lidar [np.ndarray float32, (n, 7)]: bouding box in lidar coordinate
    """
    x_size, y_size, z_size = bboxes[:, 3:4], bboxes[:, 4:5], bboxes[:, 5:6]
    xyz_size = np.concatenate([z_size, x_size, y_size], axis=1)
    extended_xyz = np.pad(bboxes[:, :3], ((0, 0), (0, 1)), 'constant', constant_values=1.0)
    rt_mat = np.linalg.inv(r0_rect @ tr_velo_to_cam)
    xyz = extended_xyz @ rt_mat.T
    bboxes_lidar = np.concatenate([xyz[:, :3], xyz_size, bboxes[:, 6:]], axis=1)
    return np.array(bboxes_lidar, dtype=np.float32)

def bbox3d2corners(bboxes):
    '''
           ^ z   x            6 ------ 5
           |   /             / |     / |
           |  /             2 -|---- 1 |   
    y      | /              |  |     | | 
    <------|o               | 7 -----| 4
                            |/   o   |/    
                            3 ------ 0 
    x: front, y: left, z: top
    
    Args:
        bboxes [np.ndarray float32, (n, 7)]: bouding box in lidar coordinate
    
    Returns:
        bboxes_corners [np.ndarray float32, (n, 8, 3)]: corners in lidar coordinate
    '''
    centers, dims, angles = bboxes[:, :3], bboxes[:, 3:6], bboxes[:, 6]

    # 1.generate bbox corner coordinates, clockwise from minimal point
    bboxes_corners = np.array([[-0.5, -0.5, 0], [-0.5, -0.5, 1.0], [-0.5, 0.5, 1.0], [-0.5, 0.5, 0.0],
                               [0.5, -0.5, 0], [0.5, -0.5, 1.0], [0.5, 0.5, 1.0], [0.5, 0.5, 0.0]], 
                               dtype=np.float32)
    bboxes_corners = bboxes_corners[None, :, :] * dims[:, None, :] # (1, 8, 3) * (n, 1, 3) -> (n, 8, 3)

    # 2. rotate around z axis
    rot_sin, rot_cos = np.sin(angles), np.cos(angles)
    # in fact, -angle
    rot_mat = np.array([[rot_cos, rot_sin, np.zeros_like(rot_cos)],
                        [-rot_sin, rot_cos, np.zeros_like(rot_cos)],
                        [np.zeros_like(rot_cos), np.zeros_like(rot_cos), np.ones_like(rot_cos)]], 
                        dtype=np.float32) # (3, 3, n)
    rot_mat = np.transpose(rot_mat, (2, 1, 0)) # (n, 3, 3)
    bboxes_corners = bboxes_corners @ rot_mat # (n, 8, 3)

    # 3. translate to centers
    bboxes_corners += centers[:, None, :]
    return bboxes_corners

def limit_period(val, offset=0.5, period=np.pi):
    """
    val: array or float
    offset: float
    period: float
    return: Value in the range of [-offset * period, (1-offset) * period]
    """
    limited_val = val - np.floor(val / period + offset) * period
    return limited_val

def nearest_bev(bboxes):
    '''
    bboxes: (n, 7), (x, y, z, w, l, h, theta)
    return: (n, 4), (x1, y1, x2, y2)
    '''    
    bboxes_bev = copy.deepcopy(bboxes[:, [0, 1, 3, 4]])
    bboxes_angle = limit_period(bboxes[:, 6].cpu(), offset=0.5, period=np.pi).to(bboxes_bev)
    bboxes_bev = torch.where(torch.abs(bboxes_angle[:, None]) > np.pi / 4, bboxes_bev[:, [0, 1, 3, 2]], bboxes_bev)
    
    bboxes_xy = bboxes_bev[:, :2]
    bboxes_wl = bboxes_bev[:, 2:]
    bboxes_bev_x1y1x2y2 = torch.cat([bboxes_xy - bboxes_wl / 2, bboxes_xy + bboxes_wl / 2], dim=-1)
    return bboxes_bev_x1y1x2y2

def iou2d(bboxes1, bboxes2, metric=0):
    '''
    bboxes1: (n, 4), (x1, y1, x2, y2)
    bboxes2: (m, 4), (x1, y1, x2, y2)
    return: (n, m)
    '''
    bboxes_x1 = torch.maximum(bboxes1[:, 0][:, None], bboxes2[:, 0][None, :]) # (n, m)
    bboxes_y1 = torch.maximum(bboxes1[:, 1][:, None], bboxes2[:, 1][None, :]) # (n, m)
    bboxes_x2 = torch.minimum(bboxes1[:, 2][:, None], bboxes2[:, 2][None, :])
    bboxes_y2 = torch.minimum(bboxes1[:, 3][:, None], bboxes2[:, 3][None, :])

    bboxes_w = torch.clamp(bboxes_x2 - bboxes_x1, min=0)
    bboxes_h = torch.clamp(bboxes_y2 - bboxes_y1, min=0)

    iou_area = bboxes_w * bboxes_h # (n, m)
    
    bboxes1_wh = bboxes1[:, 2:] - bboxes1[:, :2]
    area1 = bboxes1_wh[:, 0] * bboxes1_wh[:, 1] # (n, )
    bboxes2_wh = bboxes2[:, 2:] - bboxes2[:, :2]
    area2 = bboxes2_wh[:, 0] * bboxes2_wh[:, 1] # (m, )
    if metric == 0:
        iou = iou_area / (area1[:, None] + area2[None, :] - iou_area + 1e-8)
    elif metric == 1:
        iou = iou_area / (area1[:, None] + 1e-8)
    return iou

def iou2d_nearest(bboxes1, bboxes2):
    '''
    bboxes1: (n, 7), (x, y, z, w, l, h, theta)
    bboxes2: (m, 7),
    return: (n, m)
    '''
    bboxes1_bev = nearest_bev(bboxes1)
    bboxes2_bev = nearest_bev(bboxes2)
    iou = iou2d(bboxes1_bev, bboxes2_bev)
    return iou

def group_rectangle_vertexs(bboxes_corners):
    """ Extract a bounding box into rectangles
    Args:
        bboxes_corners [np.ndarray float32, (N, 8, 3)]: 

    Returns:
        group_rectangle_vertexs [np.ndarray float32, (N, 6, 4, 3)]: each box have 6 surfaces, each surface have 4 corners, each corners have 3 coordinate x, y, z
    """
    rec1 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 1], bboxes_corners[:, 3], bboxes_corners[:, 2]], axis=1) # (n, 4, 3)
    rec2 = np.stack([bboxes_corners[:, 4], bboxes_corners[:, 7], bboxes_corners[:, 6], bboxes_corners[:, 5]], axis=1) # (n, 4, 3)
    rec3 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 4], bboxes_corners[:, 5], bboxes_corners[:, 1]], axis=1) # (n, 4, 3)
    rec4 = np.stack([bboxes_corners[:, 2], bboxes_corners[:, 6], bboxes_corners[:, 7], bboxes_corners[:, 3]], axis=1) # (n, 4, 3)
    rec5 = np.stack([bboxes_corners[:, 1], bboxes_corners[:, 5], bboxes_corners[:, 6], bboxes_corners[:, 2]], axis=1) # (n, 4, 3)
    rec6 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 3], bboxes_corners[:, 7], bboxes_corners[:, 4]], axis=1) # (n, 4, 3)
    group_rectangle_vertexs = np.stack([rec1, rec2, rec3, rec4, rec5, rec6], axis=1)
    return group_rectangle_vertexs
    
def frustum_camera2lidar(frustum, Tr_velo_to_cam, R0_rect):
    """ Convert frustum camera coordinate to lidar coordinate
    Args:
        frustum [np.ndarray float32, (N, 8, 3)]: coordinates of frustum corners
        Tr_velo_to_cam [np.ndarray float32, (4, 4)]: convert lidar point coordinate to 3D camera coordinate
        R0_rect [np.ndarray float32, (4, 4)]: rectification matrix, convert camera coordinate to camera rectified coordinate
   
    Returns:
        frustum_xyz [np.ndarray float32, (N, 8, 3)]: frustum corners in xyz
    """
    extended_xyz = np.pad(frustum, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=1.0)
    rt_mat = np.linalg.inv(R0_rect @ Tr_velo_to_cam)
    frustum_xyz = extended_xyz @ rt_mat.T
    return frustum_xyz[..., :3]

def group_plane_equation(group_rectangle_vertexs):
    """ Plane equation: Ax + By + Cz + D = 0 
    Args: 
        group_rectangle_vertexs [np.ndarray float32, (N, 6, 4, 3)]: each box have 6 surfaces, each surface have 4 corners, each corners have 3 coordinate x, y, z
        
    Returns:
        plane_equation_params [np.ndarray float32, (N, 6, 4)]: each box have 6 surfaces, each surface have 4 parameters (A, B, C, D)
    """
    vectors = group_rectangle_vertexs[:, :, :2] - group_rectangle_vertexs[:, :, 1:3]
    normal_vectors = np.cross(vectors[:, :, 0], vectors[:, :, 1]) # (n, 6, 3)
    normal_d = np.einsum('ijk,ijk->ij', group_rectangle_vertexs[:, :, 0], normal_vectors) # (n, 6)
    plane_equation_params = np.concatenate([normal_vectors, -normal_d[:, :, None]], axis=-1)
    return plane_equation_params

def points_in_bboxes(points, plane_equation_params):
    """ Create mask to filter points in bounding box. N is number of points, n is number of bounding box.
    Args
        points [np.ndarray float32, (N, 4)]: total points
        plane_equation_params [np.ndarray float32, (N, 6, 4)]: each box have 6 surfaces, each surface have 4 parameters (A, B, C, D)
    
    Returns:
        mask [np.ndarray bool, (N, n)]: true if point inside 6 surface, false if point outside at least 1 surface
    """
    N, n = len(points), len(plane_equation_params)
    m = plane_equation_params.shape[1]
    masks = np.ones((N, n), dtype=np.bool_)
    for i in range(N):
        x, y, z = points[i, :3]
        for j in range(n):
            bbox_plane_equation_params = plane_equation_params[j]
            for k in range(m):
                a, b, c, d = bbox_plane_equation_params[k]
                if a * x + b * y + c * z + d >= 0:
                    masks[i][j] = False
                    break
    return masks

def points_in_bboxes_v2(points, R0_rect, Tr_velo_to_cam, dimensions, location, rotation_y, name):
    """
    Args:
        points [np.ndarray float32, (N, 4)]: total points
        R0_rect [np.ndarray float32, (4, 4)]: rectification matrix, convert camera coordinate to camera rectified coordinate
        Tr_velo_to_cam [np.ndarray float32, (4, 4)]: convert lidar point coordinate to 3D camera coordinate
        dimensions [np.ndarray float32, (n, 3)]: 3d dimension in legnth, height, width
        location [np.ndarray float32, (n, 3)]: 3d location of the object center in camera coordinate, include x, y, z (right, down, forward)
        name [np.ndarray string, (1, )]: name of the object category in image, include Car, Pedestrian, Cyclist, Dontcare

    Returns:
        indices [np.ndarray bool, (N, n_valid_box)]: bool mask to determine if point in bounding box
        n_total_bbox [constant int]: number of boxes
        n_valid_bbox [constant int]: number of valid boxes (remove Dontcare)
        bboxes_lidar [np.ndarray float32, (n, 7)]: bouding box in lidar coordinate
        name [np.ndarray string, (n, )]: name of the object category in image, include Car, Pedestrian, Cyclist
    """
    n_total_bbox = len(dimensions) # Number of box
    n_valid_bbox = len([item for item in name if item != 'DontCare']) # Remove Dontcare box
    location, dimensions = location[:n_valid_bbox], dimensions[:n_valid_bbox]
    rotation_y, name = rotation_y[:n_valid_bbox], name[:n_valid_bbox]
    bboxes_camera = np.concatenate([location, dimensions, rotation_y[:, None]], axis=1) # Concat to from bouding box element, shape (n, 7)
    bboxes_lidar = bbox_camera2lidar(bboxes_camera, Tr_velo_to_cam, R0_rect)
    bboxes_corners = bbox3d2corners(bboxes_lidar)
    group_rectangle_vertexs_v = group_rectangle_vertexs(bboxes_corners)
    frustum_surfaces = group_plane_equation(group_rectangle_vertexs_v)
    indices = points_in_bboxes(points[:, :3], frustum_surfaces) # (N, n), N is points num, n is bboxes number
    return indices, n_total_bbox, n_valid_bbox, bboxes_lidar, name

def get_points_num_in_bbox(points, R0_rect, Tr_velo_to_cam, dimensions, location, rotation_y, name):
    '''
    Args:
        points [np.ndarray float32, (N, 4)]: total points
        R0_rect [np.ndarray float32, (4, 4)]: rectification matrix, convert camera coordinate to camera rectified coordinate
        Tr_velo_to_cam [np.ndarray float32, (4, 4)]: convert lidar point coordinate to 3D camera coordinate
        dimensions [np.ndarray float32, (n, 3)]: 3d dimension in legnth, height, width
        location [np.ndarray float32, (n, 3)]: 3d location of the object center in camera coordinate, include x, y, z (right, down, forward)
        name [np.ndarray string, (n, )]: name of the object category in image, include Car, Pedestrian, Cyclist, Dontcare
        
    Returns:
        points_num [np.ndarray int32, (n, )]: number of points in bouding box
    '''
    indices, n_total_bbox, n_valid_bbox, bboxes_lidar, name = points_in_bboxes_v2(points=points, R0_rect=R0_rect, Tr_velo_to_cam=Tr_velo_to_cam, dimensions=dimensions, location=location, rotation_y=rotation_y, name=name)
    points_num = np.sum(indices, axis=0)
    non_valid_points_num = [-1] * (n_total_bbox - n_valid_bbox)
    points_num = np.concatenate([points_num, non_valid_points_num], axis=0)
    return np.array(points_num, dtype=np.int32)

def remove_outside_points(points, R0_rect, Tr_velo_to_cam, P2, image_shape):
    """ Remove points which are outside of image
    Args:
        points [np.ndarray float32, (N, 4)]: total points
        R0_rect [np.ndarray float32, (4, 4)]: rectification matrix, convert camera coordinate to camera rectified coordinate
        Tr_velo_to_cam [np.ndarray float32, (4, 4)]: convert lidar point coordinate to 3D camera coordinate
        P2 [np.ndarray float32, (4, 4)]: projection matrix of CAM2, convert 3D camera coordinate to 2D pixel coordinate
        image_shape [tuple int, (2, )]: shape of image
    
    Returns:
        points [np.ndarray float32, (N, 4)]: filtered points
    """
    C, R, T = projection_matrix_to_CRT_kitti(P2)
    image_bbox = [0, 0, image_shape[1], image_shape[0]]
    frustum = get_frustum(image_bbox, C)
    frustum -= T
    frustum = np.linalg.inv(R) @ frustum.T # (3, 8)
    frustum = frustum_camera2lidar(frustum.T[None, ...], Tr_velo_to_cam, R0_rect) # (1, 8, 3)
    group_rectangle_vertexs_v = group_rectangle_vertexs(frustum)
    frustum_surfaces = group_plane_equation(group_rectangle_vertexs_v)
    indices = points_in_bboxes(points[:, :3], frustum_surfaces) # (N, 1) in this case we only have 1 frustum from image
    points = points[indices.reshape([-1])]
    return points  

def projection_matrix_to_CRT_kitti(proj_matrix):
    """ Split projection matrix of Kitti
    P = C @ [R | T]
    Args:
        proj_matrix [np.ndarray float32, (4, 4)]: projection matrix, convert 3D camera coordinate to 2D pixel coordinate
    
    Retruns: tuple of following keys
        C [np.ndarray float32, (3, 3)]: intrinsic matrix
        R [np.ndarray float32, (3, 3)]: rotation matrix
        T [np.ndarray float32, (3, )]: translation matrix        
    """
    CR = proj_matrix[0:3, 0:3]
    CT = proj_matrix[0:3, 3]
    RinvCinv = np.linalg.inv(CR)
    Rinv, Cinv = np.linalg.qr(RinvCinv)
    C = np.linalg.inv(Cinv)
    R = np.linalg.inv(Rinv)
    T = Cinv @ CT
    return C, R, T   

def get_frustum(bbox_image, C, near_clip=0.001, far_clip=100):
    """ Get frustum (8 corners) in camera coordinates
    Args:
        bbox_image [list int]: box in image coordinates
        C [np.ndarray]: intrinsics matrix
        near_clip [float, optional]: nearest distance of frustum
        far_clip [float, optional]: farthest distance of frustum
        
    Returns:
        frustum [np.ndarray float32, (8, 3)]: coordinates of frustum corners
    """
    fku = C[0, 0]
    fkv = -C[1, 1]
    u0v0 = C[0:2, 2]
    z_points = np.array(
        [near_clip] * 4 + [far_clip] * 4, dtype=C.dtype)[:, np.newaxis]
    b = bbox_image
    box_corners = np.array([[b[0], b[1]], [b[0], b[3]], [b[2], b[3]], [b[2], b[1]]], dtype=C.dtype)
    near_box_corners = (box_corners - u0v0) / np.array([fku / near_clip, -fkv / near_clip], dtype=C.dtype)
    far_box_corners = (box_corners - u0v0) / np.array([fku / far_clip, -fkv / far_clip], dtype=C.dtype)
    ret_xy = np.concatenate([near_box_corners, far_box_corners], axis=0)  # [8, 2]
    frustum = np.concatenate([ret_xy, z_points], axis=1)
    return frustum   

class ToTensor(object):
    def __init__(self, mean = (0, 0, 0), std = (1, 1, 1)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """Convert a numpy image and label to a tensor.
        Args:
            img [np.ndarray uint8, (H, W, C)]: image should be a numpy array of shape (H, W, C) 
            
        Returns:
            img_tensor [torch.tensor float32, (C, H, W)]: image in tensor type
        """
        
        img = img.transpose(2, 0, 1).astype(np.float32).copy() # HWC to CHW
        img = torch.from_numpy(img).div_(255.0) # Normalize to [0, 1]
        dtype, device = img.dtype, img.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)[:, None, None]
        std = torch.as_tensor(self.std, dtype=dtype, device=device)[:, None, None]
        img_tensor = img.sub_(mean).div_(std).clone()
        
        return img_tensor
    
def image_to_tensor(img):
    to_tensor = ToTensor(mean=(0.36783523, 0.38706144, 0.3754649), std=(0.31566228, 0.31997792, 0.32575161)) 
    img_tensor = to_tensor(img).cuda()
    return img_tensor

def project_point_to_camera(self, point, calib):
    """ Project point to camera image coordinates.
    Args: 
        point (torch.Tensor): shape (P1 + P2 + ... + Pb, 3) where P1 + P2 + ... + Pb is the total number of pillars in the batch.
        calib (dict): calibration dictionary containing 'Tr_velo_to_cam', 'R0_rect', and 'P2' matrices, each is 4x4 homo matrix
        
    Returns:
        u (torch.Tensor): horizontal pixel coordinates in the camera image. Shape (P1 + P2 + ... + Pb,).
        v (torch.Tensor): vertical pixel coordinates in the camera image. Shape (P1 + P2 + ... + Pb,).     
    """        
    device = point.device
    N = point.shape[0]

    Tr = torch.as_tensor(calib['Tr_velo_to_cam'], device=device, dtype=torch.float32)  
    R0 = torch.as_tensor(calib['R0_rect'], device=device, dtype=torch.float32)     
    P2 = torch.as_tensor(calib['P2'], device=device, dtype=torch.float32)              

    pts_velo_hom = F.pad(point, (0, 1), value=1.0)  # [x, y, z, 1]
    RT = R0 @ Tr
    pts_cam_rect_hom = (RT @ pts_velo_hom.T).T
    pts_img_hom = (P2 @ pts_cam_rect_hom.T).T
    
    u = pts_img_hom[:, 0] / pts_img_hom[:, 2]
    v = pts_img_hom[:, 1] / pts_img_hom[:, 2]
    
    return u, v