import argparse
import cv2
import numpy as np
import os
import torch

from utils import read_points, read_calib, read_label, \
    keep_bbox_from_image_range, keep_bbox_from_lidar_range, \
    vis_img_3d, bbox3d2corners_camera, points_camera2image, \
    bbox_camera2lidar
from model import PillarPainting, BiSeNetV2


def point_range_filter(pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    return pts[keep_mask]


def process_sample(idx, args, model, device, CLASSES, pcd_limit_range, save_dir):
    pc_path = os.path.join(args.pc_dir, f"{idx:06d}.bin")
    calib_path = os.path.join(args.calib_dir, f"{idx:06d}.txt")
    gt_path = os.path.join(args.gt_dir, f"{idx:06d}.txt")
    img_path = os.path.join(args.img_dir, f"{idx:06d}.png")

    if not os.path.exists(pc_path):
        print(f"Không tìm thấy file {pc_path}, dừng lại.")
        return False

    pc = read_points(pc_path)
    pc = point_range_filter(pc)
    pc_torch = torch.from_numpy(pc).to(device)

    calib_info = read_calib(calib_path) if os.path.exists(calib_path) else None
    gt_label = read_label(gt_path) if os.path.exists(gt_path) else None
    img = cv2.imread(img_path, 1) if os.path.exists(img_path) else None

    with torch.no_grad():
        result_filter = model(
            batched_pts=[pc_torch],
            batched_image_paths=[img_path],
            batched_calibs=[calib_info],
            mode='test'
        )[0]

    if calib_info is not None and img is not None:
        tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
        r0_rect = calib_info['R0_rect'].astype(np.float32)
        P2 = calib_info['P2'].astype(np.float32)
        image_shape = img.shape[:2]
        result_filter = keep_bbox_from_image_range(result_filter, tr_velo_to_cam, r0_rect, P2, image_shape)

    result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)
    labels = result_filter['labels']

    # Vẽ prediction
    if calib_info is not None and img is not None:
        bboxes2d, camera_bboxes = result_filter['bboxes2d'], result_filter['camera_bboxes']
        bboxes_corners = bbox3d2corners_camera(camera_bboxes)
        image_points = points_camera2image(bboxes_corners, P2)
        img = vis_img_3d(img, image_points, labels, rt=True)

    # Vẽ ground truth
    if calib_info is not None and gt_label is not None and img is not None:
        tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
        r0_rect = calib_info['R0_rect'].astype(np.float32)
        dimensions = gt_label['dimensions']
        location = gt_label['location']
        rotation_y = gt_label['rotation_y']
        gt_labels = np.array([CLASSES.get(item, -1) for item in gt_label['name']])
        sel = gt_labels != -1
        bboxes_camera = np.concatenate([location, dimensions, rotation_y[:, None]], axis=-1)
        bboxes_camera = bboxes_camera[sel]
        gt_labels = [-1] * len(bboxes_camera)

        if len(bboxes_camera) > 0:
            bboxes_corners = bbox3d2corners_camera(bboxes_camera)
            image_points = points_camera2image(bboxes_corners, P2)
            img = vis_img_3d(img, image_points, gt_labels, rt=True)

    if img is not None:
        cv2.imshow('3D Detection', img)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):  # lưu và chuyển tiếp
                save_path = os.path.join(save_dir, f"{idx:06d}.png")
                cv2.imwrite(save_path, img)
                print(f"Đã lưu ảnh vào {save_path}")
                return True
            elif key == 27:  # ESC để thoát
                return False

    return True


def main(args):
    CLASSES = {'Pedestrian': 0, 'Cyclist': 1, 'Car': 2}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)

    # Tạo thư mục lưu ảnh
    save_dir = os.path.join(os.path.dirname(__file__), 'image')
    os.makedirs(save_dir, exist_ok=True)

    # Load model
    bisenet = BiSeNetV2().eval().to(device)
    model = PillarPainting(nclasses=len(CLASSES), bisenet=bisenet).to(device)
    checkpoint_dict = torch.load(args.ckpt)
    model.load_state_dict(checkpoint_dict['checkpoint'])
    model.eval()

    idx = args.start_idx
    while True:
        if not process_sample(idx, args, model, device, CLASSES, pcd_limit_range, save_dir):
            break
        idx += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints/epoch_100.pth')
    parser.add_argument('--pc_dir', default='dataset/velodyne_reduced/training')
    parser.add_argument('--calib_dir', default='kitti/training/calib')
    parser.add_argument('--gt_dir', default='kitti/training/label_2')
    parser.add_argument('--img_dir', default='kitti/training/image_2')
    parser.add_argument('--start_idx', type=int, default=100, help='Sample bắt đầu')
    args = parser.parse_args()

    main(args)
