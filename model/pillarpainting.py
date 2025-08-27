import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import config
from utils.anchor import Anchors, anchor_target, anchors2bboxes
from utils import image_to_tensor, project_point_to_camera, limit_period
from package import Voxelization, nms_cuda


class PillarLayer(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super(PillarLayer, self).__init__()
        self.voxel_layer = Voxelization(voxel_size=voxel_size, point_cloud_range=point_cloud_range, max_num_points=max_num_points, max_voxels=max_voxels)

    @torch.no_grad()
    def forward(self, batched_pts):
        """ generate pillar from points
        Args:
            batched_pts [list torch.tensor float32, (N, 4)]: list of batch points, each batch have shape (N, 4)
            
        Returns:
            pillars [torch.tensor float32, (p1 + p2 + ... + pb, N, c)]: c is number of features per point
            coors_batch [torch.tensor int64, (p1 + p2 + ... + pb, 1 + 3)]: coordinate of each pillar, 1 is the batch index
            num_points_per_pillar [torch.tensor int64, (p1 + p2 + ... + pb, )]: number of points in each pillar
        """
        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
            voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts) 
            # voxels_out: (max_voxel, num_points, c), coors_out: (max_voxel, 3)
            # num_points_per_voxel_out: (max_voxel, )
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out)
        
        pillars = torch.cat(pillars, dim=0) # (p1 + p2 + ... + pb, num_points, c)
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0) # (p1 + p2 + ... + pb, )
        coors_batch = []
        for i, cur_coors in enumerate(coors):
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i))
        coors_batch = torch.cat(coors_batch, dim=0) # (p1 + p2 + ... + pb, 1 + 3)

        return pillars, coors_batch, npoints_per_pillar

class SegmentLayer(nn.Module):
    def __init__(self, bisenet, new_size):
        super(SegmentLayer, self).__init__()
        self.bisenet = bisenet
        self.new_size = new_size
        self.device = next(bisenet.parameters()).device
        
    def forward(self, batch_image_paths):
        """
        Args:
            batch_image_paths [list string]: path to all batch image

        Returns:
            prob_maps_list [list torch.tensor float32, (4, H_orig, W_orig)]: List include prob_map of each batch, have shape (4, H_orig, W_orig)
        """     
        imgs = []
        origin_size = []
        
        for img_path in batch_image_paths:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"{img_path} not found!")
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H_orig, W_orig = img.shape[:2]
            origin_size.append((H_orig, W_orig))
            
            img_tensor = image_to_tensor(img).unsqueeze(0) # (1, 3, H_orig, W_orig)
            img_tensor = F.interpolate(img_tensor, size=self.new_size, mode='bilinear', align_corners=False)
            imgs.append(img_tensor.squeeze(0))
            del img
            
        batch_tensor = torch.stack(imgs, dim=0).to(self.device)  # (B, 3, H_new, W_new)
        
        with torch.no_grad():
            outputs = self.bisenet(batch_tensor)[0] # (B, C, H_new, W_new)
            
        probs = F.softmax(outputs, dim=1)
        class_map = {'Car': [13, 14, 15], 'Pedestrian': [11], 'Cyclist': [18]}
        conf_car = probs[:, class_map['Car'], :, :].sum(dim=1)
        conf_pedestrian = probs[:, class_map['Pedestrian'], :, :].sum(dim=1)
        conf_cyclist = probs[:, class_map['Cyclist'], :, :].sum(dim=1)
        conf_dontcare = probs.sum(dim=1) - (conf_car + conf_pedestrian + conf_cyclist)

        prob_map_batch = torch.stack([conf_car, conf_pedestrian, conf_cyclist, conf_dontcare], dim=1)  # (B, 4, H_new, W_new)
        
        prob_maps_list = []
        for i, (H_orig, W_orig) in enumerate(origin_size):
            prob_map_orig = F.interpolate(prob_map_batch[i].unsqueeze(0), size=(H_orig, W_orig), mode='bilinear', align_corners=False)
            prob_maps_list.append(prob_map_orig.squeeze(0)) # (4, H_orig, W_orig)
            
        return prob_maps_list
            
    
class PillarEncoder(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel):
        super(PillarEncoder, self).__init__()
        self.out_channel = out_channel
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)
        )
        
        self.semantic_proj = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        self.gate_fusion = nn.Sequential(
            nn.Linear(out_channel + 4, 64),
            nn.Sigmoid()
        )
        
    def forward(self, pillars, coors_batch, npoints_per_pillar, prob_maps_list, batched_calibs, batch_size):
    
        device = pillars.device
        
        mean_center = torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:, None, None] # (p1 + p2 + ... + pb, 1, 3)
        
        # 1. calculate offset to the points center (in each pillar)
        offset_pt_center = pillars[:, :, :3] - mean_center # (p1 + p2 + ... + pb, num_points, 3)

        # 2. calculate offset to the pillar center
        x_offset_pi_center = pillars[:, :, :1] - (coors_batch[:, None, 1:2] * self.vx + self.x_offset) # (p1 + p2 + ... + pb, num_points, 1)
        y_offset_pi_center = pillars[:, :, 1:2] - (coors_batch[:, None, 2:3] * self.vy + self.y_offset) # (p1 + p2 + ... + pb, num_points, 1)

        # 3. encoder
        features = torch.cat([pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center], dim=-1) # (p1 + p2 + ... + pb, num_points, 9)
        features[:, :, 0:1] = x_offset_pi_center # tmp
        features[:, :, 1:2] = y_offset_pi_center # tmp
        # In consitent with mmdet3d. 
        # The reason can be referenced to https://github.com/open-mmlab/mmdetection3d/issues/1150

        # 4. find mask for (0, 0, 0) and update the encoded features
        # a very beautiful implementation
        voxel_ids = torch.arange(0, pillars.size(1)).to(device) # (num_points, )
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :] # (num_points, p1 + p2 + ... + pb)
        mask = mask.permute(1, 0).contiguous()  # (p1 + p2 + ... + pb, num_points)
        features *= mask[:, :, None]

        # 5. embedding
        features = features.permute(0, 2, 1).contiguous() # (p1 + p2 + ... + pb, 9, num_points)
        features = F.relu(self.conv(features))  # (p1 + p2 + ... + pb, out_channels, num_points)
        pooling_features = torch.max(features, dim=-1)[0] # (p1 + p2 + ... + pb, out_channels)

        prob_features = []
        for i in range(batch_size):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_mean_center = mean_center[cur_coors_idx] # (pi, 1, 3)
            cur_mean_center = cur_mean_center.squeeze(1) # (pi, 3) x, y, z of the mean_center in current batch
            
            calib = batched_calibs[i]
            prob_map = prob_maps_list[i]
            H_orig, W_orig = prob_map.shape[1:]
            
            u, v = project_point_to_camera(point=cur_mean_center, calib=calib)
            u = torch.clamp(u, 0, W_orig - 1).long()
            v = torch.clamp(v, 0, H_orig - 1).long()
            
            prob = prob_map.permute(1, 2, 0)[v, u] # (pi, 4)
            
            prob_features.append(prob)
            
        prob_features = torch.cat(prob_features, dim=0)  # (p1 + p2 + ... + pb, 4)
        
        semantic_features = self.semantic_proj(prob_features) # (p1 + p2 + ... + pb, 64)
        
        concat_features = torch.cat([pooling_features, prob_features], dim=1) # (p1 + p2 + ... + pb, out_channels + 4)
        
        gate_weight = self.gate_fusion(concat_features) # (p1 + p2 + ... + pb, 64)
        
        gate_features = pooling_features * gate_weight + semantic_features * (1 - gate_weight) # (p1 + p2 + ... + pb, 64)
          
        # 6. pillar scatter
        batched_canvas = []
        for i in range(batch_size):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = gate_features[cur_coors_idx]

            canvas = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device)
            canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
            canvas = canvas.permute(2, 1, 0).contiguous()
            batched_canvas.append(canvas)
        batched_canvas = torch.stack(batched_canvas, dim=0) # (bs, 64, self.y_l, self.x_l)
        return batched_canvas


class Backbone(nn.Module):
    def __init__(self, in_channel, out_channels, layer_nums, layer_strides=[2, 2, 2]):
        super(Backbone, self).__init__()
        assert len(out_channels) == len(layer_nums)
        assert len(out_channels) == len(layer_strides)
        
        self.multi_blocks = nn.ModuleList()
        for i in range(len(layer_strides)):
            blocks = []
            blocks.append(nn.Conv2d(in_channel, out_channels[i], 3, stride=layer_strides[i], bias=False, padding=1))
            blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            blocks.append(nn.ReLU(inplace=True))

            for _ in range(layer_nums[i]):
                blocks.append(nn.Conv2d(out_channels[i], out_channels[i], 3, bias=False, padding=1))
                blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
                blocks.append(nn.ReLU(inplace=True))

            in_channel = out_channels[i]
            self.multi_blocks.append(nn.Sequential(*blocks))

        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        '''
        x: (b, c, y_l, x_l). Default: (6, 64, 496, 432)
        return: list[]. Default: [(6, 64, 248, 216), (6, 128, 124, 108), (6, 256, 62, 54)]
        '''
        outs = []
        for i in range(len(self.multi_blocks)):
            x = self.multi_blocks[i](x)
            outs.append(x)
        return outs


class Neck(nn.Module):
    def __init__(self, in_channels, upsample_strides, out_channels):
        super(Neck, self).__init__()
        assert len(in_channels) == len(upsample_strides)
        assert len(upsample_strides) == len(out_channels)

        self.decoder_blocks = nn.ModuleList()
        for i in range(len(in_channels)):
            decoder_block = []
            decoder_block.append(nn.ConvTranspose2d(in_channels[i], 
                                                    out_channels[i], 
                                                    upsample_strides[i], 
                                                    stride=upsample_strides[i],
                                                    bias=False))
            decoder_block.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            decoder_block.append(nn.ReLU(inplace=True))

            self.decoder_blocks.append(nn.Sequential(*decoder_block))
        
        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        '''
        x: [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        return: (bs, 384, 248, 216)
        '''
        outs = []
        for i in range(len(self.decoder_blocks)):
            xi = self.decoder_blocks[i](x[i]) # (bs, 128, 248, 216)
            outs.append(xi)
        out = torch.cat(outs, dim=1)
        return out


class Head(nn.Module):
    def __init__(self, in_channel, n_anchors, n_classes):
        super(Head, self).__init__()
        
        self.conv_cls = nn.Conv2d(in_channel, n_anchors*n_classes, 1)
        self.conv_reg = nn.Conv2d(in_channel, n_anchors*7, 1)
        self.conv_dir_cls = nn.Conv2d(in_channel, n_anchors*2, 1)

        # in consitent with mmdet3d
        conv_layer_id = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if conv_layer_id == 0:
                    prior_prob = 0.01
                    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
                    nn.init.constant_(m.bias, bias_init)
                else:
                    nn.init.constant_(m.bias, 0)
                conv_layer_id += 1

    def forward(self, x):
        '''
        x: (bs, 384, 248, 216)
        return: 
              bbox_cls_pred: (bs, n_anchors*3, 248, 216) 
              bbox_pred: (bs, n_anchors*7, 248, 216)
              bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        '''
        bbox_cls_pred = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        bbox_dir_cls_pred = self.conv_dir_cls(x)
        return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred


class PillarPainting(nn.Module):
    def __init__(
        self,
        nclasses=config['PILLARPAINTING']['num_classes'], 
        voxel_size=config['PILLARPAINTING']['voxel_size'],
        point_cloud_range=config['PILLARPAINTING']['pc_range'],
        max_num_points=config['PILLARPAINTING']['max_points'],
        max_voxels=config['PILLARPAINTING']['max_voxels'],
        bisenet=None
    ):
        super(PillarPainting, self).__init__()
        
        self.nclasses = nclasses
        
        self.pillar_layer = PillarLayer(
            voxel_size=voxel_size, 
            point_cloud_range=point_cloud_range, 
            max_num_points=max_num_points, 
            max_voxels=max_voxels
        )
        
        self.segment_layer = SegmentLayer(
            bisenet=bisenet,
            new_size=config['BISENET']['new_size']
        )
        
        self.pillar_encoder = PillarEncoder(
            voxel_size=voxel_size, 
            point_cloud_range=point_cloud_range, 
            in_channel=9, 
            out_channel=64
        )
        
        self.backbone = Backbone(
            in_channel=64, 
            out_channels=[64, 128, 256], 
            layer_nums=[3, 5, 5]
        )
        
        self.neck = Neck(
            in_channels=[64, 128, 256], 
            upsample_strides=[1, 2, 4], 
            out_channels=[128, 128, 128]
        )
        
        self.head = Head(
            in_channel=384, 
            n_anchors=2 * nclasses, 
            n_classes=nclasses
        )
        
        self.anchors_generator = Anchors(
            ranges=config['ANCHOR']['ranges'], 
            sizes=config['ANCHOR']['sizes'], 
            rotations=config['ANCHOR']['rotations']
        )
        
        # train
        self.assigners = [
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.6, 'neg_iou_thr': 0.45, 'min_iou_thr': 0.45},
        ]

        # val and test
        self.nms_pre = 300
        self.nms_thr = 0.1
        self.score_thr = 0.1
        self.max_num = 50

    def get_predicted_bboxes_single(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchors):
        '''
        bbox_cls_pred: (n_anchors*3, 248, 216) 
        bbox_pred: (n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (n_anchors*2, 248, 216)
        anchors: (y_l, x_l, 3, 2, 7)
        return: 
            bboxes: (k, 7)
            labels: (k, )
            scores: (k, ) 
        '''
        # 0. pre-process 
        bbox_cls_pred = bbox_cls_pred.permute(1, 2, 0).reshape(-1, self.nclasses)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 7)
        bbox_dir_cls_pred = bbox_dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        anchors = anchors.reshape(-1, 7)
        
        bbox_cls_pred = torch.sigmoid(bbox_cls_pred)
        bbox_dir_cls_pred = torch.max(bbox_dir_cls_pred, dim=1)[1]

        # 1. obtain self.nms_pre bboxes based on scores
        inds = bbox_cls_pred.max(1)[0].topk(self.nms_pre)[1]
        bbox_cls_pred = bbox_cls_pred[inds]
        bbox_pred = bbox_pred[inds]
        bbox_dir_cls_pred = bbox_dir_cls_pred[inds]
        anchors = anchors[inds]

        # 2. decode predicted offsets to bboxes
        bbox_pred = anchors2bboxes(anchors, bbox_pred)

        # 3. nms
        bbox_pred2d_xy = bbox_pred[:, [0, 1]]
        bbox_pred2d_lw = bbox_pred[:, [3, 4]]
        bbox_pred2d = torch.cat([bbox_pred2d_xy - bbox_pred2d_lw / 2,
                                 bbox_pred2d_xy + bbox_pred2d_lw / 2,
                                 bbox_pred[:, 6:]], dim=-1) # (n_anchors, 5)

        ret_bboxes, ret_labels, ret_scores = [], [], []
        for i in range(self.nclasses):
            # 3.1 filter bboxes with scores below self.score_thr
            cur_bbox_cls_pred = bbox_cls_pred[:, i]
            score_inds = cur_bbox_cls_pred > self.score_thr
            if score_inds.sum() == 0:
                continue

            cur_bbox_cls_pred = cur_bbox_cls_pred[score_inds]
            cur_bbox_pred2d = bbox_pred2d[score_inds]
            cur_bbox_pred = bbox_pred[score_inds]
            cur_bbox_dir_cls_pred = bbox_dir_cls_pred[score_inds]
            
            # 3.2 nms core
            keep_inds = nms_cuda(boxes=cur_bbox_pred2d, 
                                 scores=cur_bbox_cls_pred, 
                                 thresh=self.nms_thr, 
                                 pre_maxsize=None, 
                                 post_max_size=None)

            cur_bbox_cls_pred = cur_bbox_cls_pred[keep_inds]
            cur_bbox_pred = cur_bbox_pred[keep_inds]
            cur_bbox_dir_cls_pred = cur_bbox_dir_cls_pred[keep_inds]
            cur_bbox_pred[:, -1] = limit_period(cur_bbox_pred[:, -1].detach().cpu(), 1, np.pi).to(cur_bbox_pred) # [-pi, 0]
            cur_bbox_pred[:, -1] += (1 - cur_bbox_dir_cls_pred) * np.pi

            ret_bboxes.append(cur_bbox_pred)
            ret_labels.append(torch.zeros_like(cur_bbox_pred[:, 0], dtype=torch.long) + i)
            ret_scores.append(cur_bbox_cls_pred)

        # 4. filter some bboxes if bboxes number is above self.max_num
        if len(ret_bboxes) == 0:
            return [], [], []
        ret_bboxes = torch.cat(ret_bboxes, 0)
        ret_labels = torch.cat(ret_labels, 0)
        ret_scores = torch.cat(ret_scores, 0)
        if ret_bboxes.size(0) > self.max_num:
            final_inds = ret_scores.topk(self.max_num)[1]
            ret_bboxes = ret_bboxes[final_inds]
            ret_labels = ret_labels[final_inds]
            ret_scores = ret_scores[final_inds]
        result = {
            'lidar_bboxes': ret_bboxes.detach().cpu().numpy(),
            'labels': ret_labels.detach().cpu().numpy(),
            'scores': ret_scores.detach().cpu().numpy()
        }
        return result


    def get_predicted_bboxes(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors):
        '''
        bbox_cls_pred: (bs, n_anchors*3, 248, 216) 
        bbox_pred: (bs, n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        batched_anchors: (bs, y_l, x_l, 3, 2, 7)
        return: 
            bboxes: [(k1, 7), (k2, 7), ... ]
            labels: [(k1, ), (k2, ), ... ]
            scores: [(k1, ), (k2, ), ... ] 
        '''
        results = []
        bs = bbox_cls_pred.size(0)
        for i in range(bs):
            result = self.get_predicted_bboxes_single(bbox_cls_pred=bbox_cls_pred[i],
                                                      bbox_pred=bbox_pred[i], 
                                                      bbox_dir_cls_pred=bbox_dir_cls_pred[i], 
                                                      anchors=batched_anchors[i])
            results.append(result)
        return results

    def forward(self, batched_pts, mode='test', batched_gt_bboxes=None, batched_gt_labels=None, batched_image_paths = None, batched_calibs = None):
        batch_size = len(batched_pts)

        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)
        
        prob_maps_list = self.segment_layer(batched_image_paths)

        pillar_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar, prob_maps_list, batched_calibs, batch_size)

        xs = self.backbone(pillar_features)

        x = self.neck(xs)

        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.head(x)

        # anchors
        device = bbox_cls_pred.device
        feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=device)
        anchors = self.anchors_generator.get_multi_anchors(feature_map_size)
        batched_anchors = [anchors for _ in range(batch_size)]

        if mode == 'train':
            anchor_target_dict = anchor_target(
                batched_anchors=batched_anchors, 
                batched_gt_bboxes=batched_gt_bboxes, 
                batched_gt_labels=batched_gt_labels, 
                assigners=self.assigners,
                nclasses=self.nclasses
            )
            
            return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict
        
        elif mode == 'val':
            results = self.get_predicted_bboxes(
                bbox_cls_pred=bbox_cls_pred, 
                bbox_pred=bbox_pred, 
                bbox_dir_cls_pred=bbox_dir_cls_pred, 
                batched_anchors=batched_anchors
            )
            
            return results

        elif mode == 'test':
            results = self.get_predicted_bboxes(
                bbox_cls_pred=bbox_cls_pred, 
                bbox_pred=bbox_pred, 
                bbox_dir_cls_pred=bbox_dir_cls_pred, 
                batched_anchors=batched_anchors
            )
            
            return results
        else:
            raise ValueError   