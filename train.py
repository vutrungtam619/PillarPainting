import argparse
import os
import torch
from tqdm import tqdm
from config.config import config

from utils import setup_seed, Loss
from dataset import Kitti, get_dataloader
from model import BiSeNetV2, PillarPainting
from torch.utils.tensorboard import SummaryWriter

def save_summary(writer, loss_dict, global_step, tag, lr=None, momentum=None):
    for k, v in loss_dict.items():
        writer.add_scalar(f'{tag}/{k}', v, global_step)
    if lr is not None:
        writer.add_scalar('lr', lr, global_step)
    if momentum is not None:
        writer.add_scalar('momentum', momentum, global_step)
        
def main(args):
    setup_seed()
    
    print("Loading dataset!..............................................")
    train_dataset = Kitti(data_root=args.data_root, split='train')
    val_dataset = Kitti(data_root=args.data_root, split='val')
    
    train_dataloader = get_dataloader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)    
    print("Loading dataset succesfully!..................................")   
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    
    print("Loading Bisenet and PillarPainting model!.....................")
    bisenet = BiSeNetV2()
    bisenet.load_state_dict(torch.load(args.bisenet_ckpt))
    bisenet.to(device)
    pillarpainting = PillarPainting(bisenet=bisenet).to(device)
    print("Finished loading Bisenet and PillarPainting model!............")
    
    loss_func = Loss()
    
    max_iters = len(train_dataloader) * args.epoch    
    optimizer = torch.optim.AdamW(pillarpainting.parameters(), lr=args.init_lr, betas=(0.95, 0.99), weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.init_lr*4, total_steps=max_iters, pct_start=0.4, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=10)    
    
    start_epoch = 0
    
    # Resume train if there is checkpoint exist
    checkpoint_exist = [f for f in os.listdir(args.checkpoint_dir) if f.endswith(".pth")]
    if checkpoint_exist:
        checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_exist[0])
        checkpoint_dict = torch.load(checkpoint_path)
        
        start_epoch = int(checkpoint_dict['epoch']) 
        print(f"Founding a checkpoint!... Continue training from epoch {start_epoch}.......")
        
        pillarpainting.load_state_dict(checkpoint_dict['checkpoint'])
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        scheduler.load_state_dict(checkpoint_dict['scheduler'])
        print("......... Loading checkpoint, optimizer, schedular sucessfully!")
    else:
        print("No checkpoint found! Starting training from scratch!")     
    
    writer = SummaryWriter(args.log_dir)
    
    for epoch in range(start_epoch, args.epoch):
        print('='*20, f'Epoch {epoch}', '='*20)
        pillarpainting.train()
        train_step, val_step = 0, 0   
        
        for i, data_dict in enumerate(tqdm(train_dataloader)):
            # Move tensors to device
            for key in data_dict:
                for j, item in enumerate(data_dict[key]):
                    if torch.is_tensor(item):
                        data_dict[key][j] = data_dict[key][j].to(device)

            optimizer.zero_grad()
            
            batched_pts = data_dict['batched_pts']
            batched_gt_bboxes = data_dict['batched_gt_bboxes']
            batched_labels = data_dict['batched_labels']
            batched_image_paths = data_dict['batched_image_paths']
            batched_calibs = data_dict['batched_calib_info']

            # Forward (FP32 normal)
            bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = pillarpainting(
                batched_pts=batched_pts, mode='train',
                batched_gt_bboxes=batched_gt_bboxes, 
                batched_gt_labels=batched_labels,
                batched_image_paths=batched_image_paths,
                batched_calibs=batched_calibs
            )     
            
            # Reshape and filter positive samples
            bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.num_classes) # (B, H, W, C)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
            bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

            batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
            batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
            batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
            # batched_bbox_reg_weights = anchor_target_dict['batched_bbox_reg_weights'].reshape(-1)
            batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)
            # batched_dir_labels_weights = anchor_target_dict['batched_dir_labels_weights'].reshape(-1)
            
            pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.num_classes)
            bbox_pred = bbox_pred[pos_idx]
            batched_bbox_reg = batched_bbox_reg[pos_idx]
            # sin(a - b) = sin(a)*cos(b) - cos(a)*sin(b)
            bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1].clone()) * torch.cos(batched_bbox_reg[:, -1].clone())
            batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1].clone()) * torch.sin(batched_bbox_reg[:, -1].clone())
            bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
            batched_dir_labels = batched_dir_labels[pos_idx]

            num_cls_pos = (batched_bbox_labels < args.num_classes).sum()
            bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
            batched_bbox_labels[batched_bbox_labels < 0] = args.num_classes
            batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

            loss_dict = loss_func(
                bbox_cls_pred=bbox_cls_pred,
                bbox_pred=bbox_pred,
                bbox_dir_cls_pred=bbox_dir_cls_pred,
                batched_labels=batched_bbox_labels, 
                num_cls_pos=num_cls_pos, 
                batched_bbox_reg=batched_bbox_reg, 
                batched_dir_labels=batched_dir_labels
            )
            
            loss = loss_dict['total_loss']
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(pointpillars.parameters(), max_norm=35)
            optimizer.step()
            scheduler.step()

            global_step = epoch * len(train_dataloader) + train_step + 1

            if global_step % args.log_freq == 0:
                save_summary(writer, loss_dict, global_step, 'train', lr=optimizer.param_groups[0]['lr'], momentum=optimizer.param_groups[0]['betas'][0])
            train_step += 1
        
        if ((epoch + 1) % args.ckpt_freq == 0):
            save_dict = {
                "checkpoint": pillarpainting.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),  # Save scheduler state
                "epoch": epoch + 1  # Save current epoch number
            }
            new_ckpt_file = os.path.join(args.checkpoint_dir, f'epoch_{epoch+1}.pth')
            torch.save(save_dict, new_ckpt_file)
                    
        if (epoch % 2 == 0):
            continue
        
        pillarpainting.eval()
        with torch.no_grad():
            for i, data_dict in enumerate(tqdm(val_dataloader)):
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].to(device)
                
                batched_pts = data_dict['batched_pts']
                batched_gt_bboxes = data_dict['batched_gt_bboxes']
                batched_labels = data_dict['batched_labels']
                batched_image_paths = data_dict['batched_image_paths']
                batched_calibs = data_dict['batched_calib_info']

                # Forward (FP32 normal)
                bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = pillarpainting(
                    batched_pts=batched_pts, mode='train',
                    batched_gt_bboxes=batched_gt_bboxes, 
                    batched_gt_labels=batched_labels,
                    batched_image_paths=batched_image_paths,
                    batched_calibs=batched_calibs
                )     
                
                bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.num_classes)
                bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
                bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

                batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
                batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
                batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
                # batched_bbox_reg_weights = anchor_target_dict['batched_bbox_reg_weights'].reshape(-1)
                batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)
                # batched_dir_labels_weights = anchor_target_dict['batched_dir_labels_weights'].reshape(-1)
                
                pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.num_classes)
                bbox_pred = bbox_pred[pos_idx]
                batched_bbox_reg = batched_bbox_reg[pos_idx]
                # sin(a - b) = sin(a)*cos(b) - cos(a)*sin(b)
                bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1]) * torch.cos(batched_bbox_reg[:, -1])
                batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1]) * torch.sin(batched_bbox_reg[:, -1])
                bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
                batched_dir_labels = batched_dir_labels[pos_idx]

                num_cls_pos = (batched_bbox_labels < args.num_classes).sum()
                bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
                batched_bbox_labels[batched_bbox_labels < 0] = args.num_classes
                batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

                loss_dict = loss_func(
                    bbox_cls_pred=bbox_cls_pred,
                    bbox_pred=bbox_pred,
                    bbox_dir_cls_pred=bbox_dir_cls_pred,
                    batched_labels=batched_bbox_labels, 
                    num_cls_pos=num_cls_pos, 
                    batched_bbox_reg=batched_bbox_reg, 
                    batched_dir_labels=batched_dir_labels
                )

                global_step = epoch * len(val_dataloader) + val_step + 1
                if global_step % args.log_freq == 0:
                    save_summary(writer, loss_dict, global_step, 'val')
                val_step += 1        
                     
                   
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', default=config['DATASET']['data_root'])
    parser.add_argument('--checkpoint_dir', default=config['TRAINING']['checkpoint_dir'])
    parser.add_argument('--batch_size', default=config['TRAINING']['batch_size'])
    parser.add_argument('--num_workers', default=config['TRAINING']['num_workers'])
    parser.add_argument('--num_classes', default=config['PILLARPAINTING']['num_classes'])
    parser.add_argument('--init_lr', default=config['TRAINING']['init_lr'])
    parser.add_argument('--epoch', default=config['TRAINING']['epoch'])
    parser.add_argument('--ckpt_freq', default=config['TRAINING']['ckpt_freq'])
    parser.add_argument('--log_freq', default=config['TRAINING']['log_freq'])
    parser.add_argument('--log_dir', default=config['TRAINING']['log_dir'])
    parser.add_argument('--bisenet_ckpt', default=config['BISENET']['bisenet_ckpt'])
    args = parser.parse_args()
    
    main(args)