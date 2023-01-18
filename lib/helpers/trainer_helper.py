import os
import tqdm

import torch
import numpy as np
import torch.nn as nn
import random

from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint
from lib.losses.centernet_loss import compute_centernet3d_loss
from progress.bar import Bar
from lib.helpers.decode_helper import extract_dets_from_outputs,extract_dets_for_train
from lib.helpers.decode_helper import decode_train_detections
from lib.ops import kitti_utils_torch as kitti_utils
from lib.helpers.roi_helper import featuremap2gridpoint
import iou3d_cuda
from lib.models.pointNet import PointNetDetector,CornerLoss,PointNet_xyz, CenterLoss
from lib.helpers.disp2prob import LaplaceDisp2Prob
from lib.losses.disparity_loss.stereo_focal_loss import StereoFocalLoss

def laplacian_aleatoric_uncertainty_loss(input, target, log_variance, reduction='mean'):
    '''
    References:
        MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships, CVPR'20
        Geometry and Uncertainty in Deep Learning for Computer Vision, University of Cambridge
    '''
    assert reduction in ['mean', 'sum']
    loss = 1.4142 * torch.exp(-log_variance) * torch.abs(input - target) + log_variance
    return loss.mean() if reduction == 'mean' else loss.sum()

class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_objs = train_loader.dataset.max_objs
        # self.rcnn_head = PointNetDetector(8).cuda()
        self.rcnn_loss = CornerLoss().cuda()
        self.deppro_loss = StereoFocalLoss(80)

        # loading pretrain/resume model
        if cfg.get('pretrain_model'):
            assert os.path.exists(cfg['pretrain_model'])
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=cfg['pretrain_model'],
                            map_location=self.device,
                            logger=self.logger)

        if cfg.get('resume_model', None):
            assert os.path.exists(cfg['resume_model'])
            self.epoch = load_checkpoint(model=self.model.to(self.device),
                                         optimizer=self.optimizer,
                                         filename=cfg['resume_model'],
                                         map_location=self.device,
                                         logger=self.logger)
            self.lr_scheduler.last_epoch = self.epoch - 1

        # self.gpu_ids = list(map(int, cfg['gpu_ids'].split(',')))
        # self.model = torch.nn.DataParallel(model, device_ids=self.gpu_ids).to(self.device)
        self.model = model.to(self.device)



    def train(self):
        start_epoch = self.epoch

        progress_bar = tqdm.tqdm(range(start_epoch, self.cfg['max_epoch']), dynamic_ncols=True, leave=True, desc='epochs')
        for epoch in range(start_epoch, self.cfg['max_epoch']):
            # reset random seed
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            # train one epoch
            self.train_one_epoch()
            self.epoch += 1

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()


            # save trained model
            if (self.epoch % self.cfg['save_frequency']) == 0:
                os.makedirs('checkpoints', exist_ok=True)
                ckpt_name = os.path.join('checkpoints', 'checkpoint_epoch_%d' % self.epoch)
                save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch), ckpt_name)

            progress_bar.update()

        return None

    def boxes_iou3d_gpu(self,boxes_a, boxes_b):
        """
        :param boxes_a: (N, 7) [x, y, z, h, w, l, ry]
        :param boxes_b: (M, 7) [x, y, z, h, w, l, ry]
        :return:
            ans_iou: (M, N)
        """
        boxes_a_bev = kitti_utils.boxes3d_to_bev_torch(boxes_a)
        boxes_b_bev = kitti_utils.boxes3d_to_bev_torch(boxes_b)

        boxes_a_bev = kitti_utils.boxes3d_to_bev_torch(boxes_a)
        boxes_b_bev = kitti_utils.boxes3d_to_bev_torch(boxes_b)

        # bev overlap
        overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
        iou3d_cuda.boxes_overlap_bev_gpu(boxes_a_bev.contiguous(), boxes_b_bev.contiguous(), overlaps_bev)

        # height overlap
        boxes_a_height_min = (boxes_a[:, 1] - boxes_a[:, 3]).view(-1, 1)
        boxes_a_height_max = boxes_a[:, 1].view(-1, 1)
        boxes_b_height_min = (boxes_b[:, 1] - boxes_b[:, 3]).view(1, -1)
        boxes_b_height_max = boxes_b[:, 1].view(1, -1)

        max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
        min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
        overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

        # 3d iou
        overlaps_3d = overlaps_bev * overlaps_h

        vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
        vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

        iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-7)
        return iou3d

    def subsample_rois(self, max_overlaps):
        # sample fg, easy_bg, hard_bg
        fg_rois_per_image = 16
        fg_thresh = 0.25

        fg_inds = ((max_overlaps >= fg_thresh)).nonzero().view(-1)
        easy_bg_inds = ((max_overlaps < 0.1)).nonzero().view(-1)
        # hard_bg_inds = ((max_overlaps < 0.4) &
        #         (max_overlaps >= 0.1)).nonzero().view(-1)

        fg_num_rois = fg_inds.numel()
        bg_num_rois = easy_bg_inds.numel()

        if fg_num_rois > 0:
            # sampling fg
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

        # elif fg_num_rois > 0 and fg_num_rois < fg_rois_per_image:
        #     # sampling fg
        #     rand_num = np.floor(np.random.rand(fg_rois_per_image) * fg_num_rois)
        #     rand_num = torch.from_numpy(rand_num).type_as(max_overlaps).long()
        #     fg_inds = fg_inds[rand_num]
        #     bg_inds = []

        elif fg_num_rois == 0:
            a=1
        else:
            print('maxoverlaps:(min=%f, max=%f)' % (max_overlaps.min().item(), max_overlaps.max().item()))
            print('ERROR: FG=%d, BG=%d' % (fg_num_rois, bg_num_rois))
            raise NotImplementedError
        return fg_inds

    def get_max_iou_with_same_class(self,rois, roi_labels, gt_boxes, gt_labels):
        """
        Args:
            rois: (N, 7)
            roi_labels: (N)
            gt_boxes: (N, )
            gt_labels:

        Returns:

        """
        """
        :param rois: (N, 7)
        :param roi_labels: (N)
        :param gt_boxes: (N, 8)
        :return:
        """
        max_overlaps = rois.new_zeros(rois.shape[0])
        gt_assignment = roi_labels.new_zeros(roi_labels.shape[0])

        for k in range(gt_labels.min().item(), gt_labels.max().item() + 1):
            roi_mask = (roi_labels == k)
            gt_mask = (gt_labels == k)
            if roi_mask.sum() > 0 and gt_mask.sum() > 0:
                cur_roi = rois[roi_mask]
                cur_gt = gt_boxes[gt_mask]
                original_gt_assignment = gt_mask.nonzero().view(-1)

                iou3d = self.boxes_iou3d_gpu(cur_roi, cur_gt)  # (M, N)
                cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1)
                max_overlaps[roi_mask] = cur_max_overlaps
                gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment]

        return max_overlaps, gt_assignment

    def generate_sup(self, cur_gt, num_objs):
        sup_rois = cur_gt.new_zeros(num_objs,cur_gt.shape[1]-1)
        sup_gt = cur_gt.new_zeros(num_objs,cur_gt.shape[1])
        temp_multicls = cur_gt[cur_gt[:,7]!=1]
        if temp_multicls.shape[0]>0:
            cur_gt = torch.cat((cur_gt,temp_multicls))
        for k in range(num_objs):
            kk = random.randint(0, len(cur_gt) - 1)
            ann = cur_gt[kk]
            if ann[7]==1.0:
                if np.random.random() < 0.7:
                    sup_rois[k][3] = ann[3] + random.uniform(-1.5, 1.5)
                    sup_rois[k][4] = ann[4] + random.uniform(-1.5, 1.5)
                    sup_rois[k][5] = ann[5] + random.uniform(-1.5, 1.5)
                    sup_rois[k][6] = ann[6] - random.uniform(-0.6, 0.6)

                    sup_rois[k][0] = ann[0] + random.uniform(-2, 2)
                    sup_rois[k][1] = ann[1] + random.uniform(-0.8, 0.8)
                    sup_rois[k][2] = ann[2] + random.uniform(-3, 3)
                else:
                    sup_rois[k][3] = ann[3] + random.uniform(-0.5, 0.5)
                    sup_rois[k][4] = ann[4] + random.uniform(-0.5, 0.5)
                    sup_rois[k][5] = ann[5] + random.uniform(-0.5, 0.5)

                    sup_rois[k][6] = ann[6] - random.uniform(-0.3, 0.3)

                    sup_rois[k][0] = ann[0] + random.uniform(-0.8, 0.8)
                    sup_rois[k][1] = ann[1] + random.uniform(-0.3, 0.3)
                    sup_rois[k][2] = ann[2] + random.uniform(-1, 1)
            else:
                if np.random.random() < 0.7:
                    sup_rois[k][3] = ann[3] + random.uniform(-1.5, 1.5)
                    sup_rois[k][4] = ann[4] + random.uniform(-1.5, 1.5)
                    sup_rois[k][5] = ann[5] + random.uniform(-1.5, 1.5)
                    sup_rois[k][6] = ann[6] - random.uniform(-0.6, 0.6)

                    sup_rois[k][0] = ann[0] + random.uniform(-2, 2)
                    sup_rois[k][1] = ann[1] + random.uniform(-0.8, 0.8)
                    sup_rois[k][2] = ann[2] + random.uniform(-3, 3)
                else:
                    sup_rois[k][3] = ann[3] + random.uniform(-0.5, 0.5)
                    sup_rois[k][4] = ann[4] + random.uniform(-0.5, 0.5)
                    sup_rois[k][5] = ann[5] + random.uniform(-0.5, 0.5)

                    sup_rois[k][6] = ann[6] - random.uniform(-0.3, 0.3)

                    sup_rois[k][0] = ann[0] + random.uniform(-0.8, 0.8)
                    sup_rois[k][1] = ann[1] + random.uniform(-0.3, 0.3)
                    sup_rois[k][2] = ann[2] + random.uniform(-1, 1)
            sup_gt[k] = ann
        return sup_rois, sup_gt


    def sample_rois_for_rcnn(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:

        """
        batch_size = batch_dict['rois'].shape[0]
        rois = batch_dict['rois']
        roi_scores = batch_dict['roi_scores']
        roi_labels = batch_dict['roi_labels']
        gt_boxes = batch_dict['gt_boxes']

        code_size = rois.shape[-1]
        batch_rois = rois.new_zeros(batch_size,16, code_size)
        batch_gt_of_rois = rois.new_zeros(batch_size,16, code_size + 1)
        batch_roi_ious = rois.new_zeros(batch_size,16)
        batch_roi_scores = rois.new_zeros(batch_size,16)
        batch_roi_labels = rois.new_zeros((batch_size,16), dtype=torch.long)

        for index in range(batch_size):
            cur_roi, cur_gt, cur_roi_labels, cur_roi_scores = \
                rois[index], gt_boxes[index], roi_labels[index], roi_scores[index]
            k = cur_gt.__len__() - 1
            while k > 0 and cur_gt[k][7] == -1:
                k -= 1
            cur_gt = cur_gt[:k + 1]
            cur_gt_num = len(cur_gt)
            cur_gt = cur_gt.new_zeros((1, cur_gt.shape[1])) if len(cur_gt) == 0 else cur_gt

            max_overlaps, gt_assignment = self.get_max_iou_with_same_class(
                rois=cur_roi, roi_labels=cur_roi_labels,
                gt_boxes=cur_gt[:, 0:7], gt_labels=cur_gt[:, 7].long()
            )

            sampled_inds = self.subsample_rois(max_overlaps=max_overlaps)
            num_supplement = 16-len(sampled_inds)
            if num_supplement>0 and cur_gt_num>0 and cur_gt[0,7]>-1:
                sup_rois, sup_gt = self.generate_sup(cur_gt,num_supplement)
            if sampled_inds.size()[0]>0:
                batch_rois[index,:len(sampled_inds),:] = cur_roi[sampled_inds]
                batch_roi_labels[index,:len(sampled_inds)] = cur_roi_labels[sampled_inds]
                batch_roi_ious[index,:len(sampled_inds)] = max_overlaps[sampled_inds]
                batch_roi_scores[index,:len(sampled_inds)] = cur_roi_scores[sampled_inds]
                batch_gt_of_rois[index,:len(sampled_inds),:] = cur_gt[gt_assignment[sampled_inds]]
            if num_supplement>0 and cur_gt_num>0 and cur_gt[0,7]>0:
                batch_rois[index, len(sampled_inds):, :] = sup_rois
                batch_roi_labels[index, len(sampled_inds):] = sup_gt[:,7]
                batch_gt_of_rois[index, len(sampled_inds):, :] = sup_gt
                batch_roi_ious[index, len(sampled_inds):] = 1.0

        return batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels
    def computer_weight(self):
        epoch = torch.tensor(self.epoch).float()
        weight = torch.tensor([1.0,1.0,0.0,0.0]).cuda()
        # weight[2] = 1 / (1 + torch.exp(-(epoch - 15) / 1))
        # weight[3] = weight[2]
        if epoch>10:
            weight[2]= 1/(1+torch.exp(-(epoch-10)/5))
            weight[3] = weight[2]
        return weight
    def train_one_epoch(self):
        self.model.train()
        # self.rcnn_head.train()
        bar = Bar()
        batch_dict = {}
        targets_dict={}
        outputs_train = {}
        # progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=(self.epoch+1 == self.cfg['max_epoch']), desc='iters')
        for batch_idx, (inputs, targets, info) in enumerate(self.train_loader):
            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                self.epoch, batch_idx, len(self.train_loader), phase='Train',
                total=bar.elapsed_td, eta=bar.eta_td)
            inputs = inputs.to(self.device)
            for key in targets.keys():
                targets[key] = targets[key].to(self.device)
            # train one batch
            self.optimizer.zero_grad()
            outputs,left_features,dense_depth = self.model(inputs,targets['sparse_dep'],targets)
            #-----------------------------------------------------------------------------------------------------
            val_pixels = (targets['dep_map'] > 1e-3).unsqueeze(1)
            depth_input, depth_log_variance = dense_depth[:, 0:1], dense_depth[:, 1:2]
            depth_loss = laplacian_aleatoric_uncertainty_loss(depth_input[val_pixels], \
                                        targets['dep_map'].unsqueeze(1)[val_pixels], depth_log_variance[val_pixels])
            #-----------------------------------------------------------------------------------------------------
            # val_pixels = (targets['dep_map'] > 1e-3).float().cuda()
            # depth_loss = targets['dep_map'] * val_pixels - dense_depth * val_pixels
            # depth_loss = (depth_loss ** 2).mean()


            for (k,v) in outputs.items():
                outputs_train[k]=v.clone()
            dets = extract_dets_for_train(outputs_train, self.max_objs,targets)
            batch_dict['roi_labels'] = dets[:,:,0].long()
            batch_dict['roi_scores'] = dets[:,:,1]
            info = {key: val.detach() for key, val in info.items()}
            cls_mean_size = torch.tensor(self.train_loader.dataset.cls_mean_size).cuda()
            batch_dict['rois'] = decode_train_detections(dets=dets,
                                     info=info,
                                     P2=info['P2'],
                                     cls_mean_size=cls_mean_size,
                                     threshold=self.cfg.get('threshold', 0.2))
            batch_dict['gt_boxes'] = targets['bbox3d']
            batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels = \
                self.sample_rois_for_rcnn(batch_dict=batch_dict)
            reg_valid_mask = (batch_roi_ious > 0.25).long()
            targets_dict = {'rois': batch_rois, 'gt_of_rois': batch_gt_of_rois, 'gt_iou_of_rois': batch_roi_ious,
                            'roi_scores': batch_roi_scores, 'roi_labels': batch_roi_labels,
                            'reg_valid_mask': reg_valid_mask}
            rois = targets_dict['rois']  # (B, N, 7 + C)
            gt_of_rois = targets_dict['gt_of_rois']  # (B, N, 7 + C + 1)
            targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()

            heading_label = (gt_of_rois[:, :, 6] - (rois[:, :, 6] % (2 * np.pi))) % (2 * np.pi)  # 0 ~ 2pi
            opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
            heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (
                    2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
            flag = heading_label > np.pi
            heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
            heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)
            rois[:, :, 6] = gt_of_rois[:, :, 6] - heading_label
            targets_dict['gt_of_rois'] = gt_of_rois
            batch_dict['rois'] = rois
            targets_dict['rois'] = rois
            batch_dict['roi_labels'] = targets_dict['roi_labels']

            batch_dict['trans_output'] = info['trans_output']
            batch_dict['calib_l'] = info['P2']
            batch_dict['calib_r'] = info['P3']
            batch_dict['left_image_feature'] = left_features
            batch_dict['right_image_feature'] = left_features
            sigma = torch.exp(-depth_log_variance)
            disparity_pro, pro_unnorm = LaplaceDisp2Prob(80, depth_input, variance=sigma,
                                             start_disp=0, dilation=1).getProb()
            # label = targets['dep_map'].cuda().unsqueeze(1)
            # depth_loss = self.deppro_loss(pro_unnorm, label, variance=0.5)
            batch_dict['dep_map']= disparity_pro

            point_data = featuremap2gridpoint(batch_dict, 'train', targets_dict)
            loss_batch = point_data['input_batch']
            outputs_rcnn = self.model.rcnn_head(point_data)
            rcnn_loss, loss_stats, next_est = self.rcnn_loss(outputs_rcnn, loss_batch, self.epoch + 1)

            centernet_loss, obj_dep_loss, stats_batch = compute_centernet3d_loss(outputs, targets)
            stats_batch['rcnn_loss'] = rcnn_loss.item()
            stats_batch['depth_loss'] = depth_loss.item()
            weight = self.computer_weight()
            # total_loss = centernet_loss + depth_loss + obj_dep_loss + rcnn_loss
            total_loss = centernet_loss + depth_loss + weight[3]*obj_dep_loss+rcnn_loss
            # total_loss = centernet_loss
            total_loss.backward()
            self.optimizer.step()
            if batch_idx%100==0:
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format('total_loss', total_loss)
                for l in stats_batch:
                    Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, stats_batch[l])
                print('{}'.format(Bar.suffix))
            # progress_bar.update()
        # progress_bar.close()




