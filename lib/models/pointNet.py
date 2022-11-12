from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
# from visualDet3D.utils.model_nms_utils import boxes_iou3d_gpu
import numpy as np
import iou3d_cuda


def boxes3d_to_bev_torch(boxes3d):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    :return:
        boxes_bev: (N, 5) [x1, y1, x2, y2, ry]
    """
    boxes_bev = boxes3d.new(torch.Size((boxes3d.shape[0], 5)))

    cu, cv = boxes3d[:, 0], boxes3d[:, 2]
    half_l, half_w = boxes3d[:, 5] / 2, boxes3d[:, 4] / 2
    boxes_bev[:, 0], boxes_bev[:, 1] = cu - half_l, cv - half_w
    boxes_bev[:, 2], boxes_bev[:, 3] = cu + half_l, cv + half_w
    boxes_bev[:, 4] = boxes3d[:, 6]
    return boxes_bev

class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """

    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class CenterLoss(torch.nn.Module):
    def __init__(self, opt=None):
        super(CenterLoss, self).__init__()

    def forward(self, outputs, batch, epoch=None):

        dim_real = batch['dim_real'][:, :]
        pos_real = batch['pos_real'][:, :]
        ori_real = batch['ori_real'][:].unsqueeze(1)

        dim_est = batch['dim_est'][:, :]
        pos_est = batch['pos_est'][:, :]+outputs[:, :]
        ori_est = batch['ori_est_scalar'][:].unsqueeze(1)
        reg_valid_mask = batch['reg_mask']
        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()


        iou3d_input_gt = torch.cat([pos_real,dim_real,ori_real],dim=1)
        ES_EST = torch.cat([pos_est, dim_est, ori_est], dim=1)

        iou3d_input_est = ES_EST

        next_est = iou3d_input_est

        loss_pos = F.l1_loss(iou3d_input_est[:,:3][fg_mask],iou3d_input_gt[:,:3][fg_mask],reduction='sum')/max(fg_sum,1)
        # loss_dim = F.l1_loss(iou3d_input_est[:, 3:6][fg_mask], iou3d_input_gt[:, 3:6][fg_mask],reduction='sum')/max(fg_sum,1)
        # loss_ori = F.l1_loss(iou3d_input_est[:, 6][fg_mask], iou3d_input_gt[:, 6][fg_mask],reduction='sum')/max(fg_sum,1)
        loss = loss_pos
        loss_stats = {'loss': loss,'loss_reg':loss_pos}

        return loss, loss_stats,next_est

class CornerLoss(torch.nn.Module):
    def __init__(self, opt=None):
        super(CornerLoss, self).__init__()

        self.opt = opt
        # self.iou_loss=torch.nn.BCEWithLogitsLoss()
        self.iou_loss =BCEFocalLoss()
        self.coners_const = torch.Tensor(
            [[0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5,0],
             [0, 0, 0, 0, -1, -1, -1, -1,-0.5],
             [0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5,0]]
        )

        self.rampup_coor = self.exp_rampup(25)

    def exp_rampup(self,rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""

        def warpper(epoch):
            if epoch < rampup_length:
                epoch = np.clip(epoch, 0.0, rampup_length)
                phase = 1.0 - epoch / rampup_length
                return float(np.exp(-5.0 * phase * phase))
            else:
                return 1.0

        return warpper


    def boxes_iou3d_gpu(self,boxes_a, boxes_b):
        """
        :param boxes_a: (N, 7) [x, y, z, w, h, l, ry]
        :param boxes_b: (M, 7) [x, y, z, h, w, l, ry]
        :return:
            ans_iou: (M, N)
        """
        boxes_a_bev = boxes3d_to_bev_torch(boxes_a)
        boxes_b_bev = boxes3d_to_bev_torch(boxes_b)

        # bev overlap
        overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
        iou3d_cuda.boxes_overlap_bev_gpu(boxes_a_bev.contiguous(), boxes_b_bev.contiguous(), overlaps_bev)

        # height overlap
        boxes_a_height_min = (boxes_a[:, 1] - boxes_a[:, 3]).view(-1, 1)
        boxes_a_height_max = boxes_a[:, 1].view(-1, 1)
        boxes_b_height_min = (boxes_b[:, 1] - boxes_b[:, 3]).view(1, -1)
        boxes_b_height_max = boxes_b[:, 1].view(1, -1)

        # boxes_a_height_max = (boxes_a[:, 1] + boxes_a[:, 4] / 2).view(-1, 1)
        # boxes_a_height_min = (boxes_a[:, 1] - boxes_a[:, 4] / 2).view(-1, 1)
        # boxes_b_height_max = (boxes_b[:, 1] + boxes_b[:, 4] / 2).view(1, -1)
        # boxes_b_height_min = (boxes_b[:, 1] - boxes_b[:, 4] / 2).view(1, -1)




        max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
        min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
        overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

        # 3d iou
        overlaps_3d = overlaps_bev * overlaps_h

        vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
        vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

        iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-7)

        return iou3d

    def param2corner(self,pos,h,w,l,ry):
        #pos=pos.transpose(1,0)#3,K
        # h = h.transpose(1, 0)#1,K
        # w = w.transpose(1, 0)#1,K
        # l = l.transpose(1, 0)#1,K
        pos = pos.unsqueeze(2).expand(l.size(0),3,9)
        dim=torch.cat([l,h,w],dim=1).unsqueeze(2)
        dim=dim.expand(l.size(0),3,9)#K,3,9
        corner = self.coners_const.cuda().unsqueeze(0).expand(l.size(0),3,9)  # K,3,9
        corner=dim*corner

        R=pos.new_zeros(pos.size(0),3,3)
        R[:,0,0]=torch.cos(ry)
        R[:, 0, 2] = torch.sin(ry)
        R[:, 1, 1] = 1
        R[:, 2, 0] = -torch.sin(ry)
        R[:, 2, 1] = torch.cos(ry)#K,3,3
        corner=R.bmm(corner)+pos#K,3,9

        return corner


    def forward(self, outputs, batch, epoch=None):


        dim_real = batch['dim_real'][:, :]
        pos_real = batch['pos_real'][:, :]
        ori_real = batch['ori_real'][:].unsqueeze(1)

        dim_est = batch['dim_est'][:, :]
        pos_est = batch['pos_est'][:, :]
        ori_est = batch['ori_est_scalar'][:].unsqueeze(1)
        reg_valid_mask = batch['reg_mask']
        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()


        iou3d_input_gt = torch.cat([pos_real,dim_real,ori_real],dim=1)
        ES_EST = torch.cat([pos_est, dim_est, ori_est], dim=1)

        iou3d_input_est = ES_EST +outputs[:,:7]

        next_est = iou3d_input_est
        box_score = self.boxes_iou3d_gpu(iou3d_input_est.detach(), iou3d_input_gt.detach())
        box_score = torch.diag(box_score)

        # test_input1 = torch.cat([iou3d_input_est[:, 0, None], iou3d_input_est[:, 1, None], iou3d_input_est[:, 2, None], \
        #                         iou3d_input_est[:, 4, None], iou3d_input_est[:, 3, None], iou3d_input_est[:, 5, None], \
        #                         iou3d_input_est[:, 6, None]], dim=1)
        # test_gt1 = torch.cat([iou3d_input_gt[:, 0, None], iou3d_input_gt[:, 1, None], iou3d_input_gt[:, 2, None], \
        #                      iou3d_input_gt[:, 4, None], iou3d_input_gt[:, 3, None], iou3d_input_gt[:, 5, None], \
        #                      iou3d_input_gt[:, 6, None]], dim=1)
        # test_box_score1 = self.boxes_iou3d_gpu(test_input1.detach(), test_gt1.detach())
        # box_score = torch.diag(test_box_score1)
        #
        # test_box_score2 = self.boxes_iou3d_gpu(iou3d_input_est.detach(), iou3d_input_gt.detach())
        # test_box_score2 = torch.diag(test_box_score2)

        pos_pre = iou3d_input_est[:,:3]
        h_pre = iou3d_input_est[:,3:4]
        w_pre = iou3d_input_est[:, 4:5]
        l_pre = iou3d_input_est[:, 5:6]
        ry_pre = iou3d_input_est[:,6]
        corner_pre = self.param2corner(pos_pre, h_pre, w_pre, l_pre, ry_pre)

        pos_g = iou3d_input_gt[:, :3]
        h_g = iou3d_input_gt[:, 3:4]
        w_g = iou3d_input_gt[:, 4:5]
        l_g = iou3d_input_gt[:, 5:6]
        ry_g = iou3d_input_gt[:, 6]
        corner_g = self.param2corner(pos_g, h_g, w_g, l_g, ry_g)

        l2 = corner_g - corner_pre#K,3,9
        l2 = torch.norm(l2, p=2, dim=1)#K,9
        l2 = torch.log(l2 + 1)
        loss_reg = l2[fg_mask].sum()/(max(fg_sum*9,1))

        box_score = box_score.detach()[fg_mask]
        box_score = 2*box_score-0.5
        box_score = torch.clamp(box_score,0,1)
        loss_cls = self.iou_loss(outputs[:,7][fg_mask],box_score)

        if fg_mask.sum()>0:
            loss = self.rampup_coor(epoch)*loss_cls + loss_reg
        else:
            loss = loss_reg
        # loss = self.rampup_coor(epoch)*loss_cls+loss_reg
        # loss = loss_reg

        loss_pos = F.l1_loss(iou3d_input_est[:,:3][fg_mask],iou3d_input_gt[:,:3][fg_mask],reduction='sum')/max(fg_sum,1)
        loss_dim = F.l1_loss(iou3d_input_est[:, 3:6][fg_mask], iou3d_input_gt[:, 3:6][fg_mask],reduction='sum')/max(fg_sum,1)
        loss_ori = F.l1_loss(iou3d_input_est[:, 6][fg_mask], iou3d_input_gt[:, 6][fg_mask],reduction='sum')/max(fg_sum,1)
        # loss = loss_pos + loss_dim+loss_ori
        loss_stats = {'loss': loss,'loss_cls':loss_cls,'loss_reg':loss_reg,'box_score':box_score,'loss_pos':loss_pos,'loss_dim':loss_dim,'loss_ori':loss_ori}

        return loss, loss_stats,next_est

class CNN3D(nn.Module):
    def __init__(self, k=67):
        super(CNN3D, self).__init__()
        self.conv1 = torch.nn.Conv3d(k,64,kernel_size=3,stride=1,padding=1)
        self.conv2 = torch.nn.Conv3d(64, 128, kernel_size=3, stride=1,padding=1)
        self.conv3 = torch.nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x=x.view(x.size(0),x.size(1),10,10,10)
        x=self.conv1(x)
        x=self.conv2(x)
        x = self.conv3(x)
        x=x.view(x.size(0),x.size(1),-1)
        return x

class PointNetfeat_strAM(nn.Module):
    def __init__(self, input_c=67,opt=None):
        super(PointNetfeat_strAM, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_c, 256, 1)
        self.conv2 = torch.nn.Conv1d(256, 512, 1)
        self.conv3 = torch.nn.Conv1d(512, 1024, 1)
        self.conv4 = torch.nn.Conv1d(1024, 1024, 1)
        self.isp=nn.Sequential(
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000)
            )
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.strAM_2D=torch.nn.Conv2d(1024,1024,3,1,1)
        self.opt=opt

    def forward(self, x,xyz):
        x = F.relu(self.bn1((self.conv1(x))))
        x = F.relu((self.bn2(self.conv2(x))))
        x = self.bn3(self.conv3(x))
        isp_cube=x.view(x.size(0),x.size(1),10,10,10)
        isp=torch.mean(isp_cube,dim=3)
        isp=torch.sigmoid(self.strAM_2D(isp)).unsqueeze(3)
        isp=isp.expand_as(isp_cube)
        isp=isp*isp_cube
        isp=isp.view(x.size(0),x.size(1),10*10*10)
        x = F.relu(self.bn4(self.conv4(isp)))+x
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x

class PointNetDetector(nn.Module):
    def __init__(self, feature_transform=False,opt=None):
        super(PointNetDetector, self).__init__()
        self.feature_transform = feature_transform
        self.opt = opt
        self.feat_all = PointNetfeat_strAM(input_c=260, opt=opt)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.dim = nn.Linear(256, 3)
        self.pos = nn.Linear(256, 3)
        self.ori = nn.Linear(256, 1)
        self.conf = nn.Linear(256, 1)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    def forward(self, input_data):
        x = input_data['input_feat_consis']
        xyz = input_data['input_feat_xyz_abs']
        xa = self.feat_all(x,xyz)
        x = F.relu(self.bn1(self.fc1(xa)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        pos = self.pos(x)
        dim = self.dim(x)
        ori = self.ori(x)
        conf = self.conf(x)
        x = torch.cat([pos,dim,ori,conf],dim=1)
        return x

class PointNet_xyz(nn.Module):
    def __init__(self, feature_transform=False,opt=None):
        super(PointNet_xyz, self).__init__()
        self.feature_transform = feature_transform
        self.opt = opt
        self.feat_all = PointNetfeat_strAM(input_c=196, opt=opt)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        # self.dim = nn.Linear(256, 3)
        self.pos = nn.Linear(256, 3)
        # self.ori = nn.Linear(256, 1)
        # self.conf = nn.Linear(256, 1)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    def forward(self, input_data):
        x = input_data['input_feat_consis']
        xyz = input_data['input_feat_xyz_abs']
        xa = self.feat_all(x,xyz)
        x = F.relu(self.bn1(self.fc1(xa)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        pos = self.pos(x)
        # dim = self.dim(x)
        # ori = self.ori(x)
        # conf = self.conf(x)
        # x = torch.cat([pos,dim,ori],dim=1)
        x = pos
        return x
