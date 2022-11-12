import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
def generate_gridpoint(dim, pos, ori, calib_l, calib_r, trans_output_l, trans_output_r, opt=None):  # dim B,K,3
    '''
       generate grid point coordinates, the image featuremap coordinates corresponding the grid point.
       return:
            image_xy_l: left image featuremap coordinates corresponding the grid point.
            image_xy_r: right image featuremap coordinates corresponding the grid point.
            xyz_norm: the grid point coordinates in the object coordinate system
            xyz: the grid point coordinates in the camera coordinate system
    '''

    h = dim[0]
    w = dim[1]
    l = dim[2]
    x_axi = -torch.linspace(-l / 2., l / 2., 10).cuda()
    y_axi = torch.linspace(0, -h, 10).cuda()
    z_axi = -torch.linspace(-w / 2., w / 2., 10).cuda()
    xx, yy, zz = torch.meshgrid(x_axi, y_axi, z_axi)
    xyz = torch.stack([xx, yy, zz], 0).view((3, -1))  # 3,resl***2
    R = ori
    xyz = R.mm(xyz)
    xyz_norm = xyz.clone()
    xyz[0, :] += pos[0]
    xyz[1, :] += pos[1]
    xyz[2, :] += pos[2]
    ones = torch.ones((1, xyz.size(1))).cuda()
    xyz_hom = torch.cat((xyz, ones), dim=0)
    image_xy_hom_l = calib_l.mm(xyz_hom)
    image_xy_hom_l = image_xy_hom_l / image_xy_hom_l[2, :]

    image_xy_hom_r = calib_r.mm(xyz_hom)
    image_xy_hom_r = image_xy_hom_r / image_xy_hom_r[2, :]
    image_xy_l = []
    image_xy_r = []
    for py in range(3):
        image_xy_l.append(trans_output_l[py].mm(image_xy_hom_l))
        image_xy_r.append(trans_output_r[py].mm(image_xy_hom_r))
    # disp = calib_l[0,0]*0.54/(xyz[2, :]+1e-9)[None,:]
    # xyd= torch.cat((image_xy_l[0],disp), dim=0)
    image_xy_l = torch.stack(image_xy_l,dim=0)
    image_xy_r = torch.stack(image_xy_r, dim=0)
    return image_xy_l, image_xy_r, xyz_norm, xyz, xyz[2:3,:]


def featuremap2gridpoint(batch, phase='train', targets=None,opt=None):
    '''
       image featuremap to gridpoint
    '''
    outputs_l, outputs_r = batch['left_image_feature'], batch['right_image_feature']
    batch_for_point = {}
    # batch_for_point['dim'] = []
    # batch_for_point['pos'] = []
    # batch_for_point['ori'] = []
    # batch_for_point['dim_real'] = []
    # batch_for_point['pos_real'] = []
    # batch_for_point['ori_real'] = []
    # batch_for_point['dim_est'] = []
    # batch_for_point['pos_est'] = []
    # batch_for_point['ori_est_scalar'] = []
    # batch_for_point['reg_mask'] = []

    B = outputs_l[0].size(0)
    ## *_est represent monocular 3D detector results.
    dim = batch['rois'][:, :, 3:6].cuda()
    pos = batch['rois'][:, :, 0:3].cuda()
    ori = batch['rois'][:, :, 6].cuda()

    cosa = torch.cos(ori)
    sina = torch.sin(ori)
    zeros = ori.new_zeros((ori.shape[0], ori.shape[1]))
    ones = ori.new_ones((ori.shape[0], ori.shape[1]))
    rot_matrix = torch.stack((
        cosa, zeros, sina,
        zeros, ones, zeros,
        -sina, zeros, cosa

    ), dim=2).view(ori.shape[0], ori.shape[1], 3, 3).float()

    calib_l = batch['calib_l'].cuda()
    calib_r = batch['calib_r'].cuda()
    ## trans_output_* represent the transformation from 3D grid point to image featuremap.
    trans_output_l = batch['trans_output'].cuda()
    trans_output_r = batch['trans_output'].cuda()

    pointNet_input_list_r = []
    pointNet_input_list_l = []
    pointNet_input_list_xyz_abs = []
    pointNet_input_consis = []
    disp_map = batch['dep_map'].unsqueeze(1)
    # reg_mask = batch['reg_mask']
    obj_num = []
    for b in range(B):
        index_box_l = []
        index_box_r = []
        volume_xyz_list = []
        volume_xyz_abs_list = []
        volume_xyd_list = []
        cls_fea_list = []
        # mask = torch.nonzero(reg_mask[b])
        # K = mask.size(0)
        # obj_num.append(K)
        K = batch['rois'].shape[1]

        for k in range(K):  # range(self.opt.max_objs):
            index_l, index_r, xyz, xyz_abs,disp = generate_gridpoint(dim[b, k], pos[b, k],
                                                                rot_matrix[b, k], calib_l[b],
                                                                calib_r[b], trans_output_l[b],
                                                                trans_output_r[b])
            index_box_l.append(index_l)
            index_box_r.append(index_r)
            volume_xyz_list.append(xyz)
            volume_xyz_abs_list.append(xyz_abs)
            volume_xyd_list.append(disp)

        index_box_l = torch.stack(index_box_l, 0).transpose(3, 2).unsqueeze(0)  # 1,K,4,resl***2,2
        index_box_r = torch.stack(index_box_r, 0).transpose(3, 2).unsqueeze(0)

        volume_xyz_list = torch.stack(volume_xyz_list, 0)  # m(<=K),3,resl***2
        volume_xyz_abs_list = torch.stack(volume_xyz_abs_list, 0)
        volume_xyd_list = torch.stack(volume_xyd_list, 0).transpose(1, 2).unsqueeze(0)
        # cls_fea_list = torch.stack(cls_fea_list,0)                      #16*1*18*80
        volume_from_heatmap_l = []
        volume_from_heatmap_r = []
        volume_from_dispmap = []

        for py in range(3):
            grid_l = index_box_l[:, :, py, :, :].cuda()  # 1, K,resl***2,2
            grid_r = index_box_r[:, :, py, :, :].cuda()  # 1, K,resl***2,2
            featuremap_l = outputs_l[py].cuda()
            # featuremap_r = outputs_r[py].cuda()
            lx = 2 * (grid_l[:, :, :, 0] / featuremap_l.size(3) - 0.5)
            ly = 2 * (grid_l[:, :, :, 1] / featuremap_l.size(2) - 0.5)
            # rx = 2 * (grid_r[:, :, :, 0] / featuremap_r.size(3) - 0.5)
            # ry = 2 * (grid_r[:, :, :, 1] / featuremap_r.size(2) - 0.5)

            grid_l = torch.stack((lx, ly), dim=3)
            # grid_r = torch.stack((rx, ry), dim=3)

            if py==1:
                ld = 2 * (volume_xyd_list[:, :, :, 0] / 80 - 0.5)
                grid_d  = torch.stack((lx, ly, ld), dim=3)
                volume_from_dispmap = \
                    torch.nn.functional.grid_sample(disp_map[b:b + 1], grid_d.unsqueeze(1)).squeeze(1)
            #     volume_cls_mask = torch.nn.functional.grid_sample(cls_fea_list, \
            #                                                       grid_l.transpose(0, 1)).transpose(0,2)
            volume_from_heatmap_l.append(
                torch.nn.functional.grid_sample(featuremap_l[b:b + 1], grid_l))  # 1,64,16K,resl***2
            # volume_from_heatmap_r.append(
            #     torch.nn.functional.grid_sample(featuremap_r[b:b + 1], grid_r))  # 1,64,16K,resl***2


        volume_from_heatmap_l = torch.cat(volume_from_heatmap_l, dim=1)  # 1,mm,K,resl***2
        # volume_from_heatmap_r = torch.cat(volume_from_heatmap_r, dim=1)  # 1,mm,K,resl***2

        volume_from_heatmap_l = volume_from_heatmap_l[0].transpose(1, 0)
        # volume_from_heatmap_r = volume_from_heatmap_r[0].transpose(1, 0)
        volume_from_dispmap = volume_from_dispmap[0].transpose(1, 0)

        # volume_from_heatmap = volume_from_heatmap_l[:,:64,:] - volume_from_heatmap_r[:,:64,:]
        # BRF = (volume_from_heatmap_l[:, 64:128, :] + volume_from_heatmap_r[:, 64:128, :]) / 2
        # semantic = (volume_from_heatmap_l[:, 128:, :] + volume_from_heatmap_r[:, 128:, :]) / 2
        # volume_from_heatmap = torch.exp(-(volume_from_heatmap ** 2) * (BRF ** 2))

        volume_from_heatmap = volume_from_heatmap_l[:, :128, :] * volume_from_dispmap[:, :128, :]
        semantic = volume_from_heatmap_l[:, 128:, :]
        volume_depth = torch.norm(volume_xyz_abs_list, p=2, dim=1, keepdim=True)

        # volume_from_heatmap = volume_from_heatmap_l[:, :128, :] - volume_from_heatmap_r[:, :128, :]
        # volume_from_heatmap = torch.exp(-(volume_from_heatmap ** 2)) * volume_from_heatmap_l[:, :128, :]
        # semantic = volume_from_heatmap_l[:, 128:, :]

        # volume_depth = torch.norm(volume_xyz_abs_list, p=2, dim=1, keepdim=True)

        volume_from_heatmap = torch.cat([volume_from_heatmap, volume_xyz_list, volume_depth, semantic], dim=1)

        pointNet_input_list_l.append(volume_from_heatmap_l)
        # pointNet_input_list_r.append(volume_from_heatmap_r)
        pointNet_input_list_xyz_abs.append(volume_xyz_abs_list)
        pointNet_input_consis.append(volume_from_heatmap)

    pointNet_input_tensor_l = torch.cat(pointNet_input_list_l, dim=0)
    # pointNet_input_tensor_r = torch.cat(pointNet_input_list_r, dim=0)
    pointNet_input_tensor_consis = torch.cat(pointNet_input_consis, dim=0)
    pointNet_input_tensor_xyz_abs = torch.cat(pointNet_input_list_xyz_abs, dim=0)

    input_model = {}
    input_model['input_feat_l'] = pointNet_input_tensor_l
    # input_model['input_feat_r'] = pointNet_input_tensor_r
    input_model['input_feat_xyz_abs'] = pointNet_input_tensor_xyz_abs
    input_model['input_feat_consis'] = pointNet_input_tensor_consis

    batch_for_point['dim_est'] = batch['rois'].view(-1, batch['rois'].shape[2])[:, 3:6]
    batch_for_point['pos_est'] = batch['rois'].view(-1, batch['rois'].shape[2])[:, 0:3]
    batch_for_point['ori_est_scalar'] = batch['rois'].view(-1, batch['rois'].shape[2])[:, 6]
    if phase == 'train' or phase == 'val':
        batch_for_point['dim_real'] = targets['gt_of_rois'].view(-1, targets['gt_of_rois'].shape[2])[:, 3:6]
        batch_for_point['pos_real'] = targets['gt_of_rois'].view(-1, targets['gt_of_rois'].shape[2])[:, 0:3]
        batch_for_point['ori_real'] = targets['gt_of_rois'].view(-1, targets['gt_of_rois'].shape[2])[:, 6]
        batch_for_point['dim'] = batch_for_point['dim_real'] - batch_for_point['dim_est']
        batch_for_point['pos'] = batch_for_point['pos_real'] - batch_for_point['pos_est']
        batch_for_point['ori'] = batch_for_point['ori_real'] - batch_for_point['ori_est_scalar']
        batch_for_point['reg_mask'] = targets['reg_valid_mask'] \
            .view(targets['reg_valid_mask'].shape[0] * targets['reg_valid_mask'].shape[1])
    input_model['input_batch'] = batch_for_point
    # input_model['obj_num']=obj_num
    return input_model
