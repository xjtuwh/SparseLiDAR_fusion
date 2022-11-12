import numpy as np
import torch
import torch.nn as nn
from lib.datasets.utils import class2angle

def extract_input_from_tensor(input, ind, mask):
    input = _transpose_and_gather_feat(input, ind)  # B*C*H*W --> B*K*C
    return input[mask]
def decode_detections(dets, info, calibs, cls_mean_size, threshold):
    '''
    NOTE: THIS IS A NUMPY FUNCTION
    input: dets, numpy array, shape in [batch x max_dets x dim]
    input: img_info, dict, necessary information of input images
    input: calibs, corresponding calibs for the input batch
    output:
    '''
    results = {}
    batch_rois={}
    for i in range(dets.shape[0]):  # batch
        preds = []
        rois = []
        for j in range(dets.shape[1]):  # max_dets
            cls_id = int(dets[i, j, 0])
            score = dets[i, j, 1]
            if score < threshold:
                continue

            # 2d bboxs decoding
            x = dets[i, j, 2] * info['bbox_downsample_ratio'][i][0]
            y = dets[i, j, 3] * info['bbox_downsample_ratio'][i][1]
            w = dets[i, j, 4] * info['bbox_downsample_ratio'][i][0]
            h = dets[i, j, 5] * info['bbox_downsample_ratio'][i][1]
            bbox = [x-w/2, y-h/2, x+w/2, y+h/2]

            # 3d bboxs decoding
            # depth decoding
            depth = dets[i, j, 6]

            # dimensions decoding
            dimensions = dets[i, j, 31:34]
            dimensions += cls_mean_size[int(cls_id)]

            # positions decoding
            x3d = dets[i, j, 34] * info['bbox_downsample_ratio'][i][0]
            y3d = dets[i, j, 35] * info['bbox_downsample_ratio'][i][1]
            locations = calibs[i].img_to_rect(x3d, y3d, depth).reshape(-1)
            locations[1] += dimensions[0] / 2

            # heading angle decoding
            alpha = get_heading_angle(dets[i, j, 7:31])
            ry = calibs[i].alpha2ry(alpha, x3d)

            score = score * dets[i, j, -1]

            ##### generate 2d bbox using 3d bbox
            # h, w, l = dimensions
            # x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
            # y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
            # z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
            # R = np.array([[np.cos(ry), 0, np.sin(ry)],
            #               [0, 1, 0],
            #               [-np.sin(ry), 0, np.cos(ry)]])
            # corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
            # corners3d = np.dot(R, corners3d).T
            # corners3d = corners3d + locations
            # bbox, _ = calibs[i].corners3d_to_img_boxes(corners3d.reshape(1, 8, 3))
            # bbox = bbox.reshape(-1).tolist()

            preds.append([cls_id, alpha] + bbox + dimensions.tolist() + locations.tolist() + [ry, score])
            rois.append([cls_id,score]  + locations.tolist() + dimensions.tolist() + [ry])
        results[info['img_id'][i]] = np.array(preds)
        batch_rois[info['img_id'][i]] = np.array(rois)
    return results,batch_rois

def img_to_rect(u,v,z, p2):
    """
        bbox3d: [N, 7] homo_x, homo_y, z, w, h, l, alpha
        p2: [3, 4]
        return [x3d, y3d, z, w, h, l, alpha]
    """
    fx = p2[0, 0]
    fy = p2[1, 1]
    cx = p2[0, 2]
    cy = p2[1, 2]
    tx = p2[0, 3]
    ty = p2[1, 3]

    z3d = z #[N, 1]
    x3d = (u * z3d - cx * z3d - tx) / fx #[N, 1]
    y3d = (v * z3d - cy * z3d - ty) / fy #[N, 1]
    return torch.cat([x3d.unsqueeze(1), y3d.unsqueeze(1), z3d.unsqueeze(1)], dim=1)


def alpha2ry(alpha, u, P2):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    if torch.is_tensor(alpha):
        ry = alpha + torch.atan2(u - P2[0, 2], P2[0, 0].cuda())
        for i in range(ry.size(0)):
            if ry[i] > np.pi:
                ry[i] -= 2 * np.pi
            if ry[i] < -np.pi:
                ry[i] += 2 * np.pi
    else:
        ry = alpha + np.arctan2(u - P2[0, 2], P2[0, 0])
        if ry > np.pi:
            ry -= 2 * np.pi
        if ry < -np.pi:
            ry += 2 * np.pi

    return ry

def decode_train_detections(dets, info, P2, cls_mean_size, threshold):
    '''
    NOTE: THIS IS A NUMPY FUNCTION
    input: dets, numpy array, shape in [batch x max_dets x dim]
    input: img_info, dict, necessary information of input images
    input: calibs, corresponding calibs for the input batch
    output:
    '''
    results = []
    for i in range(dets.shape[0]):  # batch
        cls_id = dets[i, :, 0]
        score = dets[i, :, 1]
        x = dets[i, :, 2] * info['bbox_downsample_ratio'][i][0]
        y = dets[i, :, 3] * info['bbox_downsample_ratio'][i][1]
        w = dets[i, :, 4] * info['bbox_downsample_ratio'][i][0]
        h = dets[i, :, 5] * info['bbox_downsample_ratio'][i][1]
        bbox = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
        depth = dets[i, :, 6]
        # dimensions decoding
        dimensions = dets[i, :, 31:34]
        dimensions += cls_mean_size[cls_id.long()]
        # positions decoding
        x3d = dets[i, :, 34] * info['bbox_downsample_ratio'][i][0].float()
        y3d = dets[i, :, 35] * info['bbox_downsample_ratio'][i][1].float()
        locations = img_to_rect(x3d, y3d, depth,P2[i])
        locations[:,1] += dimensions[:,0] / 2

        # heading angle decoding
        alpha = get_heading_angle_tensor(dets[i, :, 7:31])
        ry = alpha2ry(alpha, x3d, P2[i])

        score = score * dets[i, :, -1]

        preds = torch.cat((locations,dimensions,ry.unsqueeze(1)),dim=1)
        results.append(preds)
    return torch.stack(results)


def extract_dets_from_outputs(outputs, K=50):
    # get src outputs
    heatmap = outputs['heatmap']
    heading = outputs['heading']
    batch, channel, height, width = heatmap.size()  # get shape
    # depth = outputs['depth'][:, 0:1, :, :]
    # sigma = outputs['depth'][:, 1:2, :, :]
    depth = outputs['depth'].view(batch, K, -1)[:, :, 0:1]
    sigma = outputs['depth'].view(batch, K, -1)[:, :, 1:2]
    size_3d = outputs['size_3d']
    offset_3d = outputs['offset_3d']
    size_2d = outputs['size_2d']
    offset_2d = outputs['offset_2d']

    heatmap= torch.clamp(heatmap.sigmoid_(), min=1e-4, max=1 - 1e-4)
    # heatmap = heatmap.sigmoid_()
    depth = 1. / (depth.sigmoid() + 1e-6) - 1.
    sigma = torch.exp(-sigma)

    # perform nms on heatmaps
    heatmap = _nms(heatmap)
    scores, inds, cls_ids, xs, ys = _topk(heatmap, K=K)

    offset_2d = _transpose_and_gather_feat(offset_2d, inds)
    offset_2d = offset_2d.view(batch, K, 2)
    xs2d = xs.view(batch, K, 1) + offset_2d[:, :, 0:1]
    ys2d = ys.view(batch, K, 1) + offset_2d[:, :, 1:2]

    offset_3d = _transpose_and_gather_feat(offset_3d, inds)
    offset_3d = offset_3d.view(batch, K, 2)
    xs3d = xs.view(batch, K, 1) + offset_3d[:, :, 0:1]
    ys3d = ys.view(batch, K, 1) + offset_3d[:, :, 1:2]

    heading = _transpose_and_gather_feat(heading, inds)
    heading = heading.view(batch, K, 24)
    # depth = _transpose_and_gather_feat(depth, inds)
    # depth = depth.view(batch, K, 1)
    # sigma = _transpose_and_gather_feat(sigma, inds)
    # sigma = sigma.view(batch, K, 1)
    size_3d = _transpose_and_gather_feat(size_3d, inds)
    size_3d = size_3d.view(batch, K, 3)
    cls_ids = cls_ids.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    # check shape
    xs2d = xs2d.view(batch, K, 1)
    ys2d = ys2d.view(batch, K, 1)
    xs3d = xs3d.view(batch, K, 1)
    ys3d = ys3d.view(batch, K, 1)

    size_2d = _transpose_and_gather_feat(size_2d, inds)
    size_2d = size_2d.view(batch, K, 2)

    detections = torch.cat([cls_ids, scores, xs2d, ys2d, size_2d, depth, heading, size_3d, xs3d, ys3d, sigma], dim=2)

    return detections

def extract_dets_for_train(outputs, K, targets):
    # get src outputs
    heatmap = outputs['heatmap'].detach()
    heading = outputs['heading'].detach()
    batch, channel, height, width = heatmap.size()  # get shape
    # depth = outputs['depth'][:, 0:1, :, :].detach()
    # sigma = outputs['depth'][:, 1:2, :, :].detach()
    depth = outputs['depth'].view(batch,K,-1)[:,:,0:1].detach()
    sigma = outputs['depth'].view(batch,K,-1)[:,:,1:2].detach()
    size_3d = outputs['size_3d'].detach()
    offset_3d = outputs['offset_3d'].detach()
    size_2d = outputs['size_2d'].detach()
    offset_2d = outputs['offset_2d'].detach()

    heatmap= torch.clamp(heatmap.sigmoid_(), min=1e-4, max=1 - 1e-4)
    depth = 1. / (depth.sigmoid() + 1e-6) - 1.
    sigma = torch.exp(-sigma)

    # # perform nms on heatmaps
    heatmap = _nms(heatmap)
    scores, inds, cls_ids, xs, ys = _topk(heatmap, K=K)

    offset_2d = _transpose_and_gather_feat(offset_2d, inds)
    offset_2d = offset_2d.view(batch, K, 2)
    xs2d = xs.view(batch, K, 1) + offset_2d[:, :, 0:1]
    ys2d = ys.view(batch, K, 1) + offset_2d[:, :, 1:2]

    offset_3d = _transpose_and_gather_feat(offset_3d, inds)
    offset_3d = offset_3d.view(batch, K, 2)
    xs3d = xs.view(batch, K, 1) + offset_3d[:, :, 0:1]
    ys3d = ys.view(batch, K, 1) + offset_3d[:, :, 1:2]

    heading = _transpose_and_gather_feat(heading, inds)
    heading = heading.view(batch, K, 24)
    # depth = _transpose_and_gather_feat(depth, inds)
    # depth = depth.view(batch, K, 1)
    # sigma = _transpose_and_gather_feat(sigma, inds)
    # sigma = sigma.view(batch, K, 1)
    size_3d = _transpose_and_gather_feat(size_3d, inds)
    size_3d = size_3d.view(batch, K, 3)
    cls_ids = cls_ids.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    # check shape
    xs2d = xs2d.view(batch, K, 1)
    ys2d = ys2d.view(batch, K, 1)
    xs3d = xs3d.view(batch, K, 1)
    ys3d = ys3d.view(batch, K, 1)

    size_2d = _transpose_and_gather_feat(size_2d, inds)
    size_2d = size_2d.view(batch, K, 2)
    detections = torch.cat([cls_ids, scores, xs2d, ys2d, size_2d, depth, heading, size_3d, xs3d, ys3d, sigma], dim=2)
    return detections

############### auxiliary function ############


def _nms(heatmap, kernel=3):
    padding = (kernel - 1) // 2
    heatmapmax = nn.functional.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=padding)
    keep = (heatmapmax == heatmap).float()
    return heatmap * keep


def _topk(heatmap, K=50):
    batch, cat, height, width = heatmap.size()

    # batch * cls_ids * 50
    topk_scores, topk_inds = torch.topk(heatmap.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # batch * cls_ids * 50
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_cls_ids = (topk_ind // K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_cls_ids, topk_xs, topk_ys


def _gather_feat(feat, ind, mask=None):
    '''
    Args:
        feat: tensor shaped in B * (H*W) * C
        ind:  tensor shaped in B * K (default: 50)
        mask: tensor shaped in B * K (default: 50)

    Returns: tensor shaped in B * K or B * sum(mask)
    '''
    dim  = feat.size(2)  # get channel dim
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)  # B*len(ind) --> B*len(ind)*1 --> B*len(ind)*C
    feat = feat.gather(1, ind)  # B*(HW)*C ---> B*K*C
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)  # B*50 ---> B*K*1 --> B*K*C
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    '''
    Args:
        feat: feature maps shaped in B * C * H * W
        ind: indices tensor shaped in B * K
    Returns:
    '''
    feat = feat.permute(0, 2, 3, 1).contiguous()   # B * C * H * W ---> B * H * W * C
    feat = feat.view(feat.size(0), -1, feat.size(3))   # B * H * W * C ---> B * (H*W) * C
    feat = _gather_feat(feat, ind)     # B * len(ind) * C
    return feat


def get_heading_angle(heading):
    heading_bin, heading_res = heading[0:12], heading[12:24]
    cls = np.argmax(heading_bin)
    res = heading_res[cls]
    return class2angle(cls, res, to_label_format=True)

def get_heading_angle_tensor(heading):
    heading_bin, heading_res = heading[:,0:12], heading[:,12:24]
    cls = torch.argmax(heading_bin,dim=1)
    res = cls.new_zeros(cls.size(0)).float()
    angle = cls.new_zeros(cls.size(0)).float()
    for i in range(cls.size(0)):
        res[i] = heading_res[i,cls[i]]
        angle[i]= class2angle(cls[i].float(), res[i], to_label_format=True)
    return angle



if __name__ == '__main__':
    ## testing
    from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
    from torch.utils.data import DataLoader

    dataset = KITTI_Dataset('../../data', 'train')
    dataloader = DataLoader(dataset=dataset, batch_size=2)
