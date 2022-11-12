import os
import numpy as np
import torch.utils.data as data
from PIL import Image
from skimage import io
import skimage.transform
from lib.datasets.utils import angle2class
from lib.datasets.utils import gaussian_radius
from lib.datasets.utils import draw_umich_gaussian
from lib.datasets.kitti.kitti_utils import get_objects_from_label
from lib.datasets.kitti.kitti_utils import Calibration
from lib.datasets.kitti.kitti_utils import get_affine_transform
from lib.datasets.kitti.kitti_utils import affine_transform
from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result
from lib.datasets.kitti.kitti_eval_python.eval import get_distance_eval_result
import lib.datasets.kitti.kitti_eval_python.kitti_common as kitti
import math
import random


class KITTI_Dataset(data.Dataset):
    def __init__(self, split, cfg):
        # basic configuration
        self.root_dir = self.root_dir = cfg.get('root_dir', '/media/zd/2T/jcf/sparsepoints_fusion/data/KITTI')
        self.split = split
        self.num_classes = 3
        self.max_objs = 50
        self.max_objs1 = 16
        self.class_name = ['Pedestrian', 'Car', 'Cyclist']
        self.cls2id = {'Pedestrian': 0, 'Car': 1, 'Cyclist': 2}
        self.resolution = np.array([1248, 384])  # W * H
        self.use_3d_center = cfg.get('use_3d_center', True)
        self.writelist = cfg.get('writelist', ['Car'])
        # anno: use src annotations as GT, proj: use projected 2d bboxes as GT
        self.bbox2d_type = cfg.get('bbox2d_type', 'anno')
        assert self.bbox2d_type in ['anno', 'proj']
        self.meanshape = cfg.get('meanshape', False)
        self.class_merging = cfg.get('class_merging', False)
        self.use_dontcare = cfg.get('use_dontcare', False)

        if self.class_merging:
            self.writelist.extend(['Van', 'Truck'])
        if self.use_dontcare:
            self.writelist.extend(['DontCare'])


        # data split loading
        assert self.split in ['train', 'val', 'trainval', 'test']
        self.split_file = os.path.join(self.root_dir, 'ImageSets', self.split + '.txt')
        self.idx_list = [x.strip() for x in open(self.split_file).readlines()]

        # path configuration
        self.data_dir = os.path.join(self.root_dir, 'object', 'testing' if split == 'test' else 'training')
        self.image_dir = os.path.join(self.data_dir, 'image_2')
        self.imageR_dir = os.path.join(self.data_dir, 'image_3')
        # self.depth_dir = os.path.join(self.data_dir, 'depth')
        self.depth_dir = '/media/zd/2T/jcf/sparsepoints_fusion/depth_map'
        self.beam4depth_dir = '/media/zd/2T/jcf/sparsepoints_fusion/beam4_depth'
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'label_2')

        # data augmentation configuration
        self.data_augmentation = True if split in ['train', 'trainval'] else False
        self.random_flip = cfg.get('random_flip', 0.5)
        self.random_crop = cfg.get('random_crop', 0.5)
        self.scale = cfg.get('scale', 0.4)
        self.shift = cfg.get('shift', 0.1)

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.cls_mean_size = np.array([[1.76255119, 0.66068622, 0.84422524],
                                       [1.52563191, 1.62856739, 3.52588311],
                                       [1.73698127, 0.59706367, 1.76282397]], dtype=np.float32)  # H*W*L
        if not self.meanshape:
            self.cls_mean_size = np.zeros_like(self.cls_mean_size, dtype=np.float32)

        # others
        self.downsample = 4

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return Image.open(img_file)

    def get_imageR(self, idx):
        img_file = os.path.join(self.imageR_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return Image.open(img_file)

    # def get_depth_map(self, idx):
    #     depth_file = os.path.join(self.depth_dir, '%06d.png' % idx)
    #     assert os.path.exists(depth_file)
    #     depth1 = Image.open(depth_file)
    #     return depth1
    def get_depth_map(self, idx):
        depth_file = os.path.join(self.depth_dir, '%06d.png' % idx)
        assert os.path.exists(depth_file)
        depth1 = Image.open(depth_file)
        return depth1
    def get_beam4depth_map(self, idx):
        depth_file = os.path.join(self.beam4depth_dir, '%06d.png' % idx)
        assert os.path.exists(depth_file)
        depth1 = Image.open(depth_file)
        return depth1

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)


    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return Calibration(calib_file)

    def eval(self, results_dir, logger):
        logger.info("==> Loading detections and GTs...")
        img_ids = [int(id) for id in self.idx_list]
        dt_annos = kitti.get_label_annos(results_dir)
        gt_annos = kitti.get_label_annos(self.label_dir, img_ids)

        test_id = {'Car': 0, 'Pedestrian':1, 'Cyclist': 2}

        logger.info('==> Evaluating (official) ...')
        for category in self.writelist:
            results_str, results_dict = get_official_eval_result(gt_annos, dt_annos, test_id[category])
            logger.info(results_str)
    def E2R(self,Ry):
        '''Combine Euler angles to the rotation matrix (right-hand)

            Inputs:
                Ry, Rx, Rz : rotation angles along  y, x, z axis
                             only has Ry in the KITTI dataset
            Returns:
                3 x 3 rotation matrix

        '''
        R_yaw = np.array([[math.cos(Ry), 0, math.sin(Ry)],
                          [0, 1, 0],
                          [-math.sin(Ry), 0, math.cos(Ry)]])
        return R_yaw


    def __len__(self):
        return self.idx_list.__len__()

    def generate_corners3d(self, object):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = object.l, object.h, object.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(object.ry), 0, np.sin(object.ry)],
                      [0, 1, 0],
                      [-np.sin(object.ry), 0, np.cos(object.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + object.pos
        return corners3d
    def corners3d_to_img_boxes(self, corners3d, P2):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, P2.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner


    def __getitem__(self, item):
        #  ============================   get inputs   ===========================
        index = int(self.idx_list[item])  # index mapping, get real data id
        # image loading
        img = self.get_image(index)
        img_R = self.get_imageR(index)
        dep_map = self.get_depth_map(index)
        sparse_depth = self.get_beam4depth_map(index)
        img_size = np.array(img.size)
        features_size = self.resolution // self.downsample    # W * H

        # data augmentation for image
        center = np.array(img_size) / 2
        aug_scale, crop_size = 1.0, img_size
        random_crop_flag, random_flip_flag = False, False
        if self.data_augmentation:
            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                img_R = img_R.transpose(Image.FLIP_LEFT_RIGHT)
                # dep_map = dep_map.transpose(Image.FLIP_LEFT_RIGHT)
                dep_map = dep_map.transpose(Image.FLIP_LEFT_RIGHT)
                sparse_depth = sparse_depth.transpose(Image.FLIP_LEFT_RIGHT)
                # img, img_R = img_R, img

            # if np.random.random() < self.random_crop:
            #     random_crop_flag = True
            #     aug_scale = np.clip(np.random.randn() * self.scale + 1, 1 - self.scale, 1 + self.scale)
            #     crop_size = img_size * aug_scale
            #     center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
            #     center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)

        # add affine transformation for 2d images.
        # trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        # img = img.transform(tuple(self.resolution.tolist()),
        #                     method=Image.AFFINE,
        #                     data=tuple(trans_inv.reshape(-1).tolist()),
        #                     resample=Image.BILINEAR)
        # img_R = img_R.transform(tuple(self.resolution.tolist()),
        #                         method=Image.AFFINE,
        #                         data=tuple(trans_inv.reshape(-1).tolist()),
        #                         resample=Image.BILINEAR)
        # dep_map = dep_map.transform(tuple(self.resolution.tolist()),
        #                             method=Image.AFFINE,
        #                             data=tuple(trans_inv.reshape(-1).tolist()),
        #                             resample=Image.BILINEAR)
        # image encoding
        pad_h = (0, self.resolution[1] - img_size[1])
        pad_w = (0, self.resolution[0] - img_size[0])
        pad_width = (pad_h, pad_w, (0, 0))
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.pad(img, pad_width=pad_width,
                     mode='constant',
                     constant_values=0)
        img = img.transpose(2, 0, 1)

        dep_map = np.array(dep_map).astype(np.float32) / 256
        pad_h = (0, self.resolution[1] - dep_map.shape[0])
        pad_w = (0, self.resolution[0] - dep_map.shape[1])
        pad_width = (pad_h, pad_w)
        dep_map = np.pad(dep_map, pad_width=pad_width,
                         mode='constant',
                         constant_values=0)
        # dep_map = skimage.transform.downscale_local_mean(image=dep_map,factors=(4, 4))
        dep_map = skimage.measure.block_reduce(dep_map, (4, 4), np.max)

        sparse_depth = np.array(sparse_depth).astype(np.float32)/256
        pad_h = (0, self.resolution[1] - sparse_depth.shape[0])
        pad_w = (0, self.resolution[0] - sparse_depth.shape[1])
        pad_width = (pad_h, pad_w)
        sparse_depth = np.pad(sparse_depth, pad_width=pad_width,
                         mode='constant',
                         constant_values=0)
        # sparse_depth = skimage.transform.downscale_local_mean(image=sparse_depth, factors=(4, 4))
        info = {'img_id': index,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size/features_size}

        if self.split == 'test':
            return img, info   # img / placeholder(fake label) / info


        #  ============================   get labels   ==============================
        objects = self.get_label(index)
        calib = self.get_calib(index)
        calib_l = calib.P2
        calib_r = calib.P3
        # computed 3d projected box
        # if self.bbox2d_type == 'proj':


        # data augmentation for labels
        if random_flip_flag:
            # calib_l, calib_r = calib_r, calib_l
            calib_l[0, 3] = -calib_l[0, 3]
            calib_l[0, 2] = img_size[0] - calib_l[0, 2] - 1
            calib_r[0, 3] = -calib_r[0, 3]
            calib_r[0, 2] = img_size[0] - calib_r[0, 2] - 1

            for object in objects:
                # [x1, _, x2, _] = object.box2d
                # object.box2d[0],  object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                object.pos[0] = -object.pos[0]
                object.alpha = np.pi - object.alpha
                object.ry = np.pi - object.ry
                if object.alpha > np.pi:  object.alpha -= 2 * np.pi  # check range
                if object.alpha < -np.pi: object.alpha += 2 * np.pi
                if object.ry > np.pi:  object.ry -= 2 * np.pi
                if object.ry < -np.pi: object.ry += 2 * np.pi
        if 1:
            for object in objects:
                temp = np.array(calib.corners3d_to_img_boxes(object.generate_corners3d(object)[None, :],calib_l)[0][0],
                                             dtype=np.float32)
                x = max(0, temp[0])
                y = max(0, temp[1])
                x2 = min(temp[2], img_size[0].astype(np.float32))
                y2 = min(temp[3], img_size[1].astype(np.float32))
                object.box2d_proj = np.array([x,y,x2,y2])
                object.box2d = object.box2d_proj.copy()


        # labels encoding
        heatmap = np.zeros((self.num_classes, features_size[1], features_size[0]), dtype=np.float32) # C * H * W
        size_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
        offset_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
        depth = np.zeros((self.max_objs, 1), dtype=np.float32)
        heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
        heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
        src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        offset_3d = np.zeros((self.max_objs, 2), dtype=np.float32)
        indices = np.zeros((self.max_objs), dtype=np.int64)
        mask_2d = np.zeros((self.max_objs), dtype=np.uint8)
        mask_3d = np.zeros((self.max_objs), dtype=np.uint8)
        mask_3d1 = np.zeros((self.max_objs), dtype=np.uint8)
        cls_ids = np.zeros((self.max_objs), dtype = np.int64)
        object_num = len(objects) if len(objects) < self.max_objs else self.max_objs
        bbox3d = np.ones((self.max_objs, 8), dtype=np.float32)* -1
        bbox3d_list = []
        for i in range(object_num):
            # filter objects by writelist
            if objects[i].cls_type not in self.writelist:
                continue

            # filter inappropriate samples
            if objects[i].level_str == 'UnKnown' or objects[i].pos[-1] < 2:
                continue

            # ignore the samples beyond the threshold [hard encoding]
            threshold = 65
            if objects[i].pos[-1] > threshold:
                continue

            # process 2d bbox & get 2d center
            bbox_2d = objects[i].box2d.copy()

            # add affine transformation for 2d boxes.

            # bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
            # bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)
            # modify the 2d bbox according to pre-compute downsample ratio
            bbox_2d[:] /= self.downsample

            # process 3d bbox & get 3d center
            center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2], dtype=np.float32)  # W * H
            center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
            center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
            center_3d, _ = calib.rect_to_img(center_3d,calib_l)  # project 3D center to image plane
            center_3d = center_3d[0]  # shape adjustment
            # if random_flip_flag:  # random flip for center3d
            #     center_3d[0] = img_size[0] - center_3d[0]
            # center_3d = affine_transform(center_3d.reshape(-1), trans)
            center_3d /= self.downsample

            # generate the center of gaussian heatmap [optional: 3d center or 2d center]
            center_heatmap = center_3d.astype(np.int32) if self.use_3d_center else center_2d.astype(np.int32)
            if center_heatmap[0] < 0 or center_heatmap[0] >= features_size[0]: continue
            if center_heatmap[1] < 0 or center_heatmap[1] >= features_size[1]: continue

            # generate the radius of gaussian heatmap
            w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
            radius = gaussian_radius((w, h))
            radius = max(0, int(radius))

            if objects[i].cls_type in ['Van', 'Truck', 'DontCare']:
                draw_umich_gaussian(heatmap[1], center_heatmap, radius)
                continue

            cls_id = self.cls2id[objects[i].cls_type]
            cls_ids[i] = cls_id
            draw_umich_gaussian(heatmap[cls_id], center_heatmap, radius)

            # encoding 2d/3d offset & 2d size
            indices[i] = center_heatmap[1] * features_size[0] + center_heatmap[0]
            offset_2d[i] = center_2d - center_heatmap
            size_2d[i] = 1. * w, 1. * h

            # encoding depth
            depth[i] = objects[i].pos[-1] * aug_scale

            # encoding heading angle
            heading_angle = objects[i].alpha
            heading_bin[i], heading_res[i] = angle2class(heading_angle)

            # encoding 3d offset & size_3d
            offset_3d[i] = center_3d - center_heatmap
            src_size_3d[i] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
            mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
            size_3d[i] = src_size_3d[i] - mean_size

            mask_2d[i] = 1
            mask_3d[i] = 0 if random_crop_flag else 1
            mask_3d1[i] = 0 if random_crop_flag else 1
            bbox3d_list.append(np.concatenate([objects[i].pos,
                                             [objects[i].h, objects[i].w, objects[i].l, objects[i].ry], [cls_id]]))

        # if self.split=='train' and mask_2d.sum() ==0:
        #     new_index = np.random.randint(self.__len__())
        #     return self.__getitem__(new_index)

        if len(bbox3d_list)>0:
            bbox3d[0:len(bbox3d_list), :] = np.array(bbox3d_list)

        trans_output_l = np.zeros((3, 2, 3), dtype=np.float32)
        # trans_output_r = np.zeros((3, 2, 3), dtype=np.float32)
        for j in range(3):
            down_ratio = math.pow(2, j + 1)
            trans_output_l[j, :, :] = get_affine_transform(
                center, crop_size, 0, [self.resolution[0] // down_ratio, self.resolution[1] // down_ratio])
        #----------------------------------------------------------------------------------------------------
        # collect return data
        inputs = img
        # inputs = np.concatenate((img, img_R), axis=0)
        targets = {'depth': depth,
                   'size_2d': size_2d,
                   'heatmap': heatmap,
                   'offset_2d': offset_2d,
                   'indices': indices,
                   'size_3d': size_3d,
                   'src_size_3d': src_size_3d,
                   'offset_3d': offset_3d,
                   'heading_bin': heading_bin,
                   'heading_res': heading_res,
                   'mask_2d': mask_2d,
                   'mask_3d': mask_3d,
                   'mask_3d1': mask_3d1,
                   'bbox3d': bbox3d,
                   'dep_map':dep_map,
                   'sparse_dep':sparse_depth,
                   'cls_ids': cls_ids}
        info = {'img_id': index,
                'img_size': img_size,
                'P2': calib_l,
                'P3': calib_r,
                'trans_output': trans_output_l,
                'bbox_downsample_ratio': np.array([self.downsample,self.downsample])}
        if self.split == 'train':
            return inputs, targets, info
        else:
            return inputs, targets, info

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    cfg = {'root_dir': '../../../data/KITTI',
           'random_flip':0.0, 'random_crop':1.0, 'scale':0.8, 'shift':0.1, 'use_dontcare': False,
           'class_merging': False, 'writelist':['Pedestrian', 'Car', 'Cyclist'], 'use_3d_center':False}
    dataset = KITTI_Dataset('train', cfg)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    print(dataset.writelist)

    for batch_idx, (inputs, targets, info) in enumerate(dataloader):
        # test image
        img = inputs[0].numpy().transpose(1, 2, 0)
        img = (img * dataset.std + dataset.mean) * 255
        img = Image.fromarray(img.astype(np.uint8))
        img.show()
        # print(targets['size_3d'][0][0])

        # test heatmap
        heatmap = targets['heatmap'][0]  # image id
        heatmap = Image.fromarray(heatmap[0].numpy() * 255)  # cats id
        heatmap.show()

        break


    # print ground truth fisrt
    objects = dataset.get_label(0)
    for object in objects:
        print(object.to_kitti_format())
