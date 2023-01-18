# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Dequan Wang and Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging

import torch
import torch.nn as nn
#from .DCNv2.dcn_v2 import DCN
import torch.utils.model_zoo as model_zoo
from torch.autograd import Function
import GuideConv
import encoding
from scipy.stats import truncnorm
from torch.autograd import Variable
import torch.nn.functional as F
from lib.helpers.decode_helper import _nms, _topk
from lib.helpers.decode_helper import _transpose_and_gather_feat
import torchvision.ops.roi_align as roi_align
from lib.models.pointNet import PointNetDetector
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
def extract_input_from_tensor(input, ind, mask):
    input = _transpose_and_gather_feat(input, ind)
    return input[mask]
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Conv2dLocal_F(Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        output = GuideConv.Conv2dLocal_F(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input, grad_weight = GuideConv.Conv2dLocal_B(input, weight, grad_output)
        return grad_input, grad_weight


class Conv2dLocal(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, input, weight):
        output = Conv2dLocal_F.apply(input, weight)
        return output

class Basic2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, kernel_size=3, padding=1):
        super().__init__()
        if norm_layer:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=1, padding=padding, bias=False)
        else:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=1, padding=padding, bias=True)
        self.conv = nn.Sequential(conv, )
        if norm_layer:
            self.conv.add_module('bn', norm_layer(out_channels))
        self.conv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out
class Basic2dLocal(nn.Module):
    def __init__(self, out_channels, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv = Conv2dLocal()
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, weight):
        out = self.conv(input, weight)
        out = self.bn(out)
        out = self.relu(out)
        return out
class Guide(nn.Module):

    def __init__(self, input_planes, weight_planes, norm_layer=None, weight_ks=3):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.local = Basic2dLocal(input_planes, norm_layer)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv11 = Basic2d(input_planes + weight_planes, input_planes, None)
        self.conv12 = nn.Conv2d(input_planes, input_planes * 9, kernel_size=weight_ks, padding=weight_ks // 2)
        self.conv21 = Basic2d(input_planes + weight_planes, input_planes, None)
        self.conv22 = nn.Conv2d(input_planes, input_planes * input_planes, kernel_size=1, padding=0)
        self.br = nn.Sequential(
            norm_layer(num_features=input_planes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = Basic2d(input_planes, input_planes, norm_layer)

    def forward(self, input, weight):
        B, Ci, H, W = input.shape
        weight = torch.cat([input, weight], 1)
        weight11 = self.conv11(weight)
        weight12 = self.conv12(weight11)
        weight21 = self.conv21(weight)
        weight21 = self.pool(weight21)
        weight22 = self.conv22(weight21).view(B, -1, Ci)
        out = self.local(input, weight12).view(B, Ci, -1)
        out = torch.bmm(weight22, out).view(B, Ci, H, W)
        out = self.br(out)
        out = self.conv3(out)
        return out

class Basic2dTrans(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                       stride=2, padding=1, output_padding=1, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, act=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride
        self.act  = act

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.act:
            out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :] 

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
def dynamic_local_filtering(x, depth, dilated=1):
    padding = nn.ReflectionPad2d(dilated) # ConstantPad2d(1, 0)
    pad_depth = padding(depth)
    n, c, h, w = x.size()
    y = torch.cat((x[:, -1:, :, :], x[:, :-1, :, :]), dim=1)
    z = torch.cat((x[:, -2:, :, :], x[:, :-2, :, :]), dim=1)
    x = (x + y + z) / 3
    pad_x = padding(x)
    filter = (pad_depth[:, :, dilated: dilated + h, dilated: dilated + w] * pad_x[:, :, dilated: dilated + h, dilated: dilated + w]).clone()
    for i in [-dilated, 0, dilated]:
        for j in [-dilated, 0, dilated]:
            if i != 0 or j != 0:
                filter += (pad_depth[:, :, dilated + i: dilated + i + h, dilated + j: dilated + j + w] * pad_x[:, :, dilated + i: dilated + i + h, dilated + j: dilated + j + w]).clone()
    return filter / 9

class PoseResNet(nn.Module):

    def __init__(self, block, layers, heads, head_conv):
        self.inplanes = 64
        self.heads = heads
        self.deconv_with_bias = False

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.rcnn_head = PointNetDetector(8)
        self.conv_lidar = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.relu_lidar = nn.ReLU(inplace=True)
        self.maxpool_lidar = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inplanes = 64
        self.layer1_lidar = self._make_layer(block, 64, layers[0])
        self.layer2_lidar = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_lidar = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_lidar = self._make_layer(block, 512, layers[3], stride=2)

        self.guide1 = Guide(64, 64, nn.BatchNorm2d, 3)
        self.guide2 = Guide(64, 64, nn.BatchNorm2d, 3)
        self.guide3 = Guide(128, 128, nn.BatchNorm2d, 3)
        self.guide4 = Guide(256, 256, nn.BatchNorm2d, 3)
        self.ref = BasicBlock(64, 64, act=False)
        self.conv = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
        # self._initialize_weights()

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 128, 64],
            [4, 4, 4],
        )
        self.layer5d_img = Basic2dTrans(512, 256, nn.BatchNorm2d)
        self.layer4d_img = Basic2dTrans(256, 128, nn.BatchNorm2d)
        self.layer3d_img = Basic2dTrans(128, 64, nn.BatchNorm2d)
        self.layer2d_img = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3,
                                       stride=1, padding=1, output_padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.layer5d = Basic2dTrans(512, 256, nn.BatchNorm2d)
        self.layer4d = Basic2dTrans(256, 128, nn.BatchNorm2d)
        self.layer3d = Basic2dTrans(128, 64, nn.BatchNorm2d)
        self.layer2d = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3,
                                       stride=1, padding=1, output_padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.depth_img = nn.Sequential(
                         nn.Conv2d(64, head_conv // 2,
                         kernel_size=3, padding=1, bias=True),
                         nn.BatchNorm2d(32),
                         nn.ReLU(inplace=True), )
        self.depth_trans = nn.Sequential(nn.Conv2d(64, head_conv // 2,
                         kernel_size=3, padding=1, bias=True),
                         nn.BatchNorm2d(32),
                         nn.ReLU(inplace=True))
        # self.depth_trans = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
        #                                  nn.BatchNorm2d(16),
        #                                  nn.ReLU(inplace=True),
        #                                  nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        #                                  nn.BatchNorm2d(32),
        #                                  nn.ReLU(inplace=True),
        #                                  nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1))
        fill_fc_weights(self.depth_img)
        fill_fc_weights(self.depth_trans)

        self.depth = nn.Sequential(nn.Conv2d(64, 256, kernel_size=3, padding=1, bias=True),
                                   # nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
                                   nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0, bias=True))
        self.sigma = torch.nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.sigma.data.fill_(0.2)

        for head in self.heads:
            if head =='depth':
                continue
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                  nn.Conv2d(64, head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes, 
                    kernel_size=1, stride=1, 
                    padding=0, bias=True))
                if 'heatmap' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(64, classes,
                  kernel_size=1, stride=1, 
                  padding=0, bias=True)
                if 'heatmap' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def _initialize_weights(self):
        def truncated_normal_(num, mean=0., std=1.):
            lower = -2 * std
            upper = 2 * std
            X = truncnorm((lower - mean) / std, (upper - mean) / std, loc=mean, scale=std)
            samples = X.rvs(num)
            output = torch.from_numpy(samples)
            return output

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                data = truncated_normal_(m.weight.nelement(), mean=0, std=math.sqrt(1.3 * 2. / n))
                data = data.type_as(m.weight.data)
                m.weight.data = data.view_as(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            #fc = DCN(self.inplanes, planes, 
            #        kernel_size=(3,3), stride=1,
            #        padding=1, dilation=1, deformable_groups=1)
            fc = nn.Conv2d(self.inplanes, planes,
                     kernel_size=3, stride=1, 
                     padding=1, dilation=1, bias=False)
            fill_fc_weights(fc)
            up = nn.ConvTranspose2d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias)
            fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x, lidar,targets=None, K=50, mode='train'):
        lidar = lidar.unsqueeze(1)
        features = []
        batch_size = x.shape[0]
        # lidar = torch.zeros((batch_size, 1, 384, 1248)).cuda()
        device_id = x.device

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)
        c1_img = self.maxpool(x)
        c2_img = self.layer1(c1_img)
        features.append(c2_img)
        c3_img = self.layer2(c2_img)
        features.append(c3_img)
        c4_img = self.layer3(c3_img)
        c5_img = self.layer4(c4_img)
        x = self.deconv_layers(c5_img)
        dc5_img = self.layer5d_img(c5_img)
        c4_mix = dc5_img+c4_img
        dc4_img = self.layer4d_img(c4_mix)
        c3_mix = dc4_img + c3_img
        dc3_img = self.layer3d_img(c3_mix)
        c2_mix = dc3_img+c2_img
        dc2_img = self.layer2d_img(c2_mix)
        c1_mix = dc2_img + c1_img
        #----------------------Lidar-------------------------------------
        c1_lidar = self.conv_lidar(lidar)
        c1_lidar = self.relu_lidar(c1_lidar)
        c1_lidar = self.maxpool(c1_lidar)
        c2_lidar = self.layer1_lidar(c1_lidar)
        c2_lidar_dyn = self.guide2(c2_lidar,c2_mix)
        # c2_lidar_dyn = dynamic_local_filtering(c2_lidar, c2_mix)
        # c2_lidar_dyn = c2_lidar
        c3_lidar = self.layer2_lidar(c2_lidar)
        c3_lidar_dyn = self.guide3(c3_lidar, c3_mix)
        # c3_lidar_dyn = c3_lidar
        c4_lidar = self.layer3_lidar(c3_lidar)
        c4_lidar_dyn = self.guide4(c4_lidar, c4_mix)
        # c4_lidar_dyn = c4_lidar
        c5_lidar = self.layer4_lidar(c4_lidar)
        c5 = c5_img+c5_lidar
        dc5 = self.layer5d(c5)
        c4 = dc5+c4_lidar_dyn
        dc4 = self.layer4d(c4)
        c3 = dc4 + c3_lidar_dyn
        dc3 = self.layer3d(c3)
        c2 = dc3 + c2_lidar_dyn
        dc2 = self.layer2d(c2)
        c1 = dc2 + c1_lidar
        dense_depth = self.ref(c1)
        dense_depth = self.conv(dense_depth)
        ret={}
        for head in self.heads:
            if head=='depth':
                continue
            ret[head] = self.__getattr__(head)(x)
        if mode == 'train': # extract train structure in the train (only) and the val mode
            # inds, cls_ids = _topk(_nms(torch.clamp(ret['heatmap'].sigmoid(), min=1e-4, max=1 - 1e-4)), K=K)[1:3]
            # masks = torch.ones(inds.size()).bool().to(device_id)
            inds,cls_ids = targets['indices'], targets['cls_ids']
            masks = torch.ones(inds.size()).bool().to(device_id)
            # masks = torch.ones(inds.size()).bool().to(device_id)
        else:  # extract test structure in the test (only) and the val mode
            inds, cls_ids = _topk(_nms(torch.clamp(ret['heatmap'].sigmoid(), min=1e-4, max=1 - 1e-4)), K=K)[1:3]
            masks = torch.ones(inds.size()).bool().to(device_id)
        depth_c = c1.clone().detach()
        depth_feat_lidar = self.depth_trans(depth_c)
        depth_feat_img = self.depth_img(x)
        depth_feat_w0 = torch.sigmoid(depth_feat_img)
        depth_feat_w1 = torch.sigmoid(depth_feat_lidar)
        depth_feat_lidar1 = depth_feat_lidar * depth_feat_w0
        depth_feat_lidar2 = depth_feat_img * depth_feat_w1
        depth_feat = torch.cat((depth_feat_lidar1, depth_feat_lidar2), dim=1)
        # if mode == 'train':
        #     depth_out = self.get_roi_feat(depth_feat, inds1, masks, ret, cls_ids1)
        #     ret['depth_1'] = depth_out
        depth_out = self.get_roi_feat(depth_feat, inds, masks, ret, cls_ids)
        ret['depth'] = depth_out
        return ret, features, dense_depth.squeeze(1)


    def get_roi_feat(self, feat, inds, mask, ret, cls_ids):
        BATCH_SIZE, _, HEIGHT, WIDE = feat.size()
        device_id = feat.device
        coord_map = torch.cat([torch.arange(WIDE).unsqueeze(0).repeat([HEIGHT, 1]).unsqueeze(0), \
                               torch.arange(HEIGHT).unsqueeze(-1).repeat([1, WIDE]).unsqueeze(0)], 0).unsqueeze(0).repeat([BATCH_SIZE, 1, 1, 1]).type(torch.float).to(device_id)
        box2d_centre = coord_map + ret['offset_2d']
        box2d_maps = torch.cat([box2d_centre - ret['size_2d'] / 2, box2d_centre + ret['size_2d'] / 2], 1)
        box2d_maps = torch.cat([torch.arange(BATCH_SIZE).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(
            [1, 1, HEIGHT, WIDE]).type(torch.float).to(device_id), box2d_maps], 1)
        # box2d_maps is box2d in each bin
        res = self.get_roi_feat_by_mask(feat, box2d_maps, inds, mask)
        return res

    def get_roi_feat_by_mask(self, feat, box2d_maps, inds, mask):
        BATCH_SIZE, _, HEIGHT, WIDE = feat.size()
        device_id = feat.device
        num_masked_bin = mask.sum()
        # res = {}
        if num_masked_bin != 0:
            # get box2d of each roi region
            box2d_masked = extract_input_from_tensor(box2d_maps, inds, mask)
            # get roi feature
            roi_feature_masked = roi_align(feat, box2d_masked, [7, 7])
            depth_out = self.depth(roi_feature_masked)[:, :, 0, 0]
        else:
            depth_out = torch.zeros([1, 2]).to(device_id)
        return depth_out
    def init_weights(self, num_layers):
        if 1:
            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
            print('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(num_layers, heads, head_conv=256):
  block_class, layers = resnet_spec[num_layers]

  model = PoseResNet(block_class, layers, heads, head_conv=head_conv)
  model.init_weights(num_layers)
  return model
