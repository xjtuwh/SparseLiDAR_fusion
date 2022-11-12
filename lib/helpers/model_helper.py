from lib.models.centernet3d import CenterNet3D
from lib.models.resnet_dcn import get_pose_net

def build_model(cfg):
    if cfg['type'] == 'centernet3d':
        return CenterNet3D(backbone=cfg['backbone'], neck=cfg['neck'], num_class=cfg['num_class'])
    elif cfg['type'] == 'resnet34':
        heads = {'heatmap': 3, 'offset_2d': 2, 'size_2d': 2, 'depth': 2, 'offset_3d': 2, 'size_3d': 3,
                 'heading': 24}

        # heads = {'heatmap': 3, 'depth': 2, 'offset_3d': 2, 'size_3d': 3,
        #          'heading': 24}

        return get_pose_net(num_layers=34, heads=heads, head_conv=64)
        # raise NotImplementedError("%s model is not supported" % cfg['type'])
    else:
        raise NotImplementedError("%s model is not supported" % cfg['type'])



