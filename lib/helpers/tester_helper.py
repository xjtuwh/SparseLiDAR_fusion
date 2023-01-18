import os
import tqdm

import torch

from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
from lib.helpers.roi_helper import featuremap2gridpoint
from lib.models.pointNet import PointNetDetector
from lib.helpers.disp2prob import LaplaceDisp2Prob
import numpy as np
import math

def project_3d(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, return_3d=False):
    """
    Projects a 3D box into 2D vertices

    Args:
    p2 (nparray): projection matrix of size 4x3
    x3d: x-coordinate of center of object
    y3d: y-coordinate of center of object
    z3d: z-cordinate of center of object
    w3d: width of object
    h3d: height of object
    l3d: length of object
    ry3d: rotation w.r.t y-axis
    """

    # compute rotational matrix around yaw axis
    R = np.array([[+math.cos(ry3d), 0, +math.sin(ry3d)],
                 [0, 1, 0],
                 [-math.sin(ry3d), 0, +math.cos(ry3d)]])

    # 3D bounding box corners
    x_corners = np.array([0, l3d, l3d, l3d, l3d, 0, 0, 0])
    y_corners = np.array([0, 0, h3d, h3d, 0, 0, h3d, h3d])
    z_corners = np.array([0, 0, 0, w3d, w3d, w3d, w3d, 0])

    x_corners += -l3d / 2
    y_corners += -h3d / 2
    z_corners += -w3d / 2
    # bounding box in object co-ordinate
    corners_3d = np.array([x_corners, y_corners, z_corners])
    corners_3d = R.dot(corners_3d)
    corners_3d += np.array([x3d, y3d, z3d]).reshape((3, 1))
    corners_3D_1 = np.vstack((corners_3d, np.ones((corners_3d.shape[-1]))))
    corners_2D = p2.dot(corners_3D_1)

    corners_2D = corners_2D / corners_2D[2]

    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]
    verts3d = (corners_2D[:, bb3d_lines_verts_idx][:2]).astype(float).T
    return verts3d, corners_3d


class Tester(object):
    def __init__(self, cfg, model, dataloader, logger, eval=False):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.max_objs = dataloader.dataset.max_objs    # max objects per images, defined in dataset
        self.class_name = dataloader.dataset.class_name
        self.output_dir = './outputs'
        self.dataset_type = cfg.get('type', 'KITTI')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.eval = eval
        # self.rcnn_head = PointNetDetector(8).cuda()


    def test(self):
        assert self.cfg['mode'] in ['single', 'all']

        # test a single checkpoint
        if self.cfg['mode'] == 'single':
            assert os.path.exists(self.cfg['checkpoint'])
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=self.cfg['checkpoint'],
                            map_location=self.device,
                            logger=self.logger)
            self.model.to(self.device)
            self.inference()
            self.evaluate()

        # test all checkpoints in the given dir
        if self.cfg['mode'] == 'all':
            checkpoints_list = []
            for _, _, files in os.walk(self.cfg['checkpoints_dir']):
                checkpoints_list = [os.path.join(self.cfg['checkpoints_dir'], f) for f in files if f.endswith(".pth")]
            checkpoints_list.sort(key=os.path.getmtime)

            for checkpoint in checkpoints_list:
                load_checkpoint(model=self.model,
                                optimizer=None,
                                filename=checkpoint,
                                map_location=self.device,
                                logger=self.logger)
                self.model.to(self.device)
                self.inference()
                self.evaluate()



    def inference(self):
        torch.set_grad_enabled(False)
        self.model.eval()
        # self.rcnn_head.eval()
        batch_dict = {}
        results = {}
        progress_bar = tqdm.tqdm(total=len(self.dataloader), leave=True, desc='Evaluation Progress')
        for batch_idx, (inputs, targets, info) in enumerate(self.dataloader):
            # load evaluation data and move data to GPU.
            inputs = inputs.to(self.device)
            targets['sparse_dep']=targets['sparse_dep'].to(self.device)
            # batch_dict['dep_map'] = disparity_pro
            outputs,left_features,dense_depth = self.model(inputs,targets['sparse_dep'], mode='test')
            depth_input, depth_log_variance = dense_depth[:, 0:1], dense_depth[:, 1:2]
            sigma = torch.exp(-depth_log_variance)
            disparity_pro, _ = LaplaceDisp2Prob(80, depth_input, variance=sigma,
                                             start_disp=0, dilation=1).getProb()
            # for (k,v) in outputs.items():
            #     outputs_train[k]=v.clone()
            dets = extract_dets_from_outputs(outputs=outputs, K=self.max_objs)
            dets = dets.detach().cpu().numpy()

            # get corresponding calibs & transform tensor to numpy
            calibs = [self.dataloader.dataset.get_calib(index)  for index in info['img_id']]
            info = {key: val.detach().cpu().numpy() for key, val in info.items()}
            cls_mean_size = self.dataloader.dataset.cls_mean_size
            dets, batch_rois = decode_detections(dets=dets,
                                     info=info,
                                     calibs=calibs,
                                     cls_mean_size=cls_mean_size,
                                     threshold=self.cfg.get('threshold', 0.2))
            index = -1
            for i in batch_rois:
                if batch_rois[i].shape[0]==0:
                    index = index+1
                    continue
                else:
                    index = index + 1
                    batch_dict['roi_labels'] = torch.tensor(batch_rois[i][:,0]).cuda().long().unsqueeze(0)
                    batch_dict['roi_scores'] = torch.tensor(batch_rois[i][:,1]).cuda().float().unsqueeze(0)
                    batch_dict['rois'] = torch.tensor(batch_rois[i][:,2:]).cuda().float().unsqueeze(0)
                    batch_dict['trans_output'] = torch.tensor(info['trans_output'][index:index+1]).cuda()
                    batch_dict['calib_l'] = torch.tensor(info['P2'][index:index+1]).cuda()
                    batch_dict['calib_r'] = torch.tensor(info['P3'][index:index+1]).cuda()
                    batch_dict['dep_map'] = disparity_pro[index:index+1]
                    batch_dict['left_image_feature'] = [left_feature[index].unsqueeze(0) for left_feature in left_features]
                    batch_dict['right_image_feature'] = [right_feature[index].unsqueeze(0) for right_feature in left_features]
                    point_data = featuremap2gridpoint(batch_dict, 'test')
                    loss_batch = point_data['input_batch']
                    outputs_rcnn = self.model.rcnn_head(point_data)
                    batch_dict['rois'][0, :, 3:6] = batch_dict['rois'][0, :, 3:6] + outputs_rcnn[:, 3:6]
                    batch_dict['rois'][0, :, 6] = batch_dict['rois'][0, :, 6] + outputs_rcnn[:, 6]
                    batch_dict['rois'][0, :, 0:3] = batch_dict['rois'][0, :, 0:3] + outputs_rcnn[:, 0:3]
                    batch_dict['batch_cls_preds'] = torch.sigmoid(outputs_rcnn[:, 7])
                    batch_dict['rois']=batch_dict['rois'].cpu().numpy()[0]
                    for j in range(batch_dict['rois'].shape[0]):
                        dim = batch_dict['rois'][j,3:6]
                        location = batch_dict['rois'][j, 0:3]
                        ry = batch_dict['rois'][j, 6]
                        cal = info['P2'][index:index+1][0]
                        b = np.array([[0,0,0,1]])
                        cal = np.insert(cal,3,values=b,axis=0)
                        verts3d, corners_3d = project_3d(cal, location[0], location[1] - dim[0] / 2, \
                                                         location[2], dim[1], dim[0], dim[2], ry, return_3d=True)
                        x = max(0, min(verts3d[:, 0]))
                        y = max(0, min(verts3d[:, 1]))
                        x2 = min(max(verts3d[:, 0]), info['img_size'][0][0])
                        y2 = min(max(verts3d[:, 1]), info['img_size'][0][1])
                        dets[i][j, 2:6] = np.array([x,y,x2,y2])
                    dets[i][:,6:9] = batch_dict['rois'][:,3:6]
                    dets[i][:, 9:12] = batch_dict['rois'][:, 0:3]
                    dets[i][:, 12] = batch_dict['rois'][:, 6]
                    for m in range(dets[i].shape[0]):
                        if dets[i][m, 0]==1.0:
                            dets[i][m, 13] = batch_dict['batch_cls_preds'][m].cpu().numpy()
                        else:
                            dets[i][m, 13] = batch_dict['batch_cls_preds'][m].cpu().numpy()*dets[i][m, 13]
                    dets[i] = dets[i].tolist()
            results.update(dets)
            progress_bar.update()

        progress_bar.close()

        # save the result for evaluation.
        self.logger.info('==> Saving ...')
        self.save_results(results)



    def save_results(self, results, output_dir='./outputs'):
        output_dir = os.path.join(output_dir, 'data')
        os.makedirs(output_dir, exist_ok=True)

        for img_id in results.keys():
            if self.dataset_type == 'KITTI':
                output_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            else:
                os.makedirs(os.path.join(output_dir, self.dataloader.dataset.get_sensor_modality(img_id)), exist_ok=True)
                output_path = os.path.join(output_dir,
                                           self.dataloader.dataset.get_sensor_modality(img_id),
                                           self.dataloader.dataset.get_sample_token(img_id) + '.txt')

            f = open(output_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()



    def evaluate(self):
        self.dataloader.dataset.eval(results_dir='./outputs/data', logger=self.logger)

