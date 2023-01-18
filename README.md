# Probabilistic Instance Shape Reconstruction with Sparse LiDAR for Monocular 3D Object Detection

By Chaofeng Ji, Han Wu, Guizhong Liu.


## Introduction

This repository is an official implementation of the paper "Probabilistic Instance Shape Reconstruction with Sparse LiDAR for Monocular 3D Object Detection". 

## Usage

### Installation
This repo is tested on our local environment (python=3.6, cuda=10.0, pytorch=1.4), and we recommend you to use anaconda to create a vitural environment:

```bash
conda create -n SparseLiDAR python=3.6
```
Then, activate the environment:
```bash
conda activate SparseLiDAR
```

Install  Install PyTorch:

```bash
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0 -c pytorch
```
Compare the C++ and CUDA code for Guide convolution module
```bash
cd exts
python setup.py install
```

and other  requirements:
```bash
pip install -r requirements.txt
```

### Data Preparation
Please download [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and organize the data as follows:

```
#ROOT
  |data/
    |KITTI/
      |ImageSets/ [already provided in this repo]
      |object/			
        |training/
          |calib/
          |image_2/
          |label/
        |testing/
          |calib/
          |image_2/
```

### Training & Evaluation
1. Get groundtruth depthmap (skip this step if the depthmaps are provided)

```
cd data_prepare
python ptc2depthmap.py --output_path <output_path> \
    --input_path <input_pointcloud_path> \
    --calib_path <KITTI_calib_folder> \
    --image_path <KITTI_image_folder> \
    --split_file <split_file>  --threads <thread_number>
cd ..
```

2. Simulate 4-beam LiDAR

To extract 4-line LiDAR from the velodyne data provided by KITTI, run
```
cd data_prepare
python sparsify.py --calib_path <KITTI_calib_folder> \
    --image_path <KITTI_image_folder> --ptc_path <pointcloud_folder> \
    --W 1024 --H 64 --line_spec 5 7 9 11 \
    --split_files <split_file> --output_patch <output_path>
cd ..
```
3. Convert the 4-beam LiDAR to sparse depth_map

```
cd data_prepare
python ptc2depthmap.py --output_path <output_path> \
    --input_path <input_sparse_pointcloud_path> \
    --calib_path <KITTI_calib_folder> \
    --image_path <KITTI_image_folder> \
    --split_file <split_file>  --threads <thread_number>
cd ..
```

4. Move to the workplace and train the network:

```sh
 cd #ROOT
 cd experiments/example
 python ../../tools/train_val.py --config kitti_example.yaml
```
The model will be evaluated automatically if the training completed. If you only want evaluate your trained model , you can modify the test part configuration in the .yaml file and use the following command:

```sh
python ../../tools/train_val.py --config kitti_example.yaml --e
```
For ease of use, we also provide a pre-trained checkpoint and sparse depth map, which can be used for evaluation directly.
- [pretrained model](https://drive.google.com/file/d/18Di8KGhSsZHOrX5rfU7YpQuMIhMOlcLg/view?usp=share_link) (trained only on train.txt)
- [sparse depth map](https://drive.google.com/file/d/1mdL_QJnJuk_buGleWU-TyUTe3DXi4I8P/view?usp=share_link) 

## Acknowlegment

This repo benefits from the excellent work [CenterNet](https://github.com/xingyizhou/CenterNet),[monodle](https://github.com/xinzhuma/monodle),[pseudo_lidar++](https://github.com/mileyan/Pseudo_Lidar_V2),[GuideNet](https://github.com/kakaxi314/GuideNet).

## License

This project is released under the MIT License.

## Contact

If you have any question about this project, please feel free to contact chaofeng17@126.com.
