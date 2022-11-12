# Probabilistic Instance Shape Reconstruction with Sparse LiDAR for Monocular 3D Object Detection

By Chaofeng Ji, Han Wu Guizhong Liu.


## Introduction

This repository is an official implementation of the paper "Probabilistic Instance Shape Reconstruction with Sparse LiDAR for Monocular 3D Object Detection". 

## Usage

### Installation
This repo is tested on our local environment (python=3.6, cuda=10.0, pytorch=1.4), and we recommend you to use anaconda to create a vitural environment:

```bash
conda create -n monodle python=3.6
```
Then, activate the environment:
```bash
conda activate monodle
```

Install  Install PyTorch:

```bash
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0 -c pytorch
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

Move to the workplace and train the network:

```sh
 cd #ROOT
 cd experiments/example
 python ../../tools/train_val.py --config kitti_example.yaml
```
The model will be evaluated automatically if the training completed. If you only want evaluate your trained model , you can modify the test part configuration in the .yaml file and use the following command:

```sh
python ../../tools/train_val.py --config kitti_example.yaml --e
```

## Citation



## Acknowlegment

This repo benefits from the excellent work [CenterNet](https://github.com/xingyizhou/CenterNet),[monodle](https://github.com/xinzhuma/monodle).

## License

This project is released under the MIT License.

## Contact

If you have any question about this project, please feel free to contact chaofeng17@126.com.
"# SparseLiDAR_fusion" 
