# Efficient Human Pose Estimation via 3D Event Point Cloud
The official Pytorch implementations of [Efficient Human Pose Estimation via 3D Event Point Cloud](https://arxiv.org/abs/2206.04511). 
We propose a novel 3D event point cloud based paradigm for human pose estimation and achieve efficient results on DHP19 dataset.


## Dependencies
We test the project with the following dependencies.
* pytorch == 1.8.0+cu111
* torchvision == 0.9.0+cu111
* numpy == 1.19.2
* opencv-python == 4.4.0
* h5py == 3.3.0
* Win10 or Ubuntu18.04

## Getting started
### Dataset preparation
Download [DHP19](https://github.com/SensorsINI/DHP19) dataset and generate following [DHP19EPC](DHP19EPC/DHP19EPC.md).

### Folder Hierarchy
Your work space will look like this(note to change the data path in the codes to your own path):
```
├── DHP19EPC_dataset               # Store test/train data
|   ├─ ...                         # MeanLabel and LastLabel
├── EventPointPose                 # This repository
|   ├─ checkpoints                 # Checkpoints and debug images
|   ├─ dataset                     # Dataset
|   ├─ DHP19EPC                    # To generate data for DHP19EPC_dataset
|   ├─ evaluate                    # Evaluate model and save gif/mp4
|   ├─ logs                        # Training logs
|   ├─ models                      # Models
|   ├─ P_matrices                  # Matrices in DHP19
|   ├─ results                     # Store results or our pretrained models
|   ├─ tools                       # Utility functions
|   ├─ main.py                     # train/eval model
```

### Train model
```
cd ./EventPointPose

# train MeanLabel
python main.py --train_batch_size=16 --epochs=30 --num_points=2048 --model PointNet --name PointNet-2048 --cuda_num 0

# train LastLabel
python main.py --train_batch_size=16 --epochs=30 --num_points=2048 --model PointNet --name PointNet-2048-last --cuda_num 0 --label last
```

### Evaluate model
You can evaluate your model and output gif as well as videos following this [doc](evaluate/Evaluate.md).

### Pretrained Model
Our pretrained model in the paper can be found here: [Baidu Cloud](https://pan.baidu.com/s/1kzkLpghqwQFhU7pjrDHKlA?pwd=y61z) or [Google Drive](https://drive.google.com/drive/folders/1uYEHfMVNThp5gTKyUlNVw9Xb895Uo_LL?usp=sharing)

## Citation
If you find our project helpful in your research, please cite with:
```
@article{chen2022EPP,
  title={Efficient Human Pose Estimation via 3D Event Point Cloud},
  author={Chen, Jiaan and Shi, Hao and Ye, Yaozu and Yang, Kailun and Sun, Lei and Wang, Kaiwei},
  journal={arXiv preprint arXiv:2206.04511},
  year={2022}
}
```

For any questions, welcome to e-mail us: chenjiaan@zju.edu.cn, haoshi@zju.edu.cn, and we will try our best to help you. =)

## Acknowledgement
Thanks for these repositories:

[DHP19](https://github.com/SensorsINI/DHP19), [Simple baseline pose](https://github.com/microsoft/human-pose-estimation.pytorch), [SimDR](https://github.com/leeyegy/SimDR), [DGCNN](https://github.com/WangYueFt/dgcnn), [Point-Trans](https://github.com/qq456cvb/Point-Transformers)
