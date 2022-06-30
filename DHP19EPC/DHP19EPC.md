# Event Point Cloud Dataset of DHP19
### Dataset download
Download the DHP19 dataset following [this repo](https://github.com/SensorsINI/DHP19).
### Generate h5 files
Run `Generate_DHP19_pointcloud_all_mean.m` to generate MeanLabel setting data. Or run `Generate_DHP19_pointcloud_all_last.m` for the LastLabel setting data.
You need to generate train data and test data seperately and set path for both. Following the DHP19, S1-S12 for training and S13-S17 for testing.
Note that you should change the data path corresponding to your own path in the two files above. You may get `.h5` files at this step.
### Extract npy files
```
# For MeanLabel train/test
# Extract data and labels per frame from the h5 files, need to change path for train/test.
python extract_data_mean.py
python extract_labels_mean.py
python extract_3Dlabels.py

# Extract dict storing the event point numbers, which is used when training. Need to change path for train/test.
python extract_dict.py

# For LastLabel
# Extract data and labels per frame from the h5 files. Need to change path for train/test.
python extract_data_last.py
python extract_labels_last.py
```
Note that you should change the data path corresponding to your own path in the files above. You may get `.npy` files at this step.

For more detatils of MeanLabel and LastLabel, you can refer to [our paper](https://arxiv.org/abs/2206.04511).

### Folder Hierarchy
Your dataset folder will look like this:
```
├── DHP19EPC_dataset                                    # Store test/train data
|   ├─ test_LastLabel              
|   ├─   ├─data
|   ├─   ├─├─S13_session1_mov1_cam0.h5
|   ├─   ├─label
|   ├─   ├─├─S13_session1_mov1_cam0_label.h5
|   ├─ test_LastLabel_extract 
|   ├─ test_MeanLabel 
|   ├─ test_MeanLabel_extract 
|   ├─ train_LastLabel 
|   ├─ train_LastLabel_extract 
|   ├─   ├─data
|   ├─   ├─├─S1_session1_mov1_cam0_frame0.npy
|   ├─   ├─label
|   ├─   ├─├─S1_session1_mov1_cam0_frame0_label.npy
|   ├─ train_MeanLabel 
|   ├─   ├─data
|   ├─   ├─├─S1_session1_mov1.h5
|   ├─   ├─label
|   ├─   ├─├─S1_session1_mov1_label.h5
|   ├─ train_MeanLabel_extract
|   ├─   ├─3Dlabel
|   ├─   ├─├─S1_session1_mov1_frame0_3Dlabel.npy
|   ├─   ├─data
|   ├─   ├─├─S1_session1_mov1_frame0_cam0.npy
|   ├─   ├─label
|   ├─   ├─├─S1_session1_mov1_frame0_cam0_label.npy
|   ├─   ├─Point_Num_Dict.npy
|   ├─   ├─Point_Num_Dict_PerVideo.npy
```
