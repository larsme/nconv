########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from dataloader.OwnDepthDataset import OwnDepthDataset
import glob
import numpy as np

def OwnDepthDataloader(params, sets):
    
    # Input images are 16-bit, but only 15-bits are utilized, so we normalized the sparse_depth to [0:1] using a normalization factor
    norm_factor = params['data_normalize_factor']
    invert_depth = params['invert_depth']
    depth_dir = params['depth_dataset_dir']
    rgb_dir = params['rgb_dataset_dir'] if 'rgb_dataset_dir' in params else None
    assign_only_true_matches = params['assign_only_true_matches'] if 'assign_only_true_matches' in params else False
    load_rgb = params['load_rgb'] if 'load_rgb' in params else False 
    rgb2gray = params['rgb2gray'] if 'rgb2gray' in params else False
    lidar_padding = params['lidar_padding']


    calib_dir = params['calib_dir']
    undistorted_intrinsics = np.loadtxt(calib_dir+'/UndistortedIntrinsics')
    undistorted_intrinsics_old = np.loadtxt(calib_dir+'/UndistortedIntrinsicsOld')
    rvec = np.loadtxt(calib_dir+'/ExtrinsicsRVec')
    tvec = np.loadtxt(calib_dir+'/ExtrinsicsTVec')
    train_to_val_ratio = params['train_to_val_ratio']


    depth_paths = list(sorted(glob.iglob(depth_dir + "/*.bin")))
    if load_rgb or 'display' in sets:
        rgb_paths = list(sorted(glob.iglob(rgb_dir + "/*.png")))

        rgb_delay = 6
        # rgb_delay=0
        num_rgb = int(rgb_paths[0].split(rgb_dir+'/')[1].split('.png')[0])+rgb_delay
        num_depth = int(depth_paths[1].split(depth_dir+'/')[1].split('.bin')[0])
        i_rgb = 0
        i_depth = 0

        shared_rgb_paths = []
        shared_depth_paths = []

        disp_rgb_paths = []
        disp_depth_paths = []
        disp_num = 36083 + rgb_delay

        while True:
            if num_depth < num_rgb:
                i_depth += 1
                if i_depth < depth_paths.__len__():
                    num_prev_depth = num_depth
                    num_depth = int(depth_paths[i_depth].split(depth_dir+'/')[1].split('.bin')[0])
                    if num_depth > num_rgb and not assign_only_true_matches:
                        shared_rgb_paths.append(rgb_paths[i_rgb])
                        if abs(num_depth-num_rgb) <= abs(num_prev_depth- num_rgb):
                            shared_depth_paths.append(depth_paths[i_depth])
                        else:
                            shared_depth_paths.append(depth_paths[i_depth-1])
                else:
                    break
            elif num_rgb < num_depth:
                i_rgb += 1
                if i_rgb < rgb_paths.__len__():
                    num_rgb = int(rgb_paths[i_rgb].split(rgb_dir+'/')[1].split('.png')[0])+rgb_delay
                else:
                    break
            else:
                if num_rgb == disp_num:
                    disp_rgb_paths.append(rgb_paths[i_rgb])
                    disp_depth_paths.append(depth_paths[i_depth])
                shared_rgb_paths.append(rgb_paths[i_rgb])
                shared_depth_paths.append(depth_paths[i_depth])
                i_rgb += 1
                i_depth += 1
                if i_rgb < rgb_paths.__len__() and i_depth < depth_paths.__len__():
                    num_rgb = int(rgb_paths[i_rgb].split(rgb_dir+'/')[1].split('.png')[0])+rgb_delay
                    num_depth = int(depth_paths[i_rgb].split(depth_dir+'/')[1].split('.bin')[0])
                else:
                    break

        # 11398 depth paths
        # 08677 rgb paths
        # 04482 shared paths
        # 08169 with alternative

        num = shared_rgb_paths.__len__()
        train_num = round(num*train_to_val_ratio)
        val_depth_paths = shared_depth_paths[train_num: num]
        val_rgb_paths = shared_rgb_paths[train_num: num]

        if params['train_on'] != 'full' and params['train_on'] < train_num:
            train_num = params['train_on']
        train_depth_paths = shared_depth_paths[:train_num]
        train_rgb_paths = shared_rgb_paths[:train_num]
    else:
        train_rgb_paths = val_rgb_paths = None
        num = depth_paths.__len__()

        train_num = round(num*train_to_val_ratio)
        val_depth_paths = depth_paths[train_num: num]

        if params['train_on'] != 'full' and params['train_on'] < train_num:
            train_num = params['train_on']
        train_depth_paths = depth_paths[:train_num]


    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}
          
###### Training Set ######
    if 'train' in sets:
        image_datasets['train'] = OwnDepthDataset(train_depth_paths, train_rgb_paths,
                                                  rvec, tvec, undistorted_intrinsics, undistorted_intrinsics_old,
                                                  setname='train',
                                                  load_rgb=load_rgb, rgb2gray=rgb2gray,
                                                  lidar_padding=lidar_padding, image_width=2048, image_height=1536,
                                                  desired_image_width=2048, desired_image_height=1536,
                                                  do_flip=params['do_flip'], rotate_by=params['rotate_augmentation'])

        dataloaders['train'] = DataLoader(image_datasets['train'], shuffle=True, batch_size=params['train_batch_sz'],
                                          num_workers=0)
        dataset_sizes['train'] = {len(image_datasets['train'])}



###### Validation Set ######
    if 'val' in sets:
        image_datasets['val'] = OwnDepthDataset(val_depth_paths, val_rgb_paths,
                                                rvec, tvec, undistorted_intrinsics, undistorted_intrinsics_old,
                                                setname='val',
                                                load_rgb=load_rgb, rgb2gray=rgb2gray,
                                                lidar_padding=lidar_padding, image_width=2048, image_height=1536,
                                                desired_image_width=2048, desired_image_height=1536,
                                                do_flip=False, rotate_by=0)
        dataloaders['val'] = DataLoader(image_datasets['val'], shuffle=False, batch_size=params['val_batch_sz'],
                                        num_workers=0)
        dataset_sizes['val'] = {len(image_datasets['val'])}

    if 'display' in sets:
        image_datasets['display'] = OwnDepthDataset(disp_depth_paths, disp_rgb_paths,
                                                 rvec, tvec, undistorted_intrinsics, undistorted_intrinsics_old,
                                                 setname='display',
                                                 load_rgb=True, rgb2gray=rgb2gray,
                                                 lidar_padding=lidar_padding, image_width=2048, image_height=1536,
                                                 desired_image_width=2048, desired_image_height=1536,
                                                 do_flip=False, rotate_by=0)
        dataloaders['display'] = DataLoader(image_datasets['display'], shuffle=False, batch_size=1,
                                        num_workers=0)
        dataset_sizes['display'] = {len(image_datasets['display'])}

    print(dataset_sizes)

    return dataloaders, dataset_sizes

