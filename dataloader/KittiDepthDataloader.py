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
from dataloader.KittiDepthDataset import KittiDepthDataset

def KittiDepthDataloader(params):
    
    # Input images are 16-bit, but only 15-bits are utilized, so we normalized the sparse_depth to [0:1] using a normalization factor
    norm_factor = params['data_normalize_factor']
    invert_depth = params['invert_depth']
    kitti_depth_dir = params['kitti_depth_dataset_dir']
    kitti_rgb_dir = params['kitti_rgb_dataset_dir'] if 'kitti_rgb_dataset_dir' in params else None
    load_rgb = params['load_rgb'] if 'load_rgb' in params else False 
    rgb2gray = params['rgb2gray'] if 'rgb2gray' in params else False 
          
          
    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}
          
###### Training Set ######
    train_kitti_depth_dir = os.path.join(kitti_depth_dir, 'train')

    train_transform = transforms.Compose([transforms.CenterCrop((352, 1216))])

    image_datasets['train'] = KittiDepthDataset(train_kitti_depth_dir, setname='train',  transform=train_transform, norm_factor = norm_factor, invert_depth = invert_depth,
    load_rgb=load_rgb, rgb_dir=kitti_rgb_dir, rgb2gray=rgb2gray)
    
    # Select the desired number of images from the training set 
    if params['train_on'] != 'full':
        image_datasets['train'].sparse_depth_paths = image_datasets['train'].sparse_depth_paths[0:params['train_on']]
        image_datasets['train'].gt_depth_paths = image_datasets['train'].gt_depth_paths[0:params['train_on']]
    
    dataloaders['train'] = DataLoader(image_datasets['train'], shuffle=True, batch_size=params['train_batch_sz'], num_workers=4)
    dataset_sizes['train'] = {len(image_datasets['train'])}
    
    

###### Validation Set ######
    val_kitti_depth_dir = os.path.join(kitti_depth_dir, 'val')
    
    val_transform = transforms.Compose([transforms.CenterCrop((352,1216))])
    image_datasets['val'] = KittiDepthDataset(val_kitti_depth_dir, setname='val', transform=val_transform, norm_factor = norm_factor, invert_depth = invert_depth,
    load_rgb=load_rgb, rgb_dir=kitti_rgb_dir, rgb2gray=rgb2gray)
    dataloaders['val'] = DataLoader(image_datasets['val'], shuffle=False, batch_size=params['val_batch_sz'], num_workers=4)
    dataset_sizes['val'] = {len(image_datasets['val'])}


###### Selected Validation set ######
    selval_kitti_depth_dir = os.path.join(kitti_depth_dir, 'val_selection_cropped')

    image_datasets['selval'] = KittiDepthDataset(selval_kitti_depth_dir, setname='selval', transform=None, norm_factor = norm_factor, invert_depth = invert_depth,
                                                 load_rgb=load_rgb, rgb_dir=kitti_rgb_dir, rgb2gray=rgb2gray)

    dataloaders['selval'] = DataLoader(image_datasets['selval'], shuffle=False, batch_size=params['test_batch_sz'], num_workers=4)
    dataset_sizes['selval'] = {len(image_datasets['selval'])}


    
###### Selected test set ######
    test_kitti_depth_dir = os.path.join(kitti_depth_dir, 'test_depth_completion_anonymous')
    
    image_datasets['test'] = KittiDepthDataset(test_kitti_depth_dir, setname='test', transform=None, norm_factor = norm_factor, invert_depth = invert_depth,
    load_rgb=load_rgb, rgb_dir=kitti_rgb_dir, rgb2gray=rgb2gray)
    
    dataloaders['test'] = DataLoader(image_datasets['test'], shuffle=False, batch_size=params['test_batch_sz'], num_workers=4)
    dataset_sizes['test'] = {len(image_datasets['test'])}

    print(dataset_sizes)
    
    return dataloaders, dataset_sizes

