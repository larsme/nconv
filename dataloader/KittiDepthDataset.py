########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import glob


class KittiDepthDataset(Dataset):

    def __init__(self, kitti_depth_path, setname='train', norm_factor=256, invert_depth=False,
                 load_rgb=False, rgb_dir=None, 
                 resize=True, desired_image_width=1216, desired_image_height=352, lidar_padding=0, crop_top=0):
        self.kitti_depth_path = kitti_depth_path
        self.setname = setname
        self.norm_factor = norm_factor
        self.invert_depth = invert_depth
        self.load_rgb = load_rgb
        self.rgb_dir = rgb_dir
        self.resize = resize
        self.desired_image_width = desired_image_width
        self.desired_image_height = desired_image_height
        self.lidar_padding = lidar_padding
        self.crop_top = crop_top

        if setname == 'train' or setname == 'val':
            self.sparse_depth_paths = list(sorted(glob.iglob(self.kitti_depth_path + "/*/*/velodyne_raw/*/*.png",
                                                             recursive=True)))
            self.gt_depth_paths = list(sorted(glob.iglob(self.kitti_depth_path + "/*/*/groundtruth/*/*.png",
                                                         recursive=True)))
            assert (len(self.sparse_depth_paths) == len(self.gt_depth_paths))
        elif setname == 'selval':
            self.sparse_depth_paths = list(sorted(glob.iglob(self.kitti_depth_path + "/**/velodyne_raw/**.png",
                                                             recursive=True)))
            self.gt_depth_paths = list(sorted(glob.iglob(self.kitti_depth_path + "/**/groundtruth_depth/**.png",
                                                         recursive=True)))
            assert (len(self.sparse_depth_paths) == len(self.gt_depth_paths))
        else:
            self.sparse_depth_paths = list(sorted(glob.iglob(self.kitti_depth_path + "/**/velodyne_raw/**.png",
                                                             recursive=True)))
            self.gt_depth_paths = []
        self.sparse_depth_paths = [x.replace('\\','/') for x in  self.sparse_depth_paths]
        self.gt_depth_paths = [x.replace('\\','/') for x in  self.gt_depth_paths]

    def __len__(self):
        return len(self.sparse_depth_paths)

    def __getitem__(self, item):
        if item < 0 or item >= self.__len__():
            return None


        # Check if Data filename is equal to GT filename
        if self.setname == 'train' or self.setname == 'val':
            sparse_depth_path = self.sparse_depth_paths[item].split(self.setname)[1]
            gt_depth_path = self.gt_depth_paths[item].split(self.setname)[1]
            # print((sparse_depth_path, gt_depth_path))

            assert (sparse_depth_path[0:25] == gt_depth_path[0:25])  # Check folder name

            sparse_depth_path = sparse_depth_path.split('image')[1]
            gt_depth_path = gt_depth_path.split('image')[1]

            assert (sparse_depth_path == gt_depth_path)  # Check filename

            # Set the certainty path
            sep = str(self.sparse_depth_paths[item]).split('data_depth_velodyne')

            s = (self.gt_depth_paths[item].split(self.setname)[1]).split('/')
            drive_dir = s[1]
            day_dir = drive_dir.split('_drive')[0]
            img_source_dir = s[4]
            img_idx_dir = s[5].split('.png')[0]
            cam = img_source_dir.split('0')[1]
            
            if self.lidar_padding != 0:
               sparse_depth = generate_depth_map(self.rgb_dir, day_dir, drive_dir, img_idx_dir, cam,
                                                self.desired_image_width, self.desired_image_height,
                                                resize=self.resize, lidar_padding=self.lidar_padding)

        elif self.setname == 'selval':
            sparse_depth_path = self.sparse_depth_paths[item].split('00000')[1]
            gt_depth_path = self.gt_depth_paths[item].split('00000')[1]
            assert (sparse_depth_path == gt_depth_path)
            # Set the certainty path
            sep = str(self.sparse_depth_paths[item]).split('/velodyne_raw/')
            assert self.lidar_padding == 0
        else:
            assert self.lidar_padding == 0

        # Read RGB images
        if self.load_rgb:
            if self.setname == 'train' or self.setname == 'val':
                s = (self.gt_depth_paths[item].split(self.setname)[1]).split('/')
                drive_dir = s[1]
                day_dir = drive_dir.split('_drive')[0]
                img_source_dir = s[4]
                img_idx_dir = s[5]
                rgb_path = self.rgb_dir + '/' + day_dir + '/' + drive_dir + '/' + img_source_dir + '/data/' + img_idx_dir
            elif self.setname == 'selval':
                sparse_depth_path = str(self.sparse_depth_paths[item])
                idx = sparse_depth_path.find('velodyne_raw')
                fname = sparse_depth_path[idx + 12:]
                idx2 = fname.find('velodyne_raw')
                rgb_path = sparse_depth_path[:idx] + 'image' + fname[:idx2] + 'image' + fname[idx2 + 12:]
            elif self.setname == 'test':
                sparse_depth_path = str(self.sparse_depth_paths[item])
                idx = sparse_depth_path.find('velodyne_raw')
                fname = sparse_depth_path[idx + 12:]
                idx2 = fname.find('test')
                rgb_path = sparse_depth_path[:idx] + 'image' + fname[idx2 + 4:]
            rgb = Image.open(rgb_path)

        # Read images and convert them to 4D floats
        if self.lidar_padding == 0:
            sparse_depth = Image.open(str(self.sparse_depth_paths[item]))
        gt_depth = Image.open(str(self.gt_depth_paths[item]))

        if self.resize:
            if self.lidar_padding == 0:
                sparse_depth = sparse_depth.resize((self.desired_image_width, self.desired_image_height), Image.NEAREST)
            gt_depth = gt_depth.resize((self.desired_image_width, self.desired_image_height), Image.NEAREST)

            if self.load_rgb:
                rgb = rgb.resize((self.desired_image_width, self.desired_image_height), Image.LANCZOS)

        if self.crop_top >0:
            rect = (0, self.crop_top, gt_depth.size[0], gt_depth.size[1])
            if self.lidar_padding == 0:
                sparse_depth=sparse_depth.crop(rect)
            gt_depth=gt_depth.crop(rect)
            if self.load_rgb:
                rgb=rgb.crop(rect)

        # Convert to numpy
        sparse_depth = np.array(sparse_depth, dtype=np.float16)
        gt_depth = np.array(gt_depth, dtype=np.float16)

        # Normalize the depth
        sparse_depth = sparse_depth / self.norm_factor  #[0,1]
        gt_depth = gt_depth / self.norm_factor

        # Expand dims into Pytorch format
        sparse_depth = np.expand_dims(sparse_depth, 0)
        gt_depth = np.expand_dims(gt_depth, 0)

        # Convert to Pytorch Tensors
        sparse_depth = torch.tensor(sparse_depth, dtype=torch.float)
        gt_depth = torch.tensor(gt_depth, dtype=torch.float)

        # Convert depth to disparity
        if self.invert_depth:
            sparse_depth[sparse_depth == 0] = -1
            sparse_depth = 1 / sparse_depth
            sparse_depth[sparse_depth < 0] = 0

            gt_depth[gt_depth == 0] = -1
            gt_depth = 1 / gt_depth
            gt_depth[gt_depth < 0] = 0


        # Convert RGB image to tensor
        if self.load_rgb:
            rgb = np.array(rgb, dtype=np.float16)
            rgb /= 255
            rgb = np.transpose(rgb, (2, 0, 1))
            rgb = torch.tensor(rgb, dtype=torch.float)
            return sparse_depth, gt_depth, item, rgb
        else:
            return sparse_depth, gt_depth, item


def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI file format
    (adapted from https://github.com/hunse/kitti)
    """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    # x,y,z,i (intensity of reflected laser)
    points[:, 3] = 1.0  # homogeneous
    return points


def read_calib_file(filepath):
    ''' Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    '''
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def generate_depth_map(kitti_raw_dir, day, drive, frame, cam, desired_image_width=None, desired_image_height=None, lidar_padding=0, crop_top=0,
                       resize=True, vel_depth=False):
    """Generate a depth map from velodyne data
    Originally from monodepth2
    """
    import os
    calib_dir = day_dir = os.path.join(kitti_raw_dir, day)
    drive_dir = os.path.join(day_dir, drive)
    velo_filename = os.path.join(drive_dir, 'velodyne_points', 'data', frame) + ".bin"

    # load calibration files
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # get image shape
    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)
    if desired_image_width is None:
        desired_image_width = im_shape[1]
    if desired_image_height is None:
        desired_image_height = im_shape[0]

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0' + str(cam)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_filename)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    if resize:
        velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0] * desired_image_width / im_shape[1])
        velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1] * desired_image_height / im_shape[0])

    val_inds = (velo_pts_im[:, 0] >= -lidar_padding) \
               & (velo_pts_im[:, 1] >= -lidar_padding + self.crop_top) \
               & (velo_pts_im[:, 0] < desired_image_width+lidar_padding) \
               & (velo_pts_im[:, 1] < desired_image_height+lidar_padding) \
               & (velo_pts_im[:, 2] > 0)  # positive depth
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    sparse_depth_map = np.zeros((desired_image_height+2*lidar_padding, desired_image_width+2*lidar_padding), np.float)
    for i in range(velo_pts_im.shape[0]):
        px = int(velo_pts_im[i, 0]) + lidar_padding
        py = int(velo_pts_im[i, 1]) + lidar_padding- self.crop_top
        depth = velo_pts_im[i, 2]
        if sparse_depth_map[py, px] == 0 or sparse_depth_map[py, px] > depth:
            # for conflicts, use closer point
            sparse_depth_map[py, px] = depth
            # lidarmap[py, px, 2] = 1 # mask
            # lidarmap[py, px, 1] = pc_velo[i, 3]
            # lidarmap[py, px, 2] = times[i]

    return sparse_depth_map * 255
