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
import cv2


class OwnDepthDataset(Dataset):
    def __init__(self, depth_paths, rgb_paths,
                 rvec, tvec, undistorted_intrinsics, undistorted_intrinsics_old,
                 setname, load_rgb, 
                 lidar_padding=0, image_width=2048, image_height=1536,
                 desired_image_width=2048, desired_image_height=1536,
                 do_flip=False, rotate_by=0, input_to_gt_ratio=0.5):
        self.setname = setname
        self.depth_paths = depth_paths
        self.rgb_paths = rgb_paths
        self.rvec = rvec
        self.tvec = tvec
        self.undistorted_intrinsics = undistorted_intrinsics
        self.undistorted_intrinsics_old = undistorted_intrinsics_old
        self.load_rgb = load_rgb
        self.image_width = image_width
        self.image_height = image_height
        self.desired_image_width = desired_image_width
        self.desired_image_height = desired_image_height
        self.lidar_padding = lidar_padding
        self.do_flip = do_flip
        self.rotate_by = rotate_by
        self.input_to_gt_ratio = input_to_gt_ratio

    def __len__(self):
        return len(self.depth_paths)

    def __getitem__(self, item):
        if item < 0 or item >= self.__len__():
            return None

        input_depth_map, gt_depth_map = self.generate_depth_maps(item)

        input_depth_map = torch.Tensor(input_depth_map).unsqueeze(0)
        gt_depth_map = torch.Tensor(gt_depth_map).unsqueeze(0)
        if self.load_rgb:
            rgb = Image.fromarray(cv2.imread(self.rgb_paths[item]))
            rgb = rgb.resize((self.desired_image_width, self.desired_image_height), Image.LANCZOS)
            rgb = np.array(rgb, dtype=np.float16)
            # Convert RGB image to tensor
            rgb /= 255
            rgb = np.transpose(rgb, (2, 0, 1))
            rgb = torch.Tensor(rgb)
            if self.do_flip:
                input_depth_map = torch.flip(input_depth_map, [2])
                gt_depth_map = torch.flip(gt_depth_map, [2])
                rgb = torch.flip(rgb, [2])
            return input_depth_map, gt_depth_map, item, rgb
        else:
            return input_depth_map, gt_depth_map, item


    def generate_depth_maps(self, item):
        """Generate a depth map from velodyne data
        Originally from monodepth2
        """

        # for item in range(self.depth_paths.__len__()):
        # X Y Z Intensity Reflectivity Noise Range Ring ... (repeat)
        # to x y z per line

        lidar_scan = np.fromfile(self.depth_paths[item], dtype=np.float32).reshape(-1, 9)
        points = lidar_scan[:, :3]

        rot = np.array(cv2.Rodrigues(self.rvec)[0])
        if not self.load_rgb:
            rot = rot.dot(rand_rotation_matrix(self.do_flip, self.rotate_by))

        projectedPoints = np.dot(self.undistorted_intrinsics,
                                 (rot.dot(points.transpose()) + np.expand_dims(self.tvec, axis=1)))
        depths = projectedPoints[2,:]
        val = depths > 0
        depths = depths[val]
        u = np.round(projectedPoints[0, val]/self.image_width*self.desired_image_width/depths).astype(np.int_)
        v = np.round(projectedPoints[1, val]/self.image_height*self.desired_image_height/depths).astype(np.int_)
        val = (u >= -self.lidar_padding) & (v >= -self.lidar_padding) \
            & (u < self.desired_image_width+self.lidar_padding) & (v < self.image_height+self.lidar_padding)

        depths = depths[val]
        v = v[val]
        u = u[val]

        random_order = np.random.permutation(range(depths.shape[0]))

        if self.setname == 'display':
            num_inputs = depths.shape[0]
        elif self.input_to_gt_ratio == "Rand":
            num_inputs = np.random.randint(1, depths.shape[0]-1)
        else:
            num_inputs = int(depths.shape[0]*self.input_to_gt_ratio)


        input = random_order[:num_inputs]
        input_depth_map = np.zeros((self.desired_image_height + 2 * self.lidar_padding,
                                    self.desired_image_width + 2 * self.lidar_padding), np.float)
        input_u = u[input]
        input_v = v[input]
        input_depths = depths[input]
        for i in range(np.array(input_u).shape[0]):
            d = input_depth_map[input_v[i]+self.lidar_padding, input_u[i]+self.lidar_padding]
            if d == 0 or d > depths[i]:
                input_depth_map[input_v[i]+self.lidar_padding, input_u[i]+self.lidar_padding] = input_depths[i]

        gt = random_order[num_inputs:]
        val_gt = (u[gt] >= 0) & (v[gt] >= 0) & (u[gt] < self.desired_image_width) & (v[gt] < self.desired_image_height)
        gt = gt[val_gt]
        gt_depth_map = np.zeros((self.desired_image_height, self.desired_image_width), np.float)
        gt_u = u[gt]
        gt_v = v[gt]
        gt_depths = depths[gt]
        for i in range(np.array(gt_u).shape[0]):
            d = gt_depth_map[gt_v[i], gt_u[i]]
            if d == 0 or d > depths[i]:
                gt_depth_map[gt_v[i], gt_u[i]] = gt_depths[i]

        gt_depth_map[v[gt], u[gt]] = depths[gt]

        return input_depth_map, gt_depth_map


def rand_rotation_matrix(do_flip, rotate_by):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """

    phi = np.random.uniform(-rotate_by / 180 * np.pi, rotate_by / 180 * np.pi)

    if do_flip:
        flip = np.random.randint(0, 2) * 2 - 1
    else:
        flip = 1

    st = np.sin(phi)
    ct = np.cos(phi)

    R = np.array(((ct, st, 0), (-st*flip, ct*flip, 0), (0, 0, 1)))

    return R
