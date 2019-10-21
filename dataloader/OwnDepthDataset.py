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
from torchvision import transforms
import numpy as np
import glob
import cv2


class OwnDepthDataset(Dataset):
    def __init__(self, depth_paths, rgb_paths,
                 rvec, tvec, undistorted_intrinsics,
                 setname, load_rgb, rgb2gray,
                 lidar_padding=0, image_width=2048, image_height=1536,
                 desired_image_width=2048, desired_image_height=1536, ):
        self.setname = setname
        self.depth_paths = depth_paths
        self.rgb_paths = rgb_paths
        self.rvec = rvec
        self.tvec = tvec
        self.undistorted_intrinsics = undistorted_intrinsics
        self.load_rgb = load_rgb
        self.rgb2gray = rgb2gray
        self.image_width = image_width
        self.image_height = image_height
        self.desired_image_width = desired_image_width
        self.desired_image_height = desired_image_height
        self.lidar_padding = lidar_padding

    def __len__(self):
        return len(self.depth_paths)

    def __getitem__(self, item):
        if item < 0 or item >= self.__len__():
            return None

        input_depth_map, gt_depth_map = self.generate_depth_maps(item)

        input_depth_map = torch.Tensor(input_depth_map).unsqueeze(0)
        gt_depth_map = torch.Tensor(gt_depth_map).unsqueeze(0)
        if self.load_rgb:
            rgb = cv2.imread(self.rgb_paths[item])
            rgb = rgb.resize((self.desired_image_width, self.desired_image_height), Image.LANCZOS)
            # Convert RGB image to tensor
            rgb /= 255
            if self.rgb2gray:
                rgb = np.expand_dims(rgb, 0)
            else:
                rgb = np.transpose(rgb, (2, 0, 1))
            rgb = torch.Tensor(rgb)
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
        if self.setname=='train':
            rot = rot.dot(rand_rotation_matrix())

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

        input = random_order[:int(depths.shape[0]/2)]
        input_depth_map = np.zeros((self.desired_image_height + 2 * self.lidar_padding,
                                    self.desired_image_width + 2 * self.lidar_padding), np.float)
        input_depth_map[v[input], u[input]] = depths[input]

        gt = random_order[int(depths.shape[0]/2):]
        val_gt = (u[gt] >= 0) & (v[gt] >= 0) & (u[gt] < self.desired_image_width) & (v[gt] < self.desired_image_height)
        gt = gt[val_gt]
        gt_depth_map = np.zeros((self.desired_image_height, self.desired_image_width), np.float)
        gt_depth_map[v[gt], u[gt]] = depths[gt]

        # depth_map = np.zeros((self.desired_image_height, self.desired_image_width), np.float)
        # depth_map[v,u]=depths
        #
        # import matplotlib.pyplot as plt
        # rgb = cv2.imread(self.rgb_paths[item])
        #
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure('pc')
        # ax = Axes3D(fig)
        # ax.plot(points[:, 0], points[:, 1], points[:, 2], '.')
        #
        # cmap = plt.cm.get_cmap('nipy_spectral', 256)
        # cmap = np.ndarray.astype(np.array([cmap(i) for i in range(256)])[:, :3] * 255, np.uint8)
        #
        # q1_lidar = np.quantile(depths, 0.05)
        # q2_lidar = np.quantile(depths, 0.95)
        # depth_img = cmap[
        #             np.ndarray.astype(np.interp(depth_map, (q1_lidar, q2_lidar), (0, 255)), np.int_),
        #             :]  # depths
        #
        # # plt.figure('rgb')
        # # plt.imshow(rgb)
        # #
        # # plt.figure('depth')
        # # plt.imshow(depth_img)
        # # plt.show()
        # # path = self.depth_paths[item][:-40]+'depth_imgs/'+self.depth_paths[item][-14:-4]+'.png'
        # # Image.fromarray(depth_img).save(path)
        #
        # Image._show(Image.fromarray(rgb))
        # Image._show(Image.fromarray(depth_img))
        # bla = 0

        return input_depth_map, gt_depth_map


def rand_rotation_matrix():
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """

    phi = np.random.uniform() * 2.0 * np.pi  # For direction of pole deflection.

    flip = np.random.randint(0, 2) * 2 - 1

    st = np.sin(phi)
    ct = np.cos(phi)

    R = np.array(((ct*flip, st*flip, 0), (-st, ct, 0), (0, 0, 1)))

    return R
