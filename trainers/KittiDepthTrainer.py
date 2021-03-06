########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import os
import sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(FILE_DIR)
# sys.path.append(BASE_DIR)
from trainers.trainer import Trainer # from CVLPyDL repo
import torch
import numpy as np

import matplotlib.pyplot as plt
import os.path
from utils.AverageMeter import AverageMeter
from utils.saveTensorToImages import saveTensorToImages
from utils.error_metrics import MAE, RMSE, MRE, Deltas
from dataloader.KittiDepthDataset import generate_depth_map

err_metrics = ['MAE', 'RMSE', 'MRE', 'Delta1', 'Delta2', 'Delta3']

class KittiDepthTrainer(Trainer):
    def __init__(self, net, params, optimizer, objective, lr_scheduler, dataloaders, dataset_sizes,
             workspace_dir, sets=['train', 'val'], use_load_checkpoint = None):

        # Call the constructor of the parent class (trainer)
        super().__init__(net, optimizer, lr_scheduler, objective, use_gpu=params['use_gpu'], workspace_dir=workspace_dir)
          
        self.lr_scheduler = lr_scheduler
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.use_load_checkpoint = use_load_checkpoint
        
        self.params = params
        self.save_chkpt_each = params['save_chkpt_each']
        self.sets = sets
        self.save_images = params['save_out_imgs']
        self.load_rgb = params['load_rgb'] if 'load_rgb' in params else False 
        
        
        for s in self.sets: self.stats[s+'_loss'] = []
                
   
####### Training Function #######

    def train(self):
        print('#############################\n### Experiment Parameters ###\n#############################')
        for k, v in self.params.items(): print('{0:<22s} : {1:}'.format(k,v))
                                                
        # Load last save checkpoint
        if self.use_load_checkpoint != None:
            if self.use_load_checkpoint > 0:
                print('\n=> Loading checkpoint {} ...'.format(self.use_load_checkpoint), end=' ')
                if self.load_checkpoint(self.use_load_checkpoint):
                    print('Checkpoint was loaded successfully!\n')
            elif self.use_load_checkpoint == -1:
                print('=> Loading last checkpoint ...', end=' ')
                if self.load_checkpoint():
                    print('Checkpoint was loaded successfully!\n')

        for epoch in range(self.epoch, self.params['num_epochs']+1): # range function returns max_epochs-1
            self.epoch = epoch
                                   
            # Decay Learning Rate 
            self.lr_scheduler.step() # LR decay
            
            print('\nTraining Epoch {}: (lr={}) '.format(epoch, self.optimizer.param_groups[0]['lr']), end=' ')

            
            # Train the epoch
            loss_meter = self.train_epoch()
            
            # Add the average loss for this epoch to stats
            for s in self.sets: self.stats[s+'_loss'].append(loss_meter[s].avg)
            
            # Save checkpoint
            if self.use_save_checkpoint and (self.epoch) % self.save_chkpt_each == 0:                    
                self.save_checkpoint()
                print('\n => Checkpoint was saved successfully!')
                        
            
        # Save the final model
        torch.save(self.net, self.workspace_dir + '/final_model.pth')        
        
        print("Training Finished.")
            
        return self.net

    def return_one_prediction(self, inputs_d, inputs_rgb, original_width=None, original_height=None):
        # define the certainty


        #assert np.size(inputs_rgb, 0) == 352 and np.size(inputs_rgb, 1) == 1216

        inputs_d = np.array(inputs_d, dtype=np.float16)
        inputs_rgb = np.array(inputs_rgb, dtype=np.float16)
        inputs_c = (inputs_d > 0).astype(float)

        # Normalize the sparse_depth
        inputs_d = inputs_d / self.params['data_normalize_factor'] # [0,1]

        # Expand dims into Pytorch format
        if np.ndim(inputs_d) == 2:
            inputs_d = np.expand_dims(inputs_d, 0)
            inputs_c = np.expand_dims(inputs_c, 0)
            inputs_d = np.expand_dims(inputs_d, 0)
            inputs_c = np.expand_dims(inputs_c, 0)
        else:
            inputs_d = np.expand_dims(inputs_d, 1)
            inputs_c = np.expand_dims(inputs_c, 1)

        # Convert to Pytorch Tensors
        inputs_d = torch.tensor(inputs_d, dtype=torch.float)
        inputs_c = torch.tensor(inputs_c, dtype=torch.float)

        # Convert depth to disparity
        if self.params['invert_depth']:
            inputs_d[inputs_d == 0] = -1
            data = 1 / inputs_d
            data[data == -1] = 0

        # Convert RGB image to tensor
        if self.load_rgb:
            inputs_rgb = np.array(inputs_rgb, dtype=np.float16)
            inputs_rgb /= 255
            if np.ndim(inputs_rgb) == 3:
                inputs_rgb = np.transpose(inputs_rgb, (2, 0, 1))
                inputs_rgb = np.expand_dims(inputs_rgb, 0)
            else:
                inputs_rgb = np.transpose(inputs_rgb, (0, 3, 1, 2))
            inputs_rgb = torch.tensor(inputs_rgb, dtype=torch.float)


        device = torch.device("cuda:"+str(self.params['gpu_id']) if torch.cuda.is_available() else "cpu")

        inputs_d = inputs_d.to(device)
        inputs_c = inputs_c.to(device)

        if self.load_rgb:
            inputs_rgb = inputs_rgb.to(device)


        with torch.no_grad():

            if self.load_rgb:
                outputs_d, outputs_c = self.net(inputs_d, inputs_c, inputs_rgb)
            else:
                outputs_d, outputs_c = self.net(inputs_d, inputs_c)

            # Convert sparse_depth to depth in meters before error metrics
            # outputs_d[outputs_d==0] = -1
            if self.params['invert_depth']:
                div0 = outputs_d == 0
                outputs_d[div0] = -1
                outputs_d = 1 / outputs_d
                outputs_d[div0] = np.max(outputs_d[not div0])
            if original_width is not None and original_height is not None:
                outputs_d = torch.nn.functional.interpolate(
                    outputs_d, (original_height, original_width), mode="bilinear", align_corners=False)
            outputs_d[outputs_d < 0] = 0
            outputs_d *= self.params['data_normalize_factor'] / 256

        return np.squeeze(outputs_d.cpu().data.numpy()), np.squeeze(outputs_c.cpu().data.numpy())


    def train_epoch(self):
        device = torch.device("cuda:"+str(self.params['gpu_id']) if torch.cuda.is_available() else "cpu")
        
        loss_meter = {}
        for s in self.sets: loss_meter[s] = AverageMeter()
          
        for s in self.sets:
            # Iterate over data.
            for data in self.dataloaders[s]:
                if self.load_rgb:
                    sparse_depth, gt_depth, computed_depth, item_idxs, inputs_rgb = data
                    sparse_depth = sparse_depth.to(device)
                    gt_depth = gt_depth.to(device)
                    inputs_rgb = inputs_rgb.to(device)
                    predicted_depth, predicted_certainty = self.net(sparse_depth, (sparse_depth > 0).float(),
                                                                    inputs_rgb)
                else:
                    sparse_depth, gt_depth, computed_depth, item_idxs = data
                    sparse_depth = sparse_depth.to(device)
                    gt_depth = gt_depth.to(device)
                    predicted_depth, predicted_certainty = self.net(sparse_depth, (sparse_depth > 0).float())

                # Calculate loss for valid pixel in the ground truth
                loss = self.objective(predicted_depth, gt_depth, predicted_certainty, self.epoch)

                # backward + optimize only if in training phase
                if s == 'train':                    
                    loss.backward()   
                    self.optimizer.step()            
                
                self.optimizer.zero_grad()
    
                # statistics
                loss_meter[s].update(loss.item(), sparse_depth.size(0))
            
            print('[{}] Loss: {:.8f}'.format(s,  loss_meter[s].avg), end=' ')

        return loss_meter

    ####### Evaluation Function #######

    def evaluate(self):
        print('< Evaluate mode ! >')
        print('#############################\n### Experiment Parameters ###\n#############################')

        # Load last save checkpoint
        if self.use_load_checkpoint != None:
            if self.use_load_checkpoint > 0:
                print('=> Loading checkpoint {} ...'.format(self.use_load_checkpoint), end=' ')
                if self.load_checkpoint(self.use_load_checkpoint):
                    print('Checkpoint was loaded successfully!\n')
                else:
                    print('Evaluating using initial parameters')
            elif self.use_load_checkpoint == -1:
                print('=> Loading last checkpoint ...', end=' ')
                if self.load_checkpoint():
                    print('Checkpoint was loaded successfully!\n')
                else:
                    print('Evaluating using initial parameters')

        self.net.train(False)

        # AverageMeters for Loss
        loss_meter = {}
        for s in self.sets: loss_meter[s] = AverageMeter()

        # AverageMeters for error metrics
        err = {}
        for m in err_metrics: err[m] = AverageMeter()

        device = torch.device('cuda:0' if self.use_gpu else 'cpu')

        with torch.no_grad():
            for s in self.sets:
                print('Evaluating on [{}] set, Epoch [{}] ! \n'.format(s, str(self.epoch - 1)))
                # Iterate over data.
                for data in self.dataloaders[s]:

                    if self.load_rgb:
                        sparse_depth, gt_depth, computed_depth, item_idxs, inputs_rgb = data
                        sparse_depth = sparse_depth.to(device)
                        gt_depth = gt_depth.to(device)
                        inputs_rgb = inputs_rgb.to(device)
                        outputs, cout = self.net(sparse_depth, (sparse_depth > 0).float(), inputs_rgb)
                    else:
                        sparse_depth, gt_depth, computed_depth, item_idxs = data
                        sparse_depth = sparse_depth.to(device)
                        gt_depth = gt_depth.to(device)
                        outputs, cout = self.net(sparse_depth, (sparse_depth > 0).float())

                    # Calculate loss for valid pixel in the ground truth
                    loss = self.objective(outputs, gt_depth, cout, self.epoch)

                    # statistics
                    loss_meter[s].update(loss.item(), sparse_depth.size(0))

                    # Convert to depth in meters before error metrics
                    outputs[outputs == 0] = -1
                    # if not self.load_rgb:
                    #     outputs[outputs==outputs[0,0,0,0]] = -1
                    gt_depth[gt_depth == 0] = -1
                    if self.params['invert_depth']:
                        outputs = 1 / outputs
                        gt_depth = 1 / gt_depth
                    outputs[outputs < 0] = 0
                    gt_depth[gt_depth < 0] = 0
                    outputs *= self.params['data_normalize_factor'] / 256
                    gt_depth *= self.params['data_normalize_factor'] / 256

                    # Calculate error metrics
                    for m in err_metrics:
                        if m.find('Delta') >= 0:
                            fn = globals()['Deltas']()
                            error = fn(outputs, gt_depth)
                            err['Delta1'].update(error[0], sparse_depth.size(0))
                            err['Delta2'].update(error[1], sparse_depth.size(0))
                            err['Delta3'].update(error[2], sparse_depth.size(0))
                            break
                        else:
                            fn = globals()[m]()
                            error = fn(outputs, gt_depth)
                            err[m].update(error.item(), sparse_depth.size(0))

                    # Save output images (optional)
                    if self.save_images and s in ['selval', 'test']:
                        outputs = outputs.data

                        outputs *= 256

                        saveTensorToImages(outputs, item_idxs, os.path.join(self.workspace_dir,
                                                                            s + '_output_' + 'epoch_' + str(
                                                                                self.epoch - 1)))
                        saveTensorToImages(cout * 255, item_idxs, os.path.join(self.workspace_dir,
                                                                               s + '_cert_' + 'epoch_' + str(
                                                                                   self.epoch - 1)))

                print('Evaluation results on [{}]:\n============================='.format(s))
                print('[{}]: {:.8f}'.format('Loss', loss_meter[s].avg))
                for m in err_metrics: print('[{}]: {:.8f}'.format(m, err[m].avg))

                # Save evaluation metric to text file
                fname = 'error_' + s + '_epoch_' + str(self.epoch - 1) + '.txt'
                with open(os.path.join(self.workspace_dir, fname), 'w') as text_file:
                    text_file.write(
                        'Evaluation results on [{}], Epoch [{}]:\n==========================================\n'.format(
                            s, str(self.epoch - 1)))
                    text_file.write('[{}]: {:.8f}\n'.format('Loss', loss_meter[s].avg))
                    for m in err_metrics: text_file.write('[{}]: {:.8f}\n'.format(m, err[m].avg))

####### Generation Function #######

    def generate(self):
        resize = True
        desired_image_height = 352
        desired_image_width = 1216

        if 'obj' in self.sets:
            # 3d object detection dataset
            set_dir = os.path.join(ROOT_DIR, '../../data/kitti_object')
            with open(os.path.join(set_dir, 'devkit_object', 'mapping', 'train_rand.txt'), 'r') as f:
                for line in f.readlines():
                    line = line.rstrip()
                    if len(line)==0: continue
                    random_mapping = np.array(line.split(','), np.int_)

            drives = np.zeros_like(random_mapping, np.object)
            days = np.zeros_like(random_mapping, np.object)
            frames = np.zeros_like(random_mapping, np.object)

            with open(os.path.join(set_dir, 'devkit_object', 'mapping', 'train_mapping.txt'), 'r') as f:
                i=0
                for line in f.readlines():
                    line = line.rstrip()
                    if len(line) == 0: continue
                    days[i], drives[i], frames[i] = line.split(' ')
                    i += 1

            drives = drives[random_mapping-1]
            days = days[random_mapping-1]
            frames = frames[random_mapping-1]

            resize = False

        # Load last save checkpoint
        if self.use_load_checkpoint != None:
            if self.use_load_checkpoint > 0:
                print('=> Loading checkpoint {} ...'.format(self.use_load_checkpoint), end=' ')
                if self.load_checkpoint(self.use_load_checkpoint):
                    print('Checkpoint was loaded successfully!\n')
                else:
                    print('Evaluating using initial parameters')
            elif self.use_load_checkpoint == -1:
                print('=> Loading last checkpoint ...', end=' ')
                if self.load_checkpoint():
                    print('Checkpoint was loaded successfully!\n')
                else:
                    print('Evaluating using initial parameters')

        from PIL import Image

        self.net.train(False)

        device = torch.device('cuda:0' if self.use_gpu else 'cpu')

        kitti_raw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../data/kitti_raw')
        res_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../data/completed_depth')
        if not os.path.isdir(res_dir):
            os.mkdir(res_dir)

        with torch.no_grad():
            for i in range(len(frames)):
                print('image %d of %d' % (i, len(days)))

                day_dir = os.path.join(kitti_raw_dir, days[i])
                drive_dir = os.path.join(day_dir, drives[i])
                rgb_filename = os.path.join(drive_dir, 'image_02', 'data', frames[i]) + ".png"
                rgb = Image.open(rgb_filename)
                img_width, img_height = rgb.size

                # rgb.save('img', 'png')
                # rgb.show('img')

                if resize:
                    rgb = rgb.resize((desired_image_width, desired_image_height), Image.LANCZOS)
                    rgb = np.array(rgb, dtype=np.float16)
                    computed_sparse_depth = generate_depth_map(days[i], drives[i], frames[i], 2,
                                                               desired_image_width, desired_image_height)
                    completed_depth, completed_certainty = self.return_one_prediction(computed_sparse_depth, rgb,
                                                                                      img_width, img_height)
                else:
                    computed_sparse_depth = generate_depth_map(days[i], drives[i], frames[i], 2)
                    completed_depth, completed_certainty = self.return_one_prediction(computed_sparse_depth, rgb)

                # import matplotlib.pyplot as plt
                # cmap = plt.cm.get_cmap('nipy_spectral', 256)
                # cmap = np.ndarray.astype(np.array([cmap(i) for i in range(256)])[:, :3] * 255, np.uint8)
                #
                # q1_lidar = np.quantile(computed_sparse_depth[computed_sparse_depth > 0], 0.05)
                # q2_lidar = np.quantile(computed_sparse_depth[computed_sparse_depth > 0], 0.95)
                # print('computed lidar quantiles: %5.2f  -  %5.2f' % (q1_lidar, q2_lidar))
                # depth_img = cmap[
                #             np.ndarray.astype(np.interp(computed_sparse_depth, (q1_lidar, q2_lidar), (0, 255)), np.int_),
                #             :]  # depths
                # fig = Image.fromarray(depth_img)
                # fig.save('computed_lidar_img', 'png')
                # fig.show('computed_lidar_img')
                #
                # q1_lidar = np.quantile(completed_depth[completed_depth > 0], 0.05)
                # q2_lidar = np.quantile(completed_depth[completed_depth > 0], 0.95)
                # depth_img = cmap[
                #             np.ndarray.astype(np.interp(completed_depth, (q1_lidar, q2_lidar), (0, 255)), np.int_),
                #             :]  # depths
                # fig = Image.fromarray(depth_img)
                # fig.save('depth_img_computed', 'png')
                # fig.show('depth_img_computed')
                #
                # input()

                np.save(os.path.join(res_dir, str(i)), (completed_depth, completed_certainty))

