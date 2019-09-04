import os
import glob
import torch


class Trainer(object):
    """Base trainer class. Contains functions for training and saving/loading chackpoints.
    Trainer classes should inherit from this one and overload the train_epoch function."""

    def __init__(self, net, optimizer, lr_scheduler, objective, params, use_gpu=True, experiment_dir=None):

        self.net = net
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.objective = objective
        self.use_gpu = use_gpu
        self.use_save_checkpoint = experiment_dir is not None
        self.params=params

        if experiment_dir is None:
            self.experiment_dir = None
        else:
            self.experiment_dir = os.path.expanduser(experiment_dir)
            if not os.path.exists(self.experiment_dir):
                os.makedirs(self.experiment_dir)

        self.epoch = 1
        self.stats = {}

        if self.use_gpu:
            self.net.cuda()


    def train(self, max_epochs):
        """Do training for the given number of epochs."""

        for epoch in range(self.epoch, max_epochs):
            self.epoch = epoch
            self.train_epoch()
            if self.use_save_checkpoint:
                self.save_checkpoint()

        print('Finished training!')


    def train_epoch(self):
        raise NotImplementedError


    def save_checkpoint(self):
        """Saves a checkpoint of the network and other variables."""

        net_type = type(self.net).__name__
        state = {
            'epoch': self.epoch,
            'net_type': net_type,
            'net': self.net.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'stats' : self.stats,
            'use_gpu' : self.use_gpu,
        }
        
        chkpt_path = os.path.join(self.experiment_dir, 'checkpoints')
        if not os.path.exists(chkpt_path):
                os.makedirs(chkpt_path)
                
        file_path = '{}/{}_ep{:04d}.pth.tar'.format(chkpt_path, net_type, self.epoch)
        torch.save(state, file_path)


    def load_checkpoint(self, checkpoint = None):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """
        net_type = type(self.net).__name__
        
        chkpt_path = os.path.join(self.experiment_dir, 'checkpoints')
        
        if checkpoint is None:
            # Load most recent checkpoint            
            checkpoint_list = sorted(glob.glob('{}/{}_ep*.pth.tar'.format(chkpt_path, net_type)))
            if checkpoint_list:
                checkpoint = -1
                for possible_checkpoint_path in checkpoint_list:
                    epoch = int(possible_checkpoint_path.split('CNN_ep')[1].split('.pth.tar')[0])
                    if (epoch > checkpoint) and (epoch <= self.params["num_epochs"]):
                        checkpoint = epoch
                        checkpoint_path = possible_checkpoint_path
                if checkpoint == -1:
                    print('No matching checkpoint file smaller than the maximum epoch number was found!\n')
                    return False
            else:
                print('No matching checkpoint file found!\n')
                return False
        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_path = '{}/{}_ep{:04d}.pth.tar'.format(chkpt_path, net_type, checkpoint)
        elif isinstance(checkpoint, str):
            # checkpoint is the path
            checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        checkpoint_dict = torch.load(checkpoint_path, map_location={'cuda:1': 'cuda:0'})

        #assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'

        self.epoch = checkpoint_dict['epoch'] + 1
        self.net.load_state_dict(checkpoint_dict['net'])
        self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
        if 'lr_scheduler' in checkpoint_dict: 
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.lr_scheduler.last_epoch = checkpoint_dict['epoch'] 
        self.stats = checkpoint_dict['stats']
        self.use_gpu = checkpoint_dict['use_gpu']
        
        return True


