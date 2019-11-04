########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################


import os
import sys
import importlib
import json
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from dataloader.KittiDepthDataloader import KittiDepthDataloader
from dataloader.OwnDepthDataloader import OwnDepthDataloader
from modules.losses import ConfLossDecay, SmoothL1Loss, MSELoss

# Fix CUDNN error for non-contiguous inputs
import torch.backends.cudnn as cudnn
cudnn.enabled = True
cudnn.benchmark = True

def count_parameters(network_file=None, params_sub_dir=None, training_ws_path='workspace'):
    if training_ws_path is None:
        net_dir = BASE_DIR
        network_path = network_file
    else:
        net_dir = os.path.join(BASE_DIR, training_ws_path)
        assert os.path.isdir(net_dir)
        network_path = os.path.join(training_ws_path, network_file)
    if params_sub_dir is None:
        experiment_dir = net_dir
    else:
        experiment_dir = os.path.join(net_dir, params_sub_dir)
        assert os.path.isdir(experiment_dir)
    params_path = os.path.join(experiment_dir, 'params.json')
    assert os.path.isfile(params_path)

    # Read parameters file
    with open(params_path, 'r') as fp:
        params = json.load(fp)

    # Import the network file
    f = importlib.import_module(network_path.replace('/', '.'))
    model = f.CNN(params)

    parameter_count = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter_count += parameter.numel()
    print('%s \n with parameters %s \n has %s parameters\n' %(network_path, params_path, parameter_count))


def load_net(mode='eval', sets=None, checkpoint_num=-1,
             training_ws_path='workspace', network_file=None, params_sub_dir=None, exp_subdir=None):
    assert network_file is not None

    if training_ws_path is None:
        net_dir = BASE_DIR
        network_path = network_file
    else:
        net_dir = os.path.join(BASE_DIR, training_ws_path)
        assert os.path.isdir(net_dir)
        network_path = os.path.join(training_ws_path, network_file)
    if params_sub_dir is None:
        params_dir = net_dir
    else:
        params_dir = os.path.join(net_dir, params_sub_dir)
        assert os.path.isdir(params_dir)
    params_path = os.path.join(params_dir, 'params.json')
    assert os.path.isfile(params_path)
    if exp_subdir is None:
        exp_dir = params_dir
    else:
        exp_dir = os.path.join(params_dir, exp_subdir)
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)

    # Read parameters file
    with open(params_path, 'r') as fp:
        params = json.load(fp)
    params['experiment path:'] = params_path

    # Use GPU or not
    device = torch.device("cuda:"+str(params['gpu_id']) if torch.cuda.is_available() else "cpu")

    if 'kitti_rgb_dataset_dir' in params:
        # Dataloader for KittiDepth
        dataloaders, dataset_sizes = KittiDepthDataloader(params, sets)
    else:
        # Dataloader for Own Depth
        dataloaders, dataset_sizes = OwnDepthDataloader(params, sets)

    # Import the network file
    f = importlib.import_module(network_path.replace('/', '.'))
    model = f.CNN(params).to(device)

    # Import the trainer
    t = importlib.import_module('trainers.'+params['trainer'])

    if sets is None:
        if mode == 'train':
            sets = ['train', 'selval'] #  train  selval
        elif mode == 'eval':
            sets = ['val']
        else:
            sets = []

    with torch.cuda.device(params['gpu_id']):
        # Objective function
        objective = globals()[params['loss']]()


        # Optimize only parameters that requires_grad
        parameters = filter(lambda p: p.requires_grad, model.parameters())


        # The optimizer
        optimizer = getattr(optim, params['optimizer'])(parameters, lr=params['lr'],
                                                        weight_decay=params['weight_decay'])


        # Decay LR by a factor of 0.1 every exp_dir7 epochs
        lr_decay = lr_scheduler.StepLR(optimizer, step_size=params['lr_decay_step'], gamma=params['lr_decay'])


        mytrainer = t.KittiDepthTrainer(model, params, optimizer, objective, lr_decay, dataloaders, dataset_sizes,
                                        experiment_dir=exp_dir,
                                        sets=sets, use_load_checkpoint=checkpoint_num)

    return mytrainer


def predict(mytrainer, inputs_d, inputs_rgb):
    return mytrainer.return_one_prediction(inputs_d, inputs_rgb)


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', action='store', default=None, dest='mode',
                        help='"eval", "train", "traineval", "gen" or "count_parameters" mode')
    parser.add_argument('-ws_path', action='store', default=None, dest='ws_path',
                        help='Workspace directory')
    parser.add_argument('-network_file', action='store', default=None, dest='network_file',
                        help='Python file in workspace directory')
    parser.add_argument('-params_sub_dir', action='store', default=None, dest='params_sub_dir',
                        help='Params file in workspace directory')
    parser.add_argument('-exp_subdir', action='store', default=None, dest='exp_subdir',
                        help='Experiment subdir in params dir for running multiple experiments with same parameters')
    parser.add_argument('-checkpoint_num', action='store', dest='checkpoint_num', default=-1, type=int, nargs='?',
                        help='Checkpoint number to load')
    parser.add_argument('-set', action='store', dest='set_', default=None, type=str, nargs='?',
                        help='Which set to evaluate on "val", "selval" or "test"')
    args = parser.parse_args()

    # Path to the workspace directory

    if args.set_ is None:
        if args.mode == 'train':
            # train the network
            load_net(training_ws_path=args.ws_path, network_file=args.network_file, params_sub_dir=args.params_sub_dir,
                     exp_subdir=args.exp_subdir,
                     mode=args.mode, sets=['train', 'selval'], checkpoint_num=args.checkpoint_num)\
                .train()
        elif args.mode == 'eval':
            load_net(training_ws_path=args.ws_path, network_file=args.network_file, params_sub_dir=args.params_sub_dir,
                     exp_subdir=args.exp_subdir,
                     mode=args.mode, sets=['train', 'val'], checkpoint_num=args.checkpoint_num)\
                .evaluate()
        elif args.mode == 'traineval':
            load_net(training_ws_path=args.ws_path, network_file=args.network_file, params_sub_dir=args.params_sub_dir,
                     exp_subdir=args.exp_subdir,
                     mode='train', sets=['train'], checkpoint_num=args.checkpoint_num)\
                .train()
            load_net(training_ws_path=args.ws_path, network_file=args.network_file, params_sub_dir=args.params_sub_dir,
                     exp_subdir=args.exp_subdir,
                     mode='eval', sets=['val'], checkpoint_num=args.checkpoint_num)\
                .evaluate()
        elif args.mode == 'gen':
            load_net(training_ws_path=args.ws_path, network_file=args.network_file, params_sub_dir=args.params_sub_dir,
                     exp_subdir=args.exp_subdir,
                     mode='eval', sets=['obj'], checkpoint_num=args.checkpoint_num)\
                .generate()
        if args.mode == 'display':
            # train the network
            load_net(training_ws_path=args.ws_path, network_file=args.network_file, params_sub_dir=args.params_sub_dir,
                     exp_subdir=args.exp_subdir,
                     mode=args.mode, sets=['display'], checkpoint_num=args.checkpoint_num)\
                .display()
        elif args.mode == 'count_parameters':
            count_parameters(training_ws_path=args.ws_path, network_file=args.network_file,
                             params_sub_dir=args.params_sub_dir)

    else:
        my_trainer = load_net(training_ws_path=args.ws_path, network_file=args.network_file,
                              params_sub_dir=args.params_sub_dir, exp_subdir=args.exp_subdir,
                              mode=args.mode, sets=[args.set_], checkpoint_num=args.checkpoint_num)
        if args.mode == 'train':
            my_trainer.train()
        elif args.mode == 'eval':
            my_trainer.evaluate()
        elif args.mode == 'gen':
            my_trainer.generate()
        elif args.mode == 'count_parameters':
            assert args.set is None
