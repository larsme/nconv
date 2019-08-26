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
from modules.losses import ConfLossDecay, SmoothL1Loss, MSELoss

# Fix CUDNN error for non-contiguous inputs
import torch.backends.cudnn as cudnn
cudnn.enabled = True
cudnn.benchmark = True


def load_net(exp, params_sub_dir, mode='eval', sets=None, checkpoint_num=-1, training_ws_path='workspace'):
    exp_dir = os.path.join(BASE_DIR, training_ws_path)

    # Add the experiment's folder to python path

    # Read parameters file
    with open(os.path.join(exp_dir, params_sub_dir, 'params.json'), 'r') as fp:
        params = json.load(fp)

    # Use GPU or not
    device = torch.device("cuda:"+str(params['gpu_id']) if torch.cuda.is_available() else "cpu")

    # Dataloader for KittiDepth
    dataloaders, dataset_sizes = KittiDepthDataloader(params)

    # Import the network file
    f = importlib.import_module((training_ws_path+'.'+exp).replace("/", "."))
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
                                        workspace_dir=exp_dir, params_sub_dir=params_sub_dir,
                                        sets=sets, use_load_checkpoint=checkpoint_num)

    return mytrainer


def predict(mytrainer, inputs_d, inputs_rgb):
    return mytrainer.return_one_prediction(inputs_d, inputs_rgb)


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', action='store', dest='mode', help='"eval" or "train" mode')
    parser.add_argument('-exp', action='store', dest='exp', help='Python file in workspace directory')
    parser.add_argument('-params_sub_dir', action='store', dest='params_sub_dir',
                        help='Params file in workspace directory')
    parser.add_argument('-ws_path', action='store', dest='ws_path', help='Workspace directory')
    parser.add_argument('-checkpoint_num', action='store', dest='checkpoint_num', default=-1, type=int, nargs='?',
                        help='Checkpoint number to load')
    parser.add_argument('-set', action='store', dest='set_', default=None, type=str, nargs='?',
                        help='Which set to evaluate on "val", "selval" or "test"')
    args = parser.parse_args()

    # Path to the workspace directory

    if args.set_ is None:
        if args.mode == 'train':
            # train the network
            load_net(args.exp, args.params_sub_dir, args.mode, ['train', 'selval'], args.checkpoint_num, args.ws_path)\
                .train()
        elif args.mode == 'eval':
            load_net(args.exp, args.params_sub_dir, args.mode, ['train', 'val'], args.checkpoint_num, args.ws_path)\
                .evaluate()
        elif args.mode == 'traineval':
            load_net(args.exp, args.params_sub_dir, 'train', ['train'], args.checkpoint_num, args.ws_path)\
                .train()
            load_net(args.exp, args.params_sub_dir, 'eval', ['val'], args.checkpoint_num, args.ws_path)\
                .evaluate()
        elif args.mode == 'gen':
            load_net(args.exp, args.params_sub_dir, 'eval', ['obj'], args.checkpoint_num, args.ws_path)\
                .generate()
    else:
        my_trainer = load_net(args.exp, args.params_sub_dir, args.mode, [args.set_], args.checkpoint_num, args.ws_path)
        if args.mode == 'train':
            my_trainer.train()
        elif args.mode == 'eval':
            my_trainer.evaluate()
        elif args.mode == 'gen':
            my_trainer.generate()
