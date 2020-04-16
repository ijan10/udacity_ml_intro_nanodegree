import argparse
# Imports here
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

#import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from collections import OrderedDict
from datetime import datetime
from PIL import Image
import numpy as np
import image_classifier_common as icc
import os




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help = 'Directory with flower images is required')
    parser.add_argument('--save_dir', help = 'Set directory to save checkpoints', required=False, default="")
    parser.add_argument('--arch', type=str, dest='arch', default='vgg16', choices=['vgg16'], required=False ,help='application tested with vgg16 only')
    parser.add_argument('--learning_rate', help ='The learning rate is a hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated', required=False, default=0.001)
    parser.add_argument('--hidden_unit', help = 'size of hidden unit - assume only one hidden layer', required=False, default=4096)
    parser.add_argument('--epochs', help = 'number of forward and backward pass for all training samples.', required=False, default=3)
    parser.add_argument('--gpu', dest = 'gpu', action = 'store_true', help = 'Enable GPU support')
    args = parser.parse_args()

    print("command args: ", args)
    arch='vgg16'
    if args.arch is not None:
        arch = args.arch
    
   
   
    
    print(arch)
    device = icc.get_device(args.gpu)
    print("device is  ", device)
    checkpoint_path=""
    if args.save_dir is not None:
        checkpoint_path = args.save_dir
    
    checkpoint_file_name = os.path.join(checkpoint_path,args.arch+"_image_classifier.pth")
    print("start saving {}".format(checkpoint_file_name))
    trainloader = icc.get_train_loader(root_data_dir=args.data_dir)
    validloader = icc.get_valid_loader(root_data_dir=args.data_dir)
    testloader = icc.get_test_loader(root_data_dir=args.data_dir)
    model, criterion, optimizer = icc.get_model(arch, device, args.hidden_unit, args.hidden_unit, args.learning_rate)
    print(model)
    model = icc.train_model(device,  model, criterion, optimizer , trainloader, validloader, args.epochs)
    icc.test_network(device, model, testloader)
    
    checkpoint_path=""
    if args.save_dir is not None:
        checkpoint_path = args.save_dir
    
    checkpoint_file_name = os.path.join(checkpoint_path,args.arch+"_image_classifier.pth")
    print("start saving {}".format(checkpoint_file_name))
    train_dataset =  icc.get_data_set(args.data_dir, "train", icc.get_train_transforms()) 
    icc.save_checkpoint(arch, model, train_dataset, checkpoint_file_name)
    
    
if __name__ == "__main__":
    main()