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
import json



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='image file path')
    parser.add_argument('checkoint', type=str, help='Checkpoint full path with network', default='vgg16_image_classifier.pth')
    parser.add_argument('--top_k', help = 'Return top K most likely classes', required=False, default=3)
    parser.add_argument('--category_names', help = 'Use a mapping of categories to real names:', required=False, default='cat_to_name.json')
    parser.add_argument('--gpu', dest = 'gpu', action = 'store_true', help = 'Enable GPU support')
    args = parser.parse_args()

    print("command args: ", args)
    device = icc.get_device(args.gpu)
    print("device is  ", device)
    model = icc.load_check_point(args.checkoint)
    print("model is :", model)
    
    with open(args.category_names, 'r') as f:
        cat_to_name_dict = json.load(f)
    
    probabilities, classes = icc.predict(device, args.image_path, model, args.top_k)
    print(args.image_path, " classifications: ") 
    for c in classes[0]:
        print(c.item(), " : " , cat_to_name_dict[str(c.item())])
        
    
if __name__ == "__main__":
    main()