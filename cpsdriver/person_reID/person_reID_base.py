# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from person_reID.model import ft_net, ft_net_dense, PCB, PCB_test

#fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='../Market/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

opt = parser.parse_args()
###load config###
# load the training config
config_path = os.path.join('./person_reID/model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream)
opt.fp16 = config['fp16'] 
opt.PCB = config['PCB']
opt.use_dense = config['use_dense']
opt.stride = config['stride']

if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else: 
    opt.nclasses = 751 

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./person_reID/model',name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def preprocess_img_for_yolo(img, img_size=416):
    input_img, _ = pad_to_square(img, 127.5)
    # Resize
    input_img = cv2.resize(
        input_img, (img_size, img_size), interpolation=cv2.INTER_AREA
    )
    # Channels-first
    input_img = np.transpose(input_img, (2, 0, 1))

    # extend one dimension
    input_img = np.expand_dims(input_img, axis=0)

    # As pytorch tensor
    input_img = torch.from_numpy(input_img).float() / 255.0
    return input_img

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

# gallery_path = image_datasets['gallery'].imgs
# query_path = image_datasets['query'].imgs

# gallery_cam,gallery_label = get_id(gallery_path)
# query_cam,query_label = get_id(query_path)

# if opt.multi:
#     mquery_path = image_datasets['multi-query'].imgs
#     mquery_cam,mquery_label = get_id(mquery_path)




######################################################################
# Load Collected data Trained model
# print('-------test-----------')

model_structure = ft_net(opt.nclasses, stride = opt.stride)

model = load_network(model_structure)

model.classifier.classifier = nn.Sequential()

# Change to test mode
model = model.eval().cuda()
# if use_gpu:
#     model = model.cuda()

# # Extract feature
# with torch.no_grad():
#     gallery_feature = extract_feature(model,dataloaders['gallery'])
#     query_feature = extract_feature(model,dataloaders['query'])
#     if opt.multi:
#         mquery_feature = extract_feature(model,dataloaders['multi-query'])
    
# # Save to Matlab for check
# result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
# scipy.io.savemat('pytorch_result.mat',result)

# print(opt.name)
# result = './model/%s/result.txt'%opt.name
# os.system('python evaluate_gpu.py | tee -a %s'%result)

# if opt.multi:
#     result = {'mquery_f':mquery_feature.numpy(),'mquery_label':mquery_label,'mquery_cam':mquery_cam}
#     scipy.io.savemat('multi_query.mat',result)

def extract_person_feature(img):
    n, c, h, w = img.size()
    ff = torch.FloatTensor(1,512).zero_().cuda()

    input_img = Variable(img.cuda())
    outputs = model(input_img) 
    ff += outputs
    
    # norm feature
    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))

    feature = ff.cpu().detach().numpy()
    return feature