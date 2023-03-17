
from mmdet.apis import set_random_seed
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmcv import Config
import shutil
import xml.etree.ElementTree as ET
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.builder import DATASETS
import numpy as np
import os.path as osp
import copy
from mmdet.models import build_detector
from mmdet.apis import inference_detector
from mmcv.runner import load_checkpoint
import mmcv
import torch

CLASSES = ('Strawberry_3', 'Strawberry_2', 'Strawberry_1', 'Flower',
            'Green_small_fruit', 'Receptacle', 'Before_blooming')
            
NUM_CLASSES = 7
EVAL = False
DEVICE = "cpu"



# Choose to use a config and initialize the detector

# fast rcnn with swin
config = 'configs/strawberry/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py'
checkpoint = 'tutorial_exps/epoch_2.pth'

# mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py
# config = 'configs/strawberry/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py'
# checkpoint = 'checkpoints/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth'

# mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco
# config = 'configs/strawberry/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py'
# checkpoint = 'checkpoints/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_20210906_131725-bacf6f7b.pth'

# Set the device to be used for evaluation
# torch.
device = DEVICE

# Load the config
config = mmcv.Config.fromfile(config)
# Set pretrained to be None since we do not need pretrained model here
config.model.pretrained = None

# Initialize the detector
model = build_detector(config.model)

# Load checkpoint
checkpoint = load_checkpoint(model, checkpoint, map_location=device)

# Set the classes of models for inference
model.CLASSES = NUM_CLASSES

# We need to set the model's cfg for inference
model.cfg = config

# modify num classes of the model in box head
model.roi_head.bbox_head.num_classes = NUM_CLASSES
# if "mask_head" in model.roi_head:
#     model.roi_head.mask_head.num_classes = NUM_CLASSES

# Convert the model to GPU
model.to(device)
# Convert the model into evaluation mode
if EVAL:
    model.eval()

# Use the detector to do inference
with open("dataset/test.txt", "r") as f:
    while f:
        img = f.readline().rstrip()
        result = inference_detector(model, "dataset/image/"+img+".png")
        print(img, len(result), result)
        
    
# Let's plot the result
# model.show_result(img, result, out_file="./dataset/result.jpg")
