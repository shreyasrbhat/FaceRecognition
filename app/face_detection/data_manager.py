import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import glob
import random
import itertools
import re

from parse_config import parse_model_config
from models import Darknet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def img_transform(img_path, boxes):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    square_dim = max(w,h)
    pad_len = abs(w-h)//2
    pad_dim = (0,0, 0,0, pad_len, pad_len) if w >= h else (0,0, pad_len,pad_len, 0,0)
    img = F.pad(torch.from_numpy(img), pad_dim, )
    shift_x, shift_y = (0, pad_len) if w>=h else (pad_len, 0)
    boxes = torch.IntTensor(boxes)
    boxes[..., :2] += torch.IntTensor([shift_x, shift_y])
    boxes[:, :2] += (boxes[:, 2:]//2)
    boxes = boxes/float(img.shape[0])
    img = F.interpolate(img.permute(2,0,1).unsqueeze(0), size=416, mode='nearest').float().squeeze(0)
    return img, boxes

def draw_boxes(img, box_norm):
    h, w = img.shape[:2]
    square_dim = max(w,h)
    pad_len = abs(w-h)//2
    boxes = (box_norm * w).astype(int)
    shift_x, shift_y = (0, pad_len) if w>=h else (pad_len, 0)
    print(pad_len)
    boxes[:, :2] -= [shift_x, shift_y]
    boxes[:, :2] -= (boxes[:, 2:]//2)
    for x1, y1, w, h in boxes:
        cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (255, 0, 0), 2)
    plt.imshow(img)

class FaceTrainSet(Dataset):
    def __init__(self, data_path, labels_path, transforms=None):
        super(FaceTrainSet, self).__init__()
        self.data_path = data_path
        self.transforms = transforms
        self.labels_path = labels_path

        self.labels = np.loadtxt(self.labels_path, dtype=str, delimiter='\n')
        self.file_names = list(filter(lambda x:x.endswith('.jpg'), self.labels))
        self.file_name_index = [i for i in range(len(self.labels)) if self.labels[i].endswith('.jpg')]
        
        self.batch_count = 0
    
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        idx = self.file_name_index[index]
        try:
            label_info = self.labels[idx: self.file_name_index[self.file_name_index.index(idx)+1]]
        except IndexError as e:
            label_info = self.labels[idx:]
        file_name = label_info[0]
        num_faces = label_info[1]
        box_dims = [np.array(values.split()).astype(np.int)[:4] for values in label_info[2:]]
        img_tensor, boxes = img_transform(self.data_path+'/'+file_name, box_dims)
        return (img_tensor, boxes, num_faces)
    
    def collate_fn(self, batch):
        img_tensors, targets, num_faces = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None] 
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            index = torch.ones(len(boxes), 1) * i
            targets[i] = torch.cat((index, boxes), 1)
        targets = torch.cat(targets, 0)
        img_tensors = torch.stack([img for img in img_tensors])
        self.batch_count += 1
        return img_tensors, targets, num_faces