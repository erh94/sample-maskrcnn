import os
import numpy as np
import torch
from PIL import Image, ImageDraw

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils

from _params import hparams
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F

from engine import train_one_epoch, evaluate
import utils
import transforms as T
import pandas as pd
from _params import hparams
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class MalariaDataset(object):
    def __init__(self,root,csvpath,transforms):
        self.root = root
        self.transforms = transforms
        self.csv_path = csvpath
        self.classes = {
            'ring':0,
            'trophozoite':1,
            'gametocyte':2,
            'schizont':3,
            'difficult':4,
            'leukocyte':5,
        }

    def __len__(self):
        return len(pd.read_csv(self.csv_path))


    def __getitem__(self,idx):
#         assert os.path.exists(self.csv_path),False

        df = pd.read_csv(self.csv_path)

        
        img_path = os.path.join(self.root, df['imagepath'][idx].strip())

        mask_path = os.path.join(self.root, df['maskpath'][idx].strip())
        #print(mask_path)
        ly = df['minimum_r'][idx]
        lx = df['minimum_c'][idx]
        uy = df['maximum_r'][idx]
        ux = df['maximum_c'][idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
#         mask =  torch.as_tensor(masks, dtype=torch.uint8)
#         np.array(mask)
        mask = F.to_tensor(mask)
        mask = mask.type(torch.uint8)


        bbox = [[lx,ly,ux,uy]]

        bbox = torch.as_tensor(bbox, dtype=torch.float32)
        label = torch.as_tensor([self.classes[df['type'][idx]]],dtype=torch.int64)
        area = (bbox[:, 3] - bbox[:, 1]) * (bbox[:, 2] - bbox[:, 0])

        target = {}
        target["boxes"] = bbox
        target["labels"] = label
        target["masks"] = mask
        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = torch.tensor([0])

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            img = F.to_tensor(img)
        

        return img, target








class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

        
