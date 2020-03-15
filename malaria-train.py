import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T

from models import get_model_instance_segmentation
from dataset import MalariaDataset, PennFudanDataset

from _params import hparams
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from tensorboardX import SummaryWriter
import argparse

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


    
parser = argparse.ArgumentParser(description="MaskRCNN Implementation")
parser.add_argument('--dataset')
args = parser.parse_args()





def main():
    global hparams, args
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device(hparams.device) if torch.cuda.is_available() else torch.device('cpu')

    if(args.dataset == 'malaria'):
        hparams.dataset_root = 'malaria'
        hparams.exp_name = f'maskrcnn-{hparams.dataset_root}'
        dataset = MalariaDataset(hparams.train_dir,hparams.train_csv, get_transform(train=True))
        dataset_test = MalariaDataset(hparams.test_dir,hparams.test_csv,get_transform(False))
    else:
        dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
        dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))
        hparams.num_classes = 2

    writer = SummaryWriter(f'runs/{hparams.exp_name}_{hparams.timestamp}')

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(hparams)

    # move model to the right device
    model.to(device)

    model_without_ddp = model
    if hparams.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[hparams.device_ids])
        model_without_ddp = model.module

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = hparams.num_epochs

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10,writer=writer)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

        torch.save({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch},
                os.path.join(hparams.model_dir, 'model_{}.pth'.format(epoch)))

    print("That's it!")
    
if __name__ == "__main__":
    main()
