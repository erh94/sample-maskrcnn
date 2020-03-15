import os
import sys
import torch
import time

def get_timestamp():
    return time.strftime("%d%b%Y%H%M",time.localtime())


class DefaultParams():
    def __init__(self):
        self.cuda =  True if torch.cuda.is_available() else False

        self.parallel = False

        self.device = torch.device("cuda:0" if self.cuda else "cpu")

        self.device_ids = [0,1,2,3]

        self.distributed = False


        self.pretrained  = False 
        
        self.dataset_root = 'PennFudanPed-pretrained'

        self.train_dir = '/home1/SharedFolder/dataset/train/'
        self.train_csv = '/home1/SharedFolder/dataset/train/old_csvs/train_modified.csv'

        self.test_dir = '/home1/SharedFolder/dataset/resize/'
        self.test_csv = '/home1/SharedFolder/dataset/resize/resized_test.csv'

        self.timestamp = get_timestamp()

        self.exp_name = f'maskrcnn-{self.dataset_root}'
        
        self.model_dir = f'./models/{self.exp_name}/{self.timestamp}/'
        os.makedirs(self.model_dir, exist_ok=True)

        self.num_epochs = 11

        self.hidden_layer = 256

        self.num_classes = 6

hparams = DefaultParams()
