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

        self.device = torch.device("cuda:4" if self.cuda else "cpu")

        self.device_ids = [4,5,6,7,1,2,3]

        self.pretrained  = True
        
        self.dataset_root = 'PennFudanPed'

        self.timestamp = get_timestamp()

        self.exp_name = f'temp-maskrcnn-{self.dataset_root}'
        
        self.model_dir = f'./models/{self.exp_name}/{self.timestamp}/'
        os.makedirs(self.model_dir, exist_ok=True)

        self.model = self.model_dir + 'model'

        self.num_epochs = 20

hparams = DefaultParams()
