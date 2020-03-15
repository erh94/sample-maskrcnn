# sample-maskrcnn

install

#### Requirements
Pytorch 1.4 Cuda 10

#### Conda environment
````sh
conda create --name maskrcnn -y
conda activate maskrcnn
conda install ipython pip
pip install ninja yacs cython matplotlib tqdm opencv-python
````

````sh
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
cd cocoapi/PythonAPI
make
cp -r pycocotools ../../

python maskrcnn-PennFudanPed.py
````

#### TO train on Malaria Dataset (Single point of interest images)
```` python 
python malaria-train.py --dataset malaria
````
# change the directories and path to csv files in _params.py
````python
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
````

Number of classes used for this training dataset is 6
````
        self.classes = {
            'ring':0,
            'trophozoite':1,
            'gametocyte':2,
            'schizont':3,
            'difficult':4,
            'leukocyte':5,
        }
````
to change the number of classes change the csv file and self.classes in dataset.py


