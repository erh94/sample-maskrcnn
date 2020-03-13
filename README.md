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
