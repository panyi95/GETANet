
## Preparatinon
```shell
git clone https://github.com/zRamsey1234/IAWT-pytorch.git
cd droneRGBT && mkdir data
```
## Requirements
The code is tested on Ubuntu 20.04 with cuda 11.0
```shell
conda create -n py38 python=3.8
conda activate py38
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch
conda install h5py, numpy, opencv

```
## Data and Model Preparation
Download the DroneRGBT dataset from [official DroneRGBT](https://github.com/VisDrone/DroneRGBT)
or our processed one from [Baidu Cloud](https://pan.baidu.com/s/1bWvZkB7mrx4dDKsfeR2IBQ) [code: iawt] 


Download our model from  [Baidu Cloud](https://pan.baidu.com/s/1jmHkMaQ6QaphqSRukqwKRg )
[code: iawt] 

## Usage
```shell
mkdir data
# move both the dataset and model to data/
python test.py
```
