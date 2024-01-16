# DCPA

Official implementation of *Dual-Decoder Consistency via Pseudo-Labels Guided Data Augmentation for Semi-Supervised Medical Image Segmentation*  

**Authors:**  

> Yuanbin Chen, Tao Wang, Hui Tang, Longxuan Zhao, Ruige Zong, Shun Chen, Tao Tan, Xinlin Zhang, Tong Tong

manuscript link:  

- https://arxiv.org/abs/2308.16573 (preprint on arXiv)  

This repo contains the implementation of the proposed *Dual-Decoder Consistency via Pseudo-Labels Guided Data Augmentation (DCPA)* on three public benchmarks in medical images.  
**If you find our work useful, please cite the paper:**  

> @article{chen2023dual,  
> title={Dual-Decoder Consistency via Pseudo-Labels Guided Data Augmentation for Semi-Supervised Medical Image Segmentation},  
> author={Chen, Yuanbin and Wang, Tao and Tang, Hui and Zhao, Longxuan and Zong, Ruige and Tan, Tao and Zhang, Xinlin and Tong, Tong},  
> journal={arXiv preprint arXiv:2308.16573},  
> year={2023}  
> }

## Requirements
This repository is based on PyTorch 1.12.1, CUDA 11.6 and Python 3.8; All experiments in our paper were conducted on a single NVIDIA GeForce RTX 3090 24GB GPU.

## Data 

Following previous works, we have validated our method on three benchmark datasets, including 2018 Atrial Segmentation Challenge, Pancreas-CT dataset and Automated Cardiac Diagnosis Challenge.  
It should be noted that we do not have permissions to redistribute the data. Thus, for those who are interested, please follow the instructions below and process the data, or you will get a mismatching result compared with ours.

### Data Preparation

#### Download

Atrial Segmentation: http://atriaseg2018.cardiacatlas.org/  
Pancreas-CT dataset: https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT  
Automated Cardiac Diagnosis: https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html

#### Data Split

We split the data following previous works. Detailed split could be found in folder `data`, which are stored in .list files.

#### Data Preprocessing

Download the data from the url above, then run the script `./code/dataloaders/la_heart_processing.py`, `./code/dataloaders/acdc_data_processing.py` and `./data/Pancreas/Pre_processing.ipynb` by passing the arguments of data location.

### Prepare Your Own Data

Our DCPA could be extended to other datasets with some modifications.  

## Usage
1. Clone the repo.;
```
git clone https://github.com/BinYCn/DCPA.git
```
2. Put the data in `./DCPA/data`;

3. Train the model;
```
cd DCPA
# e.g., for 20% labels on LA
python ./code/train_3d.py --dataset_name LA --labelnum 16 --gpu 0
```
4. Test the model;
```
cd DCPA
# e.g., for 20% labels on LA
python ./code/test_3d.py --dataset_name LA --exp DCPA3d --labelnum 16 --gpu 0
```

## Acknowledgements:
Our code is adapted from [UAMT](https://github.com/yulequan/UA-MT), [SASSNet](https://github.com/kleinzcy/SASSnet), [DTC](https://github.com/HiLab-git/DTC), [URPC](https://github.com/HiLab-git/SSL4MIS), [SSL4MIS](https://github.com/HiLab-git/SSL4MIS) and [MC-Net](https://github.com/ycwu1997/MC-Net.git). Thanks for these authors for their valuable works and hope our model can promote the relevant research as well.

## Questions
If any questions, feel free to contact me at 'binycn904363330@gmail.com'
