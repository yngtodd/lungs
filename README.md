# lungXnet for Classification and Localization of Thoracic Diseases

This is a Python3 (Pytorch) reimplementation of [CheXNet](https://stanfordmlgroup.github.io/projects/chexnet/). The model takes a chest X-ray image as input and outputs the probability of each thoracic disease along with a likelihood map of pathologies.

<div align=center><img width="500" height="500" src="./img/00008473_011-3.png"/></div>

## Dataset

The [ChestX-ray14 dataset](http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf) comprises 112,120 frontal-view chest X-ray images of 30,805 unique patients with 14 disease labels. To evaluate the model, we randomly split the dataset into training (70%), validation (10%) and test (20%) sets, following the work in paper. Partitioned image names and corresponding labels are placed under the directory [labels](./ChestX-ray14/labels).

## Profiling

This branch is used for data parallel


