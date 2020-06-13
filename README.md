# NRS_pytorch
official pytorch implementation of Neural Random Subapce (NRS)

paper is availabel at [[arxiv]](https://arxiv.org/abs/1911.07845)

![](https://github.com/CupidJay/hello-world/blob/master/neural_network.jpg)

## Abstract
Random subspace is the pillar of random forests. We propose Neural Random Subspace (NRS), a novel deep learning based random subspace method. In contrast to previous forest methods, NRS enjoys the benefits of end-to-end, data-driven representation learning, as well as pervasive support from deep learning software and hardware platforms, hence achieving faster inference speed and higher accuracy. Furthermore, as a non-linear component to be encoded into Convolutional Neural Networks (CNNs), NRS learns non-linear feature representations in CNNs more efficiently than previous higher-order pooling methods, producing good results with negligible increase in parameters, floating point operations (FLOPs) and real running time. We achieve superior performance on 35 machine learning datasets when compared to random subspace, random forests and gradient boosting decision trees (GBDTs). Moreover, on both 2D image and 3D point cloud recognition tasks, integration of NRS with CNN architectures achieves consistent improvements with negligible extra cost. 

Keywords: random subspace, ensemble learning, deep neural networks

## Getting Started

### Prerequisites
* python 3
* PyTorch (> 1.0)
* torchvision (> 0.2)
* Numpy

### Train Examples
- Uci datasets: We use letter for example
```
CUDA_VISIBLE_DEVICES="0" python main.py \
--config-file configs/letter.yaml \
```
- CUB200: We used 4 GPUs to train CUB200. 
```
CUDA_VISIBLE_DEVICES="0,1,2,3" python main.py \
--config-file configs/cub200resnet.yaml
```

- More configurations
```
CUDA_VISIBLE_DEVICES="0" python main.py \
--config-file configs/letter.yaml \
MODEL.META_ARCHITECTURE UciFCNet \ #if you want to use MLP, else default
MODEL.N_MUL 50 \ #if you want to use different values for nMul, the same for other hyper-parameters, e.g., nPer
SOLVER.NUM_EPOCHS 50 \ # if you want to specify the total epochs, the same for other settings, e.g., batch-size
OUTPUT_DIR "Results/uci/letter" \ #the output directory for log file and model file
```

### Test Examples using Pretrained model
```
CUDA_VISIBLE_DEVICES="0,1,2,3" python main.py \
--config-file configs/cub200resnet.yaml \
MODEL.RESUME Results/ResNet_model_best.pth.tar \
TRAIN False 
```

## Citation
```
@article{NRS,
   title         = {Neural Random Subspace},
   author        = {Yun-Hao Cao and Jianxin Wu and Hanchen Wang and Joan Lasenby},
   year          = {2020},
   journal = {arXiv preprint arXiv:1911.07845}}
```
