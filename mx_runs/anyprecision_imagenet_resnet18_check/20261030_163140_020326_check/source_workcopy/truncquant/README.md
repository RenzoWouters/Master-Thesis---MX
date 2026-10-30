# TruncQuant
Official code and models in PyTorch for the paper [TruncQuant: Accurate Truncation-Ready Quantization for Deep Neural Networks with Flexible Weight Bit Precision](#).

## Run
#### Environment
* Python 3.7
* PyTorch 1.1.0
* torchvision 0.2.1
* [tensorboardX](https://github.com/lanpa/tensorboardX)
* [gpustat](https://github.com/wookayin/gpustat)

## Getting started 
- (update) 	**Available for any model**: `models/model_quan.py` enables converting any model into a quantized version for any-precision training.
- **Train ResNet-20 on CIFAR-10**: Run `./train_cifar10.sh` (dataset downloads automatically).
- **Train ResNet-18/50 on ImageNet**: Manually download ImageNet, set data_paths in `dataset/data.py`, then run `./train_imagenet.sh`.
- **Key scripts**: `train.py` (ResNet-20 BatchNorm calibration), `trunc_train.py` (ResNet-20 testing/training), `train_imagenet.py` (ResNet-18/50 BatchNorm calibration), `test_imagenet.py` (ResNet-18/50 testing/training).
- **Training details**: Initial LR 0.1 → 0.5 for any-precision models.


## Train
##### Resnet-20 Models on CIFAR10
Run the script below and dataset will download automatically.
 ```
 ./train_cifar10.sh
 ```

##### Resnet-18, Resnet-50 Models on ImageNet
Before running the script below, one needs to manually download ImageNet and save it properly according to `data_paths` in `dataset/data.py`.
 ```
 ./train_imagenet.sh
 ```

## Implementation Details 
- `train.py`: Used for calibrating ResNet-20 BatchNorm Layers. 
- `trunc_train.py`: Used for testing ResNet-20 models. Also used for training ResNet-20 from scratch. 
- `train_imagenet.py`: Used for calibrating ResNet-18,50 BatchNorm Layers.
- `test_imagenet.py`: Used for testing ResNet-18,50 models. Also used for training ResNet-18,50 from scratch. 


## Trained Models
* Init lr for any-precision models: 0.1 -> 0.5.
* We use ReLU for 32-bit model instead of Clamp (check [here](https://github.com/haichaoyu/any-precision-nets/blob/master/models/resnet_quan.py#L22)).
* We use tanh nonlinearity for 32-bit model for consistency with other precisions (check [here](https://github.com/haichaoyu/any-precision-nets/blob/master/models/quan_ops.py#L88)).


##### Resnet-20(expansion 1) on CIFAR10
|                                                      Bit |   1   |   2   |   4   |   8   |  32   |
|---------------------------------------------------------:|:-----:|:-----:|:-----:|:-----:|:-----:|
|                          (train) Quant (inference) Quant | 72.5 $\pm$ 1.4 | 86.0 $\pm$ 0.2 | 87.5 $\pm$ 0.2 | 87.7 $\pm$ 0.3 | 87.7 $\pm$ 0.3 |
|                          (train) Quant (inference) Trunc | 72.6 $\pm$ 1.4 | 57.5 $\pm$ 6.9 | 84.9 $\pm$ 0.5 | 87.7 $\pm$ 0.3 | 87.7 $\pm$ 0.3 |
|                     (train) TruncQuant (inference) Trunc | 72.1 $\pm$ 1.2 | 85.8 $\pm$ 0.3 | 87.6 $\pm$ 0.3 | 87.8 $\pm$ 0.3 | 87.8 $\pm$ 0.3 |



##### MobileNetV2 on CIFAR10
|                                                      Bit |   1   |   2   |   4   |   8   |  32   |
|---------------------------------------------------------:|:-----:|:-----:|:-----:|:-----:|:-----:|
|                          (train) Quant (inference) Quant | 83.4 $\pm$ 0.2 | 90.6 $\pm$ 0.2 | 91.5 $\pm$ 0.2 | 91.5 $\pm$ 0.2 | 91.7 $\pm$ 0.2 |
|                          (train) Quant (inference) Trunc | 83.4 $\pm$ 0.2 | 53.8 $\pm$ 4.7 | 87.1 $\pm$ 0.5 | 91.5 $\pm$ 0.2 | 91.7 $\pm$ 0.2 |
|                     (train) TruncQuant (inference) Trunc | 83.5 $\pm$ 0.5 | 90.5 $\pm$ 0.2 | 91.4 $\pm$ 0.1 | 91.5 $\pm$ 0.2 | 91.6 $\pm$ 0.1 |



##### AlexNet on CIFAR10
|                                                      Bit |   1   |   2   |   4   |   8   |  32   |
|---------------------------------------------------------:|:-----:|:-----:|:-----:|:-----:|:-----:|
|                          (train) Quant (inference) Quant | 84.6 $\pm$ 0.4 | 86.5 $\pm$ 0.4 | 86.7 $\pm$ 0.4 | 86.8 $\pm$ 0.3 | 86.7 $\pm$ 0.3 |
|                          (train) Quant (inference) Trunc | 84.6 $\pm$ 0.4 | 82.9 $\pm$ 0.5 | 86.5 $\pm$ 0.3 | 86.8 $\pm$ 0.3 | 86.7 $\pm$ 0.3 |
|                     (train) TruncQuant (inference) Trunc | 84.7 $\pm$ 0.5 | 86.7 $\pm$ 0.2 | 86.9 $\pm$ 0.3 | 86.9 $\pm$ 0.3 | 86.8 $\pm$ 0.4 |



##### ResNet-8 on SVHN
|                                                      Bit |   1   |   2   |   4   |   8   |  32   |
|---------------------------------------------------------:|:-----:|:-----:|:-----:|:-----:|:-----:|
|                          (train) Quant (inference) Quant | 91.4 $\pm$ 0.2 | 95.6 $\pm$ 0.1 | 95.7 $\pm$ 0.1 | 95.7 $\pm$ 0.1 | 95.6 $\pm$ 0.1 |
|                          (train) Quant (inference) Trunc | 91.5 $\pm$ 0.2 | 87.4 $\pm$ 0.4 | 95.1 $\pm$ 0.3 | 95.7 $\pm$ 0.1 | 95.6 $\pm$ 0.1 |
|                     (train) TruncQuant (inference) Trunc | 90.8 $\pm$ 0.9 | 95.6 $\pm$ 0.1 | 95.7 $\pm$ 0.1 | 95.6 $\pm$ 0.1 | 95.5 $\pm$ 0.1 |



##### CNN-8 on SVHN
|                                                      Bit |   1   |   2   |   4   |   8   |  32   |
|---------------------------------------------------------:|:-----:|:-----:|:-----:|:-----:|:-----:|
|                          (train) Quant (inference) Quant | 87.5 $\pm$ 0.6 | 94.9 $\pm$ 0.1 | 96.3 $\pm$ 0.1 | 96.3 $\pm$ 0.1 | 96.3 $\pm$ 0.1 |
|                          (train) Quant (inference) Trunc | 87.5 $\pm$ 0.5 | 93.8 $\pm$ 0.6 | 96.1 $\pm$ 0.1 | 96.3 $\pm$ 0.1 | 96.3 $\pm$ 0.1 |
|                     (train) TruncQuant (inference) Trunc | 86.8 $\pm$ 1.1 | 94.9 $\pm$ 0.1 | 96.2 $\pm$ 0.0 | 96.3 $\pm$ 0.1 | 96.3 $\pm$ 0.1 |



##### Resnet-18 on ImageNet
|                                Bit |   1   |   2   |   3   |   4   |   5   |   6   |   7   |   8   |
|-----------------------------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|          (train) Quant (inference) Quant | 37.3 | 62.0 | 62.8 | 64.4 | 63.1 | 63.1 | 63.1 | 64.5 |
|          (train) Quant (inference) Trunc | 37.1 | 11.8 | 48.6 | 59.1 | 62.2 | 63.0 | 63.1 | 64.5 |
|     (train) TruncQuant (inference) Trunc | 39.1 | 61.6 | 62.6 | 64.2 | 63.2 | 63.2 | 63.1 | 64.6 |



##### Resnet-50(expansion 4) on ImageNet
|                                      Bit |   1   |   2   |   3   |   4   |   5   |   6   |   7   |   8   |
|-----------------------------------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|          (train) Quant (inference) Quant | 58.7 | 71.7 | 72.0 | 73.8 | 72.3 | 72.2 | 72.2 | 74.1 |
|          (train) Quant (inference) Trunc | 58.7 |  3.9 | 36.8 | 68.6 | 70.9 | 71.9 | 72.1 | 74.0 |
|     (train) TruncQuant (inference) Trunc | 57.5 | 71.4 | 72.3 | 73.9 | 72.4 | 72.4 | 72.4 | 74.0 |


## Reference
- [Any-Precision Deep Neural Network](https://github.com/SHI-Labs/Any-Precision-DNNs)


## Citation
If you find our study helpful, please cite our paper:
```
@article{kim2025truncquant,
  title={TruncQuant: Truncation-Ready Quantization for DNNs with Flexible Weight Bit Precision},
  author={Kim, Jinhee and Yoon, Seoyeon and Lee, Taeho and Lee, Joo Chan and Jeon, Kang Eun and Ko, Jong Hwan},
  journal={arXiv preprint arXiv:2506.11431},
  year={2025}
}
```
