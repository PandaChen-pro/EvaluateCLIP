# 复现CLIP模型在Flickr30k、MSCOCO以及Zero-shot 猫狗分类的结果

## 安装环境(需自行配置shell crash)
```shell
git clone https://github.com/openai/CLIP.git
conda create -n clip python=3.8
conda activate clip
pip install torch==1.7.1 torchvision  -i https://pypi.mirrors.ustc.edu.cn/simple/
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```
## 测试环境是否ok
```shell
python test.py
# 应输出[[0.9927937  0.00421068 0.00299572]]
```
## Flickr30k
2G显存足矣

可修改参数：
```python
# 三个模型 ViT-L/14@336px, ViT-L/14, ViT-B/32
model_name = 'ViT-L/14' 
```
执行命令：
```shell
conda activate clip
python Flickr30k.py
```

本次测试使用的Flickr30k测试数据集为：https://huggingface.co/datasets/royokong/flickr30k_test

文章中的结果如下(Table 13)：
| Metric | Text Retrieval (Zero-Shot CLIP on Flickr30k) | Image Retrieval (Zero-Shot CLIP on Flickr30k) |
|:-------:|:--------:|:-------:|
| R@1  |   88.0%  |   68.7%|
| R@5  |   98.7%  |   90.6%|
| R@10  |   99.4%  |   95.2%|

复现结果（ViT-B/32）：
| Metric | Text Retrieval (Zero-Shot CLIP on Flickr30k) | Image Retrieval (Zero-Shot CLIP on Flickr30k) |
|:-------:|:--------:|:-------:|
| R@1  |  81.20%  |   59.38%   |
| R@5  |  95.80%  |   83.62%   |
| R@10  |  98.80%  |   90.10%   |

rSum: 508.90

复现结果（ViT-L/14）：
| Metric | Text Retrieval (Zero-Shot CLIP on Flickr30k) | Image Retrieval (Zero-Shot CLIP on Flickr30k) |
|:-------:|:--------:|:-------:|
| R@1  |   86.40%  |   64.8%|
| R@5  |   97.7%  |   87.6%|
| R@10  |   99.20%  |   92.52%|

rSum: 528.22

复现结果（ViT-L/14@336px）：
| Metric | Text Retrieval (Zero-Shot CLIP on Flickr30k) | Image Retrieval (Zero-Shot CLIP on Flickr30k) |
|:-------:|:--------:|:-------:|
| R@1  |   87.20%  |   67.34%|
| R@5  |   98.90%  |   89.18%|
| R@10  |   99.60%  |   93.58%|

rSum: 535.80

## MSCOCO数据集

可修改参数：
```python
# 三个模型 ViT-L/14@336px, ViT-L/14, ViT-B/32
model_name = 'ViT-L/14' 
```
执行命令：
```shell
conda activate clip
python MSCOCO.py
```

本次测试使用的MSCOCO测试数据集为：https://huggingface.co/datasets/nlphuji/mscoco_2014_5k_test_image_text_retrieval

文章中的结果如下(Table 13)：
| Metric | Text Retrieval (Zero-Shot CLIP on Flickr30k) | Image Retrieval (Zero-Shot CLIP on Flickr30k) |
|:-------:|:--------:|:-------:|
| R@1    | 58.4%                               | 37.8%                               |
| R@5    | 81.5%                               | 62.4%                               |
| R@10   | 88.1%                               | 72.2%                               |

复现结果（ViT-B/32）：
| Metric | Text Retrieval (Zero-Shot CLIP on Flickr30k) | Image Retrieval (Zero-Shot CLIP on Flickr30k) |
|:-------:|:--------:|:-------:|
| R@1    | 51.62%                             | 30.68%                             |
| R@5    | 75.92%                             | 56.11%                             |
| R@10    | 84.64%                             | 67.11%                             |

rSum: 366.07

复现结果（ViT-L/14）：
| Metric | Text Retrieval (T2I / Text -> Img) | Image Retrieval (I2T / Img -> Text) |
|:-------:|:--------:|:-------:|
| R@1    | 57.02%                             | 36.38%                             |
| R@5    | 79.68%                             | 60.94%                             |
| R@10    | 87.22%                             | 71.00%                             |

rSum: 392.24


复现结果（ViT-L/14@336px）：
| Metric | Text Retrieval (T2I / Text -> Img) | Image Retrieval (I2T / Img -> Text) |
|:-------:|:--------:|:-------:|
| R@1    | 59.00%                             | 36.49%                             |
| R@5    | 82.10%                             | 61.03%                             |
| R@10    | 88.64%                             | 71.35%                             |

rSum: 398.61


## Zero-Shot猫狗分类

可修改参数：
```python
# 三个模型 ViT-L/14@336px, ViT-L/14, ViT-B/32
model_name = 'ViT-L/14' 
```
执行命令：
```shell
conda activate clip
python CatsVsDogs_Classification.py
```

| Metric | ViT-B/32 | ViT-L/14 | ViT-L/14@336px ｜
|:-------:|:--------:|:-------:|:-------:|
|正确预测数量|499|499|499|
|猫类别准确率|99.80%|99.80%|99.80%|
|狗类别准确率|99.80%|99.80%|99.80%|


|  | 预测 Cat | 预测 Dog | 
|:-------:|:--------:|:-------:|
|实际 Cat|250|0|
|实际 Dog|1|249|
