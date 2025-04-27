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
```
## Flickr30k
2G显存足矣

本次测试使用的Flickr30k测试数据集为：https://huggingface.co/datasets/royokong/flickr30k_test

文章中的结果如下(Table 13)：
|  | Text Retrieval (Zero-Shot CLIP on Flickr30k) | Image Retrieval (Zero-Shot CLIP on Flickr30k) |
|:-------:|:--------:|:-------:|
| R@1  |   88.0%  |   68.7%|
| R@5  |   98.7%  |   90.6%|
| R@10  |   99.4%  |   95.2%|

复现结果（ViT-B/32）：
|  | Text Retrieval (Zero-Shot CLIP on Flickr30k) | Image Retrieval (Zero-Shot CLIP on Flickr30k) |
|:-------:|:--------:|:-------:|
| R@1  |  81.20%  |   59.38%   |
| R@5  |  95.80%  |   83.62%   |
| R@10  |  98.80%  |   90.10%   |

rSum: 508.90

复现结果（ViT-L/14）：
|  | Text Retrieval (Zero-Shot CLIP on Flickr30k) | Image Retrieval (Zero-Shot CLIP on Flickr30k) |
|:-------:|:--------:|:-------:|
| R@1  |   86.40%  |   64.8%|
| R@5  |   97.7%  |   87.6%|
| R@10  |   99.20%  |   92.52%|

rSum: 528.22

复现结果（ViT-L/14@336px）：
|  | Text Retrieval (Zero-Shot CLIP on Flickr30k) | Image Retrieval (Zero-Shot CLIP on Flickr30k) |
|:-------:|:--------:|:-------:|
| R@1  |   87.20%  |   67.34%|
| R@5  |   98.90%  |   89.18%|
| R@10  |   99.60%  |   93.58%|

rSum: 535.80

## MSCOCO数据集
文章中的结果如下(Table 13)：
|  | Text Retrieval (Zero-Shot CLIP on Flickr30k) | Image Retrieval (Zero-Shot CLIP on Flickr30k) |
|:-------:|:--------:|:-------:|
| R@1  |   58.4%  |   37.8%|
| R@5  |   81.5%  |   62.4%|
| R@10  |   88.1%  |   72.2%|

复现结果：
|  | Text Retrieval (Zero-Shot CLIP on Flickr30k) | Image Retrieval (Zero-Shot CLIP on Flickr30k) |
|:-------:|:--------:|:-------:|
| R@1  |   58.4%  |   37.8%|
| R@5  |   81.5%  |   62.4%|
| R@10  |   88.1%  |   72.2%|