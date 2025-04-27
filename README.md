# 作业三——CLIP模型
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