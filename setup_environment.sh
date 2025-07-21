#!/bin/bash

# 设置代理
export HTTP_PROXY="http://127.0.0.1:7890" 
export HTTPS_PROXY="http://127.0.0.1:7890"

# echo "Setting up FLORES evaluation environment..."

# # 创建conda环境
# echo "Creating conda environment..."
# conda env create -f environment.yml

# # 激活环境
# echo "Activating conda environment..."
# source activate flores_eval

# Clone FLORES仓库
echo "Cloning FLORES repository..."
git clone https://github.com/facebookresearch/flores.git

# Clone fairseq仓库
echo "Cloning fairseq repository..."
git clone https://github.com/facebookresearch/fairseq.git
cd fairseq
git checkout nllb
pip install -e .
cd ..

# 下载FLORES-200数据集
echo "Downloading FLORES-200 dataset..."
bash download_data.sh

echo "Environment setup completed!"
echo "To activate the environment, run: conda activate flores_eval" 