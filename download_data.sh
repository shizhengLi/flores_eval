#!/bin/bash

# 设置代理
export HTTP_PROXY="http://127.0.0.1:7890" 
export HTTPS_PROXY="http://127.0.0.1:7890"

# 激活conda环境
source activate flores_eval

echo "Downloading FLORES+ dataset..."

# 下载数据
python download_flores_data.py \
    --data_dir "data" \
    --splits "dev" "devtest" \
    --formats "json" "txt"

echo "Download completed!"
echo "Data saved in data/dev/ and data/devtest/ directories"
echo ""
echo "To view available languages:"
echo "python download_flores_data.py --list_languages"
echo ""
echo "To view dataset information:"
echo "python download_flores_data.py --info" 