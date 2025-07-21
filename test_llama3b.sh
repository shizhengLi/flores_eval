#!/bin/bash

# 设置代理
export HTTP_PROXY="http://127.0.0.1:7890" 
export HTTPS_PROXY="http://127.0.0.1:7890"

# 激活conda环境
source activate flores_eval

# 测试Llama 3.2 3B模型
echo "Testing Llama 3.2 3B model on FLORES+ dataset..."

# 运行评估脚本
python evaluate_model.py \
    --model_name "meta-llama/Llama-3.2-3B" \
    --model_type "causal" \
    --device "auto" \
    --max_length 512 \
    --batch_size 4 \
    --data_dir "data" \
    --split "devtest" \
    --max_samples 5 \
    --source_lang "eng_Latn" \
    --target_langs "spa_Latn" "fra_Latn" "deu_Latn" "ita_Latn" "por_Latn" \
    --output "results/results_llama3b.json" \
    --use_hf_dataset \
    --save_data_locally \
    --data_format "json"

echo "Evaluation completed! Results saved to results/results_llama3b.json"
#echo "FLORES+ data saved locally in data/devtest/ directory" 