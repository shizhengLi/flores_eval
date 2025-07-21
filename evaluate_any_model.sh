#!/bin/bash

# 设置代理
export HTTP_PROXY="http://127.0.0.1:7890" 
export HTTPS_PROXY="http://127.0.0.1:7890"

# 检查参数
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_name> [model_type] [max_samples] [save_data_locally]"
    echo "Example: $0 meta-llama/Llama-3.2-3B causal 100 true"
    echo "Example: $0 facebook/nllb-200-distilled-600M seq2seq 50 false"
    exit 1
fi

MODEL_NAME=$1
MODEL_TYPE=${2:-"causal"}  # 默认为causal
MAX_SAMPLES=${3:-100}      # 默认为100个样本
SAVE_DATA=${4:-"true"}     # 默认保存数据到本地

# 从模型名称生成输出文件名
MODEL_SHORT_NAME=$(echo $MODEL_NAME | sed 's/[^a-zA-Z0-9]/_/g')
OUTPUT_FILE="results_${MODEL_SHORT_NAME}.json"

# 激活conda环境
source activate flores_eval

echo "Testing model: $MODEL_NAME"
echo "Model type: $MODEL_TYPE"
echo "Max samples: $MAX_SAMPLES"
echo "Save data locally: $SAVE_DATA"
echo "Output file: $OUTPUT_FILE"

# 运行评估脚本
if [ "$SAVE_DATA" = "true" ]; then
    python evaluate_model.py \
        --model_name "$MODEL_NAME" \
        --model_type "$MODEL_TYPE" \
        --device "auto" \
        --max_length 512 \
        --batch_size 4 \
        --data_dir "data" \
        --split "devtest" \
        --max_samples $MAX_SAMPLES \
        --source_lang "eng_Latn" \
        --target_langs "spa_Latn" "fra_Latn" "deu_Latn" "ita_Latn" "por_Latn" \
        --output "$OUTPUT_FILE" \
        --use_hf_dataset \
        --save_data_locally \
        --data_format "json"
else
    python evaluate_model.py \
        --model_name "$MODEL_NAME" \
        --model_type "$MODEL_TYPE" \
        --device "auto" \
        --max_length 512 \
        --batch_size 4 \
        --data_dir "data" \
        --split "devtest" \
        --max_samples $MAX_SAMPLES \
        --source_lang "eng_Latn" \
        --target_langs "spa_Latn" "fra_Latn" "deu_Latn" "ita_Latn" "por_Latn" \
        --output "$OUTPUT_FILE" \
        --use_hf_dataset
fi

echo "Evaluation completed! Results saved to $OUTPUT_FILE"
if [ "$SAVE_DATA" = "true" ]; then
    echo "FLORES+ data saved locally in data/devtest/ directory"
fi 