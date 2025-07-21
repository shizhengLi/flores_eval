
# FLORES+ Model Evaluation Environment

这个环境用于在FLORES+数据集上评估多语言翻译模型的性能。FLORES+是FLORES-200的升级版本，现在托管在HuggingFace上。

## 环境设置


### 0. 克隆这个仓库：

```bash
git clone git@github.com:shizhengLi/flores_eval.git
cd flores_eval
```

### 1. 创建conda环境

```bash
# 设置代理（如果需要）
export HTTP_PROXY="http://127.0.0.1:7890" 
export HTTPS_PROXY="http://127.0.0.1:7890"

# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate flores_eval
```

### 2. 运行完整设置脚本

```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

这个脚本会自动：
- 克隆FLORES和fairseq仓库
- 安装fairseq（nllb分支）
- 下载FLORES+数据集

### 3. 下载FLORES+数据集 (可跳过，上面已经下载过)

```bash
# 快速下载数据
./download_data.sh

# 或者手动下载
python download_flores_data.py --data_dir "data" --splits "dev" "devtest" --formats "json" "txt"
```

## 使用方法

目前测试的正常的代码：

```bash

python evaluate_model.py --model_name "meta-llama/Llama-3.2-3B" --model_type causal --max_samples 5 --source_lang "eng_Latn" --target_langs "spa_Latn" --output "results/test_results.json" --use_hf_dataset --save_data_locally --data_format json
```

```bash
./test_llama3b.sh
```

#### 请注意：
- 输出都放在results目录下
- 脚本最好放在script目录下
_ 这样不会push到github仓库


### 方法1：使用通用评估脚本

```bash
# 测试Llama 3.2 3B模型（自动下载并保存数据）
./evaluate_any_model.sh "meta-llama/Llama-3.2-3B" causal 100 true

# 测试NLLB模型（不保存数据）
./evaluate_any_model.sh "facebook/nllb-200-distilled-600M" seq2seq 50 false

# 测试本地模型
./evaluate_any_model.sh "/path/to/your/local/model" causal 200 true
```

### 方法2：直接使用Python脚本

```bash
# 使用FLORES+数据集进行评估
python evaluate_model.py \
    --model_name "meta-llama/Llama-3.2-3B" \
    --model_type "causal" \
    --max_samples 100 \
    --source_lang "eng_Latn" \
    --target_langs "spa_Latn" "fra_Latn" "deu_Latn" \
    --output "results.json" \
    --use_hf_dataset \
    --save_data_locally \
    --data_format "json"
```

### 方法3：使用预配置的测试脚本

```bash
# 测试Llama 3.2 3B模型
chmod +x test_llama3b.sh
./test_llama3b.sh
```

### 方法4：仅下载数据

```bash
# 查看数据集信息
python download_flores_data.py --info

# 查看可用语言
python download_flores_data.py --list_languages

# 下载特定格式的数据
python download_flores_data.py --splits "dev" "devtest" --formats "json" "csv" "txt"
```

## 参数说明

### 主要参数

- `--model_name`: HuggingFace模型名称或本地模型路径
- `--model_type`: 模型类型 ("causal" 或 "seq2seq")
- `--max_samples`: 最大测试样本数
- `--source_lang`: 源语言代码
- `--target_langs`: 目标语言代码列表
- `--output`: 结果输出文件路径
- `--use_hf_dataset`: 使用HuggingFace FLORES+数据集
- `--save_data_locally`: 保存数据到本地
- `--data_format`: 本地数据格式 ("json", "csv", "txt")

### 可选参数

- `--device`: 设备 ("auto", "cuda", "cpu")
- `--max_length`: 最大序列长度
- `--batch_size`: 批处理大小
- `--data_dir`: FLORES数据目录
- `--split`: 数据集分割 ("dev", "devtest", "test")

## 支持的语言

FLORES+支持200+种语言，常用语言代码包括：

- `eng_Latn`: 英语
- `spa_Latn`: 西班牙语
- `fra_Latn`: 法语
- `deu_Latn`: 德语
- `ita_Latn`: 意大利语
- `por_Latn`: 葡萄牙语
- `rus_Cyrl`: 俄语
- `jpn_Jpan`: 日语
- `kor_Hang`: 韩语
- `cmn_Hans`: 简体中文
- `ara_Arab`: 阿拉伯语
- `hin_Deva`: 印地语

查看完整语言列表：
```bash
python download_flores_data.py --list_languages
```

## 数据格式

### JSON格式
```json
{
  "iso_639_3": "eng",
  "iso_15924": "Latn",
  "text": "Hello, how are you?",
  "domain": "general",
  "source": "flores_plus"
}
```

### 文本格式
```
Hello, how are you?
How is the weather today?
What time is it?
```

### CSV格式
包含所有字段的表格格式，便于数据分析。

## 输出结果

评估完成后会生成两个文件：

1. `results_*.json`: 详细的JSON格式结果
2. `results_*.csv`: CSV格式的语言对结果汇总

结果包含：
- 每个语言对的BLEU分数
- 生成时间统计
- 总体性能指标
- 示例翻译

## 示例结果

```json
{
  "model_name": "meta-llama/Llama-3.2-3B",
  "model_type": "causal",
  "overall": {
    "avg_bleu": 25.6,
    "std_bleu": 3.2,
    "min_bleu": 20.1,
    "max_bleu": 30.5,
    "total_time": 1200.5,
    "num_language_pairs": 5
  },
  "languages": {
    "eng_Latn_spa_Latn": {
      "bleu": 28.3,
      "num_samples": 100,
      "generation_time": 240.2,
      "avg_time_per_sample": 2.4
    }
  }
}
```

## 注意事项

1. **内存要求**: 大模型可能需要大量GPU内存，建议使用较小的batch_size
2. **网络代理**: 如果在中国大陆，需要设置代理来下载模型和数据
3. **模型访问**: 某些模型（如Llama）需要HuggingFace访问权限
4. **数据下载**: FLORES+数据集较大，首次下载需要时间
5. **数据格式**: 支持JSON、CSV、TXT三种格式，推荐使用JSON格式保存完整信息

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减小batch_size和max_length
   python evaluate_model.py --batch_size 1 --max_length 256
   ```

2. **模型下载失败**
   ```bash
   # 检查代理设置
   export HTTP_PROXY="http://127.0.0.1:7890"
   export HTTPS_PROXY="http://127.0.0.1:7890"
   ```

3. **数据集下载失败**
   ```bash
   # 重新下载数据
   python download_flores_data.py --splits "dev" "devtest"
   ```

4. **语言代码不匹配**
   ```bash
   # 查看可用的语言代码
   python download_flores_data.py --list_languages
   ```

## 扩展功能

### 添加新的评估指标

可以在`evaluate_model.py`中添加新的评估指标，如METEOR、ROUGE等。

### 支持更多模型类型

可以扩展支持更多模型架构，如T5、mT5等。

### 批量评估

可以创建脚本批量评估多个模型并比较结果。

### 自定义数据格式

可以修改`download_flores_data.py`来支持其他数据格式。 
