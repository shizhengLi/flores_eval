#!/usr/bin/env python3
"""
FLORES-200 Model Evaluation Script

This script evaluates a language model on the FLORES-200 dataset.
Supports both HuggingFace models and local models.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import re # Added for _clean_translation

# 设置代理
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    AutoModelForCausalLM,
    pipeline,
    GenerationConfig
)
from datasets import load_dataset
import sacrebleu
from sacremoses import MosesTokenizer, MosesDetokenizer
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FLORESEvaluator:
    """FLORES-200 dataset evaluator"""
    
    def __init__(
        self,
        model_name: str,
        model_type: str = "causal",  # "causal" or "seq2seq"
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 8,
        data_dir: str = "data",
        languages: Optional[List[str]] = None,
        use_hf_dataset: bool = True
    ):
        """
        Initialize the FLORES evaluator
        
        Args:
            model_name: HuggingFace model name or local path
            model_type: "causal" for decoder-only models, "seq2seq" for encoder-decoder
            device: Device to run inference on
            max_length: Maximum sequence length
            batch_size: Batch size for inference
            data_dir: Directory containing FLORES data
            languages: List of language codes to evaluate (None for all)
            use_hf_dataset: Whether to use HuggingFace FLORES+ dataset
        """
        self.model_name = model_name
        self.model_type = model_type
        self.max_length = max_length
        self.batch_size = batch_size
        self.data_dir = Path(data_dir)
        self.use_hf_dataset = use_hf_dataset
        
        # 设置设备
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # 加载模型和tokenizer
        self._load_model()
        
        # 加载语言列表
        self.languages = self._load_languages(languages)
        
        # 初始化tokenizer
        self.moses_tokenizer = MosesTokenizer()
        self.moses_detokenizer = MosesDetokenizer()
        
    def _load_model(self):
        """加载模型和tokenizer"""
        logger.info(f"Loading model: {self.model_name}")
        
        try:
            if self.model_type == "seq2seq":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # 设置pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _load_languages(self, languages: Optional[List[str]]) -> List[str]:
        """加载支持的语言列表（优先从本地文件获取）"""
        if languages is not None:
            return languages
        
        # 首先尝试从本地文件夹获取语言列表
        local_langs = set()
        for split in ["dev", "devtest"]:
            split_dir = self.data_dir / split
            if split_dir.exists():
                # 查找所有txt和json文件
                txt_files = list(split_dir.glob("*.txt"))
                json_files = list(split_dir.glob("*.json"))
                
                # 从文件名中提取语言代码
                for file_path in txt_files + json_files:
                    lang_code = file_path.stem  # 获取文件名（不含扩展名）
                    if "_" in lang_code:  # 确保是有效的语言代码格式
                        local_langs.add(lang_code)
        
        # 如果本地找到了足够多的语言（至少100个），使用本地语言列表
        if len(local_langs) >= 100:
            languages = sorted(list(local_langs))
            logger.info(f"Loaded {len(languages)} languages from local files")
            return languages
        
        # 如果本地文件不足，尝试从FLORES+数据集加载
        if self.use_hf_dataset:
            try:
                logger.info("Loading language list from FLORES+ dataset...")
                dataset = load_dataset("openlanguagedata/flores_plus", split="dev")
                # 获取所有唯一的语言代码
                languages = list(set([item['iso_639_3'] + '_' + item['iso_15924'] for item in dataset]))
                logger.info(f"Found {len(languages)} languages in FLORES+ dataset")
                return languages
            except Exception as e:
                logger.warning(f"Failed to load languages from HF dataset: {e}")
                # 回退到默认语言列表或本地部分列表
                if local_langs:
                    languages = sorted(list(local_langs))
                    logger.info(f"Using {len(languages)} languages found locally")
                    return languages
        
        # 默认语言列表（FLORES-200的常用语言）
        default_langs = [
            "eng_Latn", "spa_Latn", "fra_Latn", "deu_Latn", "ita_Latn",
            "por_Latn", "rus_Cyrl", "jpn_Jpan", "kor_Hang", "cmn_Hans",
            "ara_Arab", "hin_Deva", "ben_Beng", "urd_Arab", "tel_Telu",
            "tam_Taml", "mar_Deva", "guj_Gujr", "kan_Knda", "mal_Mlym"
        ]
        logger.info(f"Using default language list: {len(default_langs)} languages")
        return default_langs
    
    def _load_flores_data(self, split: str = "devtest") -> Dict[str, List[str]]:
        """加载FLORES数据集（优先使用本地文件）"""
        data = {}
        split_dir = self.data_dir / split
        
        # 先检查本地文件是否存在
        if split_dir.exists():
            logger.info(f"Attempting to load data from local directory: {split_dir}")
            local_files_found = 0
            
            # 遍历所有需要的语言
            for lang in self.languages:
                # 优先尝试加载txt文件（包含纯文本数据）
                txt_file = split_dir / f"{lang}.txt"
                if txt_file.exists():
                    try:
                        with open(txt_file, 'r', encoding='utf-8') as f:
                            data[lang] = [line.strip() for line in f if line.strip()]
                        local_files_found += 1
                        continue
                    except Exception as e:
                        logger.warning(f"Error reading {txt_file}: {e}")
                
                # 如果txt文件不存在或读取失败，尝试json文件
                json_file = split_dir / f"{lang}.json"
                if json_file.exists():
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            items = json.load(f)
                            data[lang] = [item["text"] for item in items]
                        local_files_found += 1
                        continue
                    except Exception as e:
                        logger.warning(f"Error reading {json_file}: {e}")
            
            # 如果找到了大部分语言的本地文件，直接返回
            if local_files_found >= len(self.languages) * 0.8:  # 至少找到80%的语言
                logger.info(f"Successfully loaded data for {local_files_found}/{len(self.languages)} languages from local files")
                return data
            else:
                logger.warning(f"Only found {local_files_found}/{len(self.languages)} languages locally, falling back to HuggingFace")
        
        # 如果本地文件不存在或不完整，则从HuggingFace加载
        if self.use_hf_dataset:
            try:
                logger.info(f"Loading FLORES+ dataset from HuggingFace (split: {split})...")
                dataset = load_dataset("openlanguagedata/flores_plus", split=split)
                
                # 按语言分组数据
                for item in dataset:
                    lang_code = f"{item['iso_639_3']}_{item['iso_15924']}"
                    if lang_code in self.languages:
                        if lang_code not in data:
                            data[lang_code] = []
                        data[lang_code].append(item['text'])
                
                logger.info(f"Loaded data for {len(data)} languages from FLORES+")
                return data
                
            except Exception as e:
                logger.error(f"Failed to load FLORES+ dataset from HuggingFace: {e}")
                # 如果本地已有部分数据，则返回这些数据
                if data:
                    logger.info(f"Returning partial data loaded from local files: {len(data)} languages")
                    return data
                else:
                    raise
        
        # 如果没有设置use_hf_dataset，且本地数据不完整，报错
        if not data:
            raise FileNotFoundError(f"Could not load FLORES data from local directory or HuggingFace: {split_dir}")
            
        return data
    
    def _save_flores_data_locally(self, split: str = "devtest", format: str = "json"):
        """将FLORES+数据集保存到本地（如已存在则跳过下载）"""
        try:
            split_dir = self.data_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)

            # 检查是否已经完整下载了数据（devtest应该有约210个语言文件）
            txt_files = list(split_dir.glob("*.txt"))
            json_files = list(split_dir.glob("*.json"))
            
            # 如果已经有210个语言文件，说明数据已完整下载
            if len(txt_files) >= 200 and len(json_files) >= 200:
                logger.info(f"FLORES+ dataset already exists in {split_dir} ({len(txt_files)} txt files, {len(json_files)} json files), skipping download.")
                return True

            logger.info(f"Downloading and saving FLORES+ dataset locally (split: {split})...")
            # 加载数据集
            dataset = load_dataset("openlanguagedata/flores_plus", split=split)
            # 按语言分组并保存
            lang_data = {}
            for item in dataset:
                lang_code = f"{item['iso_639_3']}_{item['iso_15924']}"
                if lang_code not in lang_data:
                    lang_data[lang_code] = []
                lang_data[lang_code].append(item)
            # 保存每种语言的数据
            for lang_code, items in lang_data.items():
                if format == "json":
                    json_file = split_dir / f"{lang_code}.json"
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(items, f, indent=2, ensure_ascii=False)
                    logger.info(f"Saved {len(items)} items for {lang_code} to {json_file}")
                    txt_file = split_dir / f"{lang_code}.txt"
                    with open(txt_file, 'w', encoding='utf-8') as f:
                        for item in items:
                            f.write(item['text'] + '\n')
                    logger.info(f"Saved text format for {lang_code} to {txt_file}")
                elif format == "csv":
                    csv_file = split_dir / f"{lang_code}.csv"
                    df = pd.DataFrame(items)
                    df.to_csv(csv_file, index=False, encoding='utf-8')
                    logger.info(f"Saved {len(items)} items for {lang_code} to {csv_file}")
            logger.info(f"Successfully saved FLORES+ dataset to {split_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to save FLORES+ dataset locally: {e}")
            return False
    
    def _generate_translation(
        self, 
        source_text: str, 
        source_lang: str, 
        target_lang: str
    ) -> str:
        """生成翻译"""
        try:
            if self.model_type == "seq2seq":
                # 对于seq2seq模型，构建输入格式
                if "nllb" in self.model_name.lower():
                    # NLLB格式
                    input_text = f"{source_lang} {source_text}"
                else:
                    # 通用格式（更简洁的提示词）
                    input_text = f"translate {source_lang} to {target_lang}: {source_text}"
                
                inputs = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=self.max_length,
                        num_beams=5,
                        early_stopping=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
            else:
                # 对于causal模型，使用更直接的prompt格式
                prompt = f"Translate this text from {source_lang} to {target_lang}. Only output the translation:\n{source_text}\n\nTranslation:"
                
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=self.max_length + len(inputs['input_ids'][0]),
                        num_beams=5,
                        early_stopping=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        do_sample=False
                    )
                
                # 提取生成的部分
                generated_tokens = outputs[0][len(inputs['input_ids'][0]):]
                translation = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # 清理翻译结果，去除额外文本
            cleaned_translation = self._clean_translation(translation, source_text, target_lang)
            return cleaned_translation
            
        except Exception as e:
            logger.error(f"Error generating translation: {e}")
            return ""
    
    def _clean_translation(self, translation: str, source_text: str, target_lang: str) -> str:
        """清理模型生成的翻译，去除额外文本和提示词"""
        # 移除首尾空白
        translation = translation.strip()
        
        # 移除源文本（如果模型复制了源文本）
        if source_text in translation:
            parts = translation.split(source_text, 1)
            if len(parts) > 1:
                translation = parts[1].strip()
        
        # 移除常见提示词模式
        patterns = [
            f"Translation:", 
            f"Translate this text from .* to .*:", 
            f"Here is the translation:",
            f"The translation is:",
            f"Translate the following text from .* to .*:",
            f"Translating from .* to .*:"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, translation, re.IGNORECASE)
            for match in matches:
                translation = translation.replace(match, "").strip()
        
        # 处理引号（有些模型会在翻译结果外加引号）
        if translation.startswith('"') and translation.endswith('"'):
            translation = translation[1:-1].strip()
        
        # 如果模型输出了多个翻译方案，只取第一个
        if "\n\n" in translation:
            translation = translation.split("\n\n")[0].strip()
        
        return translation
    
    def _compute_bleu(self, references: List[str], hypotheses: List[str]) -> float:
        """计算BLEU分数"""
        try:
            # 过滤空字符串
            valid_refs = []
            valid_hyps = []
            for ref, hyp in zip(references, hypotheses):
                if ref.strip() and hyp.strip():
                    valid_refs.append(ref.strip())
                    valid_hyps.append(hyp.strip())
            
            if not valid_refs:
                logger.warning("No valid reference-hypothesis pairs found")
                return 0.0
            
            # 正确格式: sacrebleu期望参考译文是列表的列表 (每个翻译有多个参考)
            # 我们这里每个翻译只有一个参考，所以构造一个列表的列表
            refs_list = [[ref] for ref in valid_refs]
            
            # 计算BLEU (让sacrebleu自己处理tokenization)
            bleu = sacrebleu.corpus_bleu(valid_hyps, refs_list)
            return bleu.score
            
        except Exception as e:
            logger.error(f"Error computing BLEU: {e}")
            return 0.0
    
    def evaluate(
        self, 
        split: str = "devtest",
        max_samples: Optional[int] = None,
        source_lang: str = "eng_Latn",
        target_langs: Optional[List[str]] = None,
        save_data_locally: bool = False,
        data_format: str = "json"
    ) -> Dict:
        """
        评估模型在FLORES数据集上的性能
        
        Args:
            split: 数据集分割 ("dev" or "devtest")
            max_samples: 最大样本数（None表示全部）
            source_lang: 源语言
            target_langs: 目标语言列表（None表示所有语言）
            save_data_locally: 是否保存数据到本地
            data_format: 本地数据格式 ("json" or "csv")
            
        Returns:
            评估结果字典
        """
        logger.info(f"Starting evaluation on {split} split")
        
        # 如果需要，先保存数据到本地
        if save_data_locally and self.use_hf_dataset:
            self._save_flores_data_locally(split, data_format)
        
        # 加载数据
        data = self._load_flores_data(split)
        
        if target_langs is None:
            target_langs = self.languages
        
        results = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "split": split,
            "source_lang": source_lang,
            "target_langs": target_langs,
            "languages": {},
            "overall": {}
        }
        
        total_start_time = time.time()
        
        for target_lang in target_langs:
            if target_lang == source_lang:
                continue
                
            logger.info(f"Evaluating {source_lang} -> {target_lang}")
            
            if source_lang not in data:
                logger.warning(f"Source language {source_lang} not found in data")
                continue
                
            source_texts = data[source_lang]
            if max_samples:
                source_texts = source_texts[:max_samples]
            
            # 生成翻译
            translations = []
            start_time = time.time()
            
            for i, source_text in enumerate(tqdm(source_texts, desc=f"{source_lang}->{target_lang}")):
                translation = self._generate_translation(source_text, source_lang, target_lang)
                translations.append(translation)
                # print("\n\nsource_text: ", source_text)
                # print("\ntranslation: ", translation)
                # print("\n\n")
                
                # 每100个样本保存一次中间结果
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(source_texts)} samples")
            
            generation_time = time.time() - start_time
            
            # 获取目标语言的参考译文
            if target_lang not in data:
                logger.warning(f"Target language {target_lang} not found in data")
                bleu_score = 0.0
            else:
                target_texts = data[target_lang]
                if max_samples:
                    target_texts = target_texts[:max_samples]
                
                # 确保参考译文和生成译文数量一致
                if len(target_texts) != len(translations):
                    logger.warning(f"Mismatch in number of references ({len(target_texts)}) and translations ({len(translations)})")
                    min_len = min(len(target_texts), len(translations))
                    target_texts = target_texts[:min_len]
                    translations = translations[:min_len]
                
                # 计算BLEU分数
                # print("\n\ntarget_texts: ", target_texts)
                # print("\ntranslations: ", translations)
                # print("\n\n")
                bleu_score = self._compute_bleu(target_texts, translations)
            
            # 保存结果
            results["languages"][f"{source_lang}_{target_lang}"] = {
                "bleu": bleu_score,
                "num_samples": len(source_texts),
                "generation_time": generation_time,
                "avg_time_per_sample": generation_time / len(source_texts),
                "translations": translations[:10] if len(translations) > 10 else translations  # 只保存前10个作为示例
            }
            
            logger.info(f"{source_lang} -> {target_lang}: BLEU = {bleu_score:.2f}")
        
        total_time = time.time() - total_start_time
        
        # 计算总体统计
        bleu_scores = [lang_result["bleu"] for lang_result in results["languages"].values()]
        results["overall"] = {
            "avg_bleu": np.mean(bleu_scores),
            "std_bleu": np.std(bleu_scores),
            "min_bleu": np.min(bleu_scores),
            "max_bleu": np.max(bleu_scores),
            "total_time": total_time,
            "num_language_pairs": len(results["languages"])
        }
        
        logger.info(f"Evaluation completed in {total_time:.2f}s")
        logger.info(f"Overall BLEU: {results['overall']['avg_bleu']:.2f} ± {results['overall']['std_bleu']:.2f}")
        
        return results
    
    def save_results(self, results: Dict, output_file: str):
        """保存评估结果"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
        
        # 同时保存CSV格式的详细结果
        csv_path = output_path.with_suffix('.csv')
        self._save_csv_results(results, csv_path)
    
    def _save_csv_results(self, results: Dict, csv_path: Path):
        """保存CSV格式的详细结果"""
        rows = []
        for lang_pair, lang_result in results["languages"].items():
            rows.append({
                "language_pair": lang_pair,
                "bleu": lang_result["bleu"],
                "num_samples": lang_result["num_samples"],
                "generation_time": lang_result["generation_time"],
                "avg_time_per_sample": lang_result["avg_time_per_sample"]
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        logger.info(f"Detailed results saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on FLORES-200 dataset")
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model name or local path")
    parser.add_argument("--model_type", type=str, default="causal", choices=["causal", "seq2seq"], help="Model type")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--data_dir", type=str, default="data", help="FLORES data directory")
    parser.add_argument("--split", type=str, default="devtest", choices=["dev", "devtest"], help="Dataset split")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate")
    parser.add_argument("--source_lang", type=str, default="eng_Latn", help="Source language")
    parser.add_argument("--target_langs", type=str, nargs="+", default=None, help="Target languages")
    parser.add_argument("--output", type=str, default="results.json", help="Output file path")
    parser.add_argument("--use_hf_dataset", action="store_true", default=True, help="Use HuggingFace FLORES+ dataset")
    parser.add_argument("--save_data_locally", action="store_true", help="Save FLORES+ data locally")
    parser.add_argument("--data_format", type=str, default="json", choices=["json", "csv"], help="Local data format")
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = FLORESEvaluator(
        model_name=args.model_name,
        model_type=args.model_type,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        use_hf_dataset=args.use_hf_dataset
    )
    
    # 运行评估
    results = evaluator.evaluate(
        split=args.split,
        max_samples=args.max_samples,
        source_lang=args.source_lang,
        target_langs=args.target_langs,
        save_data_locally=args.save_data_locally,
        data_format=args.data_format
    )
    
    # 保存结果
    evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main() 