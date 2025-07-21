#!/usr/bin/env python3
"""
Download and save FLORES+ dataset locally

This script downloads the FLORES+ dataset from HuggingFace and saves it locally
in various formats (JSON, CSV, TXT) for offline use.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

# 设置代理
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FLORESDataDownloader:
    """FLORES+ dataset downloader"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the downloader
        
        Args:
            data_dir: Directory to save the data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_and_save(self, splits: List[str] = None, formats: List[str] = None):
        """
        下载并保存FLORES+数据集
        
        Args:
            splits: 要下载的数据分割 ("dev", "devtest", "test")
            formats: 保存格式 ("json", "csv", "txt")
        """
        if splits is None:
            splits = ["dev", "devtest"]
        
        if formats is None:
            formats = ["json", "txt"]
        
        logger.info(f"Downloading FLORES+ dataset splits: {splits}")
        logger.info(f"Saving in formats: {formats}")
        
        for split in splits:
            logger.info(f"Processing split: {split}")
            self._download_split(split, formats)
    
    def _download_split(self, split: str, formats: List[str]):
        """下载单个数据分割"""
        try:
            # 创建分割目录
            split_dir = self.data_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            
            # 加载数据集
            logger.info(f"Loading {split} split from HuggingFace...")
            dataset = load_dataset("openlanguagedata/flores_plus", split=split)
            
            # 按语言分组数据
            lang_data = {}
            for item in tqdm(dataset, desc=f"Processing {split}"):
                lang_code = f"{item['iso_639_3']}_{item['iso_15924']}"
                if lang_code not in lang_data:
                    lang_data[lang_code] = []
                lang_data[lang_code].append(item)
            
            logger.info(f"Found {len(lang_data)} languages in {split} split")
            
            # 保存每种语言的数据
            for lang_code, items in lang_data.items():
                self._save_language_data(split_dir, lang_code, items, formats)
            
            # 保存语言统计信息
            self._save_language_stats(split_dir, lang_data)
            
            logger.info(f"Successfully processed {split} split")
            
        except Exception as e:
            logger.error(f"Failed to download {split} split: {e}")
            raise
    
    def _save_language_data(self, split_dir: Path, lang_code: str, items: List[Dict], formats: List[str]):
        """保存单个语言的数据"""
        for format_type in formats:
            try:
                if format_type == "json":
                    # 保存为JSON格式
                    json_file = split_dir / f"{lang_code}.json"
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(items, f, indent=2, ensure_ascii=False)
                    logger.debug(f"Saved {len(items)} items for {lang_code} to {json_file}")
                    
                elif format_type == "csv":
                    # 保存为CSV格式
                    csv_file = split_dir / f"{lang_code}.csv"
                    df = pd.DataFrame(items)
                    df.to_csv(csv_file, index=False, encoding='utf-8')
                    logger.debug(f"Saved {len(items)} items for {lang_code} to {csv_file}")
                    
                elif format_type == "txt":
                    # 保存为纯文本格式
                    txt_file = split_dir / f"{lang_code}.txt"
                    with open(txt_file, 'w', encoding='utf-8') as f:
                        for item in items:
                            f.write(item['text'] + '\n')
                    logger.debug(f"Saved {len(items)} items for {lang_code} to {txt_file}")
                    
            except Exception as e:
                logger.error(f"Failed to save {lang_code} in {format_type} format: {e}")
    
    def _save_language_stats(self, split_dir: Path, lang_data: Dict[str, List]):
        """保存语言统计信息"""
        stats = {
            "total_languages": len(lang_data),
            "languages": {}
        }
        
        for lang_code, items in lang_data.items():
            stats["languages"][lang_code] = {
                "count": len(items),
                "sample_texts": [item['text'][:100] + "..." if len(item['text']) > 100 else item['text'] 
                               for item in items[:3]]  # 前3个样本的预览
            }
        
        # 保存统计信息
        stats_file = split_dir / "language_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved language statistics to {stats_file}")
    
    def list_available_languages(self, split: str = "dev"):
        """列出可用的语言"""
        try:
            logger.info(f"Loading language list from {split} split...")
            dataset = load_dataset("openlanguagedata/flores_plus", split=split)
            
            # 获取所有唯一的语言代码
            languages = list(set([f"{item['iso_639_3']}_{item['iso_15924']}" for item in dataset]))
            languages.sort()
            
            logger.info(f"Found {len(languages)} languages in {split} split:")
            for i, lang in enumerate(languages, 1):
                print(f"{i:3d}. {lang}")
            
            return languages
            
        except Exception as e:
            logger.error(f"Failed to list languages: {e}")
            return []
    
    def get_dataset_info(self):
        """获取数据集信息"""
        try:
            logger.info("Loading dataset information...")
            dataset = load_dataset("openlanguagedata/flores_plus")
            
            info = {
                "dataset_name": "openlanguagedata/flores_plus",
                "splits": list(dataset.keys()),
                "features": list(dataset["dev"].features.keys()) if "dev" in dataset else [],
                "total_languages": 0,
                "sample_data": {}
            }
            
            if "dev" in dataset:
                dev_dataset = dataset["dev"]
                info["total_languages"] = len(set([f"{item['iso_639_3']}_{item['iso_15924']}" for item in dev_dataset]))
                info["sample_data"] = dev_dataset[0] if len(dev_dataset) > 0 else {}
            
            logger.info("Dataset information:")
            print(json.dumps(info, indent=2, ensure_ascii=False))
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get dataset info: {e}")
            return {}


def main():
    parser = argparse.ArgumentParser(description="Download and save FLORES+ dataset locally")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to save data")
    parser.add_argument("--splits", type=str, nargs="+", default=["dev", "devtest"], 
                       choices=["dev", "devtest", "test"], help="Dataset splits to download")
    parser.add_argument("--formats", type=str, nargs="+", default=["json", "txt"], 
                       choices=["json", "csv", "txt"], help="Output formats")
    parser.add_argument("--list_languages", action="store_true", help="List available languages")
    parser.add_argument("--info", action="store_true", help="Show dataset information")
    parser.add_argument("--split", type=str, default="dev", help="Split to use for listing languages")
    
    args = parser.parse_args()
    
    downloader = FLORESDataDownloader(args.data_dir)
    
    if args.info:
        downloader.get_dataset_info()
    elif args.list_languages:
        downloader.list_available_languages(args.split)
    else:
        downloader.download_and_save(args.splits, args.formats)
        logger.info("Download completed successfully!")


if __name__ == "__main__":
    main() 