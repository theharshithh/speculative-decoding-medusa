import json
import random
import argparse
import os

def split_data(input_file, train_file, eval_file, split_ratio=0.9, seed=42, total_data_size=50000):
    if not os.path.exists(input_file):
        #download_dataset()
        raise FileNotFoundError(f"File {input_file} not found. Please download the dataset first.")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    random.seed(seed)
    random.shuffle(data)
    
    train_size = int(len(data) * split_ratio)
    data = data[:total_data_size]
    train_data = data[:train_size]
    eval_data = data[train_size:]
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    
    print(f"Split complete: {len(train_data)} train examples and {len(eval_data)} eval examples.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split a JSON file into train and evaluation sets (90/10 split by default).")
    parser.add_argument('--input_file', type=str, default="ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json", help="Path to the input JSON file.")
    parser.add_argument('--train_file', type=str, default="train.json", help="Output train JSON file (default: train.json).")
    parser.add_argument('--eval_file', type=str, default="eval.json", help="Output eval JSON file (default: eval.json).")
    parser.add_argument('--split_ratio', type=float, default=0.9, help="Train split ratio (default 0.9).")
    parser.add_argument('--seed', type=int, default=42, help="Random seed (default: 42).")
    
    args = parser.parse_args()
    split_data(args.input_file, args.train_file, args.eval_file, args.split_ratio, args.seed)