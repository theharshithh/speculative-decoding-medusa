import torch
from typing import Dict
import transformers
from torch.utils.data import Dataset, DataLoader
import json
import argparse
from dataclasses import dataclass

def rank0_print(*args):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*args)
    else:
        print(*args)

IGNORE_TOKEN_ID = -100

def convert_conversation_format(raw_data):
    formatted_data = []
    for item in raw_data:
        conversation = []
        for turn in item["conversations"]:
            role = "assistant" if turn["from"].lower() == "gpt" else "user"
            conversation.append({
                "role": role,
                "content": turn["value"]
            })
        formatted_data.append(conversation)
    return formatted_data

def format_conversation(conversation):
    """
    Format a conversation into a single string using a simple template.
    
    Args:
        conversation (list): List of conversation turns with role and content
        
    Returns:
        str: Formatted conversation string
    """
    formatted_text = ""
    for turn in conversation:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            formatted_text += f"Human: {content}\n"
        else:
            formatted_text += f"Assistant: {content}\n"
    return formatted_text.strip()

def preprocess(sources, tokenizer: transformers.PreTrainedTokenizer):
    formatted_sources = convert_conversation_format(sources)
    
    conversations = []
    prompts = []
    for i, conversation in enumerate(formatted_sources):
        # Use our custom formatting instead of apply_chat_template
        prompt = format_conversation(conversation)
        prompts.append(prompt)
        conversations.append(conversation)

    encoding = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
    )
    targets = torch.full_like(encoding.input_ids, IGNORE_TOKEN_ID)
    input_ids = encoding.input_ids

    for conv_index, (conversation, target, prompt) in enumerate(zip(conversations, targets, prompts)):
        for turn in conversation:
            if turn["role"] == "assistant":
                content = f"Assistant: {turn['content']}"  # Match our formatting
                try:
                    start = prompt.index(content.strip())
                    stop = start + len(content)
                    indices = []
                    for tok_index, (tok_start, tok_stop) in enumerate(encoding.offset_mapping[conv_index]):
                        if tok_stop >= start or tok_start < tok_stop:
                            indices.append(tok_index)
                    target[indices] = encoding.input_ids[conv_index][indices]
                except ValueError as e:
                    print(f"Warning: Could not find assistant response in prompt. Skipping this turn. Error: {e}")
                    continue

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = raw_data
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )
    
class LazySupervisedDataset(Dataset):
    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret

def create_dataset(tokenizer: transformers.PreTrainedTokenizer, data_args):
    rank0_print("Loading data...")
    dataset_cls = (LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset)

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

@dataclass
class DataArguments:
    data_path: str
    eval_data_path: str = None
    model_name: str = "lmsys/vicuna-7b-v1.3"
    batch_size: int = 4
    max_length: int = 2048

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="eval.json", help="Path to training data JSON file")
    parser.add_argument("--eval_data_path", type=str, default="eval.json", help="Path to evaluation data JSON file")
    parser.add_argument("--model_name", type=str, default="lmsys/vicuna-7b-v1.3", help="Model name or path")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    args = parser.parse_args()

    data_args = DataArguments(**vars(args))

    print(f"Loading tokenizer from {data_args.model_name}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        data_args.model_name,
        model_max_length=data_args.max_length,
        padding_side="right",
        use_fast=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    special_tokens = {
        "additional_special_tokens": ["Human:", "Assistant:"]
    }
    tokenizer.add_special_tokens(special_tokens)

    print("Creating datasets...")
    datasets = create_dataset(tokenizer, data_args)
    train_dataset = datasets["train_dataset"]
    eval_dataset = datasets["eval_dataset"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_args.batch_size,
        shuffle=True,
    )

    print("\nDataset Statistics:")
    print(f"Number of training examples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Number of validation examples: {len(eval_dataset)}")

    print("\nTesting data loading...")
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 2:
            break
            
        print(f"\nBatch {batch_idx + 1}:")
        print(f"Input shape: {batch['input_ids'].shape}")
        print(f"Label shape: {batch['labels'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        
        # Print both the input text and where the labels are non-ignored
        decoded_text = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False)
        print(f"\nSample decoded text (truncated):\n{decoded_text[:200]}...")
        
        # Show where the labels are (assistant responses)
        label_mask = batch['labels'][0] != IGNORE_TOKEN_ID
        label_tokens = batch['input_ids'][0][label_mask]
        print(f"\nAssistant response tokens (truncated):\n{tokenizer.decode(label_tokens, skip_special_tokens=False)[:200]}...") 
