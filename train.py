import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SupervisedDataset, create_dataset, DataArguments, format_conversation
import transformers
from model.medusa import MedusaModel, MedusaConfig
from transformers import Trainer, BitsAndBytesConfig
from transformers.trainer_pt_utils import LabelSmoother
from dataclasses import dataclass, field
from typing import Optional
from torch.nn import CrossEntropyLoss
import deepspeed
import math
import pathlib
import os
from safetensors.torch import save_file

IGNORE_TOKEN_ID = LabelSmoother.ignore_index # -100

def rank0_print(*args):
    """Print only on rank 0 in distributed training"""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*args)
    else:
        print(*args)

class CustomizedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if hasattr(model, "module"):
            medusa = model.module.medusa
        else:
            medusa = model.medusa

        # Debug input shapes and values
        if torch.distributed.get_rank() == 0:  # Only print on rank 0
            max_token = inputs['input_ids'].max().item()
            min_token = inputs['input_ids'].min().item()
            vocab_size = self.model.config.vocab_size
            rank0_print(f"Max token ID: {max_token}, Min token ID: {min_token}, Vocab size: {vocab_size}")
            if max_token >= vocab_size:
                rank0_print(f"WARNING: Token ID {max_token} out of bounds for vocab size {vocab_size}")

        # Create proper position_ids
        seq_length = inputs["input_ids"].size(1)
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=inputs["input_ids"].device)
        position_ids = position_ids.unsqueeze(0).expand_as(inputs["input_ids"])
            
        # Pass position_ids to the model
        logits = model(
            input_ids=inputs["input_ids"], 
            attention_mask=inputs["attention_mask"],
            position_ids=position_ids
        )
        
        labels = inputs["labels"]
        # Shift so that tokens < n predict n
        loss = 0
        loss_fct = CrossEntropyLoss()
        log = {}
        for i in range(medusa):
            medusa_logits = logits[i, :, : -(2 + i)].contiguous()
            medusa_labels = labels[..., 2 + i :].contiguous()
            medusa_logits = medusa_logits.view(-1, logits.shape[-1])
            medusa_labels = medusa_labels.view(-1)
            medusa_labels = medusa_labels.to(medusa_logits.device)
            loss_i = loss_fct(medusa_logits, medusa_labels)
            loss += loss_i
            not_ignore = medusa_labels.ne(IGNORE_TOKEN_ID)
            medusa_labels = medusa_labels[not_ignore]

            for k in range(1, 2):
                _, topk = medusa_logits.topk(k, dim=-1)
                topk = topk[not_ignore]
                correct = topk.eq(medusa_labels.unsqueeze(-1)).any(-1)
                log[f"medusa{i}_top{k}"] = correct.float().mean().item()

            log[f"medusa{i}_loss"] = loss_i.item()
            
        self.log(log)
        return (loss, logits) if return_outputs else loss


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="lmsys/vicuna-7b-v1.3"
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load model in 4-bit precision"}
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load model in 8-bit precision"}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default="train.json",
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(
        default="eval.json", metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    report_to: Optional[str] = None
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    medusa_num_heads: int = field(
        default=1,
        metadata={"help": "Number of Medusa heads."},
    )
    medusa_num_layers: int = field(
        default=1,
        metadata={"help": "Number of layers for each Medusa head."},
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.local_rank != -1:
        # Now we can simply use local_rank as is, since PyTorch only sees the A100s
        device_map = f"cuda:{training_args.local_rank}"
        torch.cuda.set_device(training_args.local_rank)
        rank0_print(f"Local rank {training_args.local_rank} using device: {device_map}")
    else:
        device_map = "cuda:0"
        torch.cuda.set_device(0)

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    rank0_print(f"Loading tokenizer from {model_args.model_name_or_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    
    special_tokens = {
        "additional_special_tokens": ["Human:", "Assistant:"]
    }
    num_added = tokenizer.add_special_tokens(special_tokens)
    rank0_print(f"Added {num_added} special tokens. Tokenizer vocab size: {len(tokenizer)}")

    rank0_print(tokenizer(["This is a test", "secondary"], padding=True))
    
    test_conversation = [{"role": "user", "content": "This is a test"}]
    formatted_test = format_conversation(test_conversation)
    rank0_print("Formatted conversation test:", formatted_test)
    rank0_print("Tokenized conversation:", tokenizer(formatted_test))
    print('Tokenizer working')

    rank0_print("Loading base model...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
    )
    
    # IMPORTANT: Resize the model's token embeddings to match the tokenizer
    if len(tokenizer) > model.config.vocab_size:
        rank0_print(f"Resizing token embeddings from {model.config.vocab_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    
    model = model.to(device_map)

    for param in model.parameters():
        param.requires_grad = False

    rank0_print(f"hitting medusa model on device {device_map}")
    medusa_model = MedusaModel(
        base_model=model,
        medusa_num_heads=training_args.medusa_num_heads,
        medusa_num_layers=training_args.medusa_num_layers,
        base_model_name_or_path=model_args.model_name_or_path
    )
    medusa_model = medusa_model.to(device_map)

    training_args.output_dir = os.path.join(
        training_args.output_dir,
        f"medusa_{model_args.model_name_or_path.split('/')[-1]}_heads_{training_args.medusa_num_heads}_layers_{training_args.medusa_num_layers}"
    )
    os.makedirs(training_args.output_dir, exist_ok=True)

    rank0_print("Loading datasets...")
    data_module = create_dataset(tokenizer=tokenizer, data_args=data_args)

    rank0_print("Saving Medusa config...")
    medusa_config = MedusaConfig(
        medusa_num_heads=training_args.medusa_num_heads,
        medusa_num_layers=training_args.medusa_num_layers,
        base_model_name_or_path=model_args.model_name_or_path,
    )
    medusa_config.save_pretrained(training_args.output_dir)

    trainer = CustomizedTrainer(
        model=medusa_model,
        tokenizer=tokenizer,
        args=training_args,        
        **data_module
    )

    rank0_print("Starting training...")
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    model.config.use_cache = True
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    if training_args.local_rank in [-1, 0]:
        rank0_print("Saving Medusa heads...")
        if hasattr(medusa_model, "module"):
            medusa_head = medusa_model.module.medusa_head
        else:
            medusa_head = medusa_model.medusa_head

        with deepspeed.zero.GatheredParameters(medusa_head.parameters()):
            state_dict = medusa_head.state_dict()

        tokenizer.save_pretrained(training_args.output_dir)
        save_file(
            state_dict,
            os.path.join(training_args.output_dir, "medusa_lm_head.safetensors"),
        )

    rank0_print("Training completed successfully!")


if __name__ == "__main__":
    train()