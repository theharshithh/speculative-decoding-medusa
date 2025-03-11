import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
# from transformers import LlamaForCausalLM as KVLlamaForCausalLM
from llama import LlamaForCausalLM
from utils import *
from kv_cache import initialize_past_key_values
from medusa_choices import vicuna_7b_stage1
from transformers import AutoTokenizer
import os
from huggingface_hub import hf_hub_download


class MedusaConfig(PretrainedConfig):
    def __init__(
        self,
        medusa_num_heads=4,
        medusa_num_layers=1,
        version="2",
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.version = version
        self.base_model_name_or_path = base_model_name_or_path


class ResBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        #(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


class MedusaModel(nn.Module):
    def __init__(
        self,
        base_model,
        medusa_num_heads=4,
        medusa_num_layers=1,
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
    ):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.config.hidden_size
        self.vocab_size = base_model.config.vocab_size
        self.medusa = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        self.medusa_head = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(self.hidden_size)] * medusa_num_layers),
                )
                for _ in range(medusa_num_heads)
            ]
        )

        self.medusa_head.to(self.base_model.dtype).to(self.base_model.device)

    def get_tokenizer(self):
        return self.tokenizer

    @classmethod
    def from_pretrained(
        cls,
        medusa_head_name_or_path,
        base_model=None,
        medusa_num_heads=None,
        **kwargs,
    ):
        medusa_config = MedusaConfig.from_pretrained(medusa_head_name_or_path)
        if medusa_num_heads is not None:
            print("Overriding medusa_num_heads as:", medusa_num_heads)
            medusa_config.medusa_num_heads = medusa_num_heads
        if base_model is not None:
            print("Overriding base_model as:", base_model)
            medusa_config.base_model_name_or_path = base_model
            
        base_model = LlamaForCausalLM.from_pretrained(
            medusa_config.base_model_name_or_path,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            # use_flash_attention_2=False,
            use_cache=True
        )

        model = cls(
            base_model,
            medusa_config.medusa_num_heads,
            medusa_config.medusa_num_layers,
            medusa_config.base_model_name_or_path,
        )
        medusa_head_path = os.path.join(medusa_head_name_or_path, "medusa_lm_head.pt")
        if os.path.exists(medusa_head_path):
            filename = medusa_head_path
        else:
            filename = hf_hub_download(medusa_head_name_or_path, "medusa_lm_head.pt")
        medusa_head_state_dict = torch.load(filename, map_location=base_model.device)
        model.medusa_head.load_state_dict(medusa_head_state_dict, strict=False)
        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        medusa_forward=False
    ):
        """Forward pass of the MedusaModel.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.
            medusa_forward (bool, optional): Whether this is a forward pass for Medusa generation.

        Returns:
            torch.Tensor: A tensor containing predictions from all Medusa heads.
            (Optional) Original predictions from the base model's LM head.
        """
        with torch.no_grad():

            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
        hidden_states = outputs[0].clone()
        medusa_logits = []
        for i in range(self.medusa):
            mhidden_states = self.medusa_head[i](hidden_states)
            mlogits = self.base_model.lm_head(mhidden_states)
            medusa_logits.append(mlogits)
        if output_orig:
            return torch.stack(medusa_logits, dim=0), outputs, orig
        return torch.stack(medusa_logits, dim=0)

    def medusa_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        max_steps=512,
        # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
        medusa_choices=vicuna_7b_stage1,
        posterior_threshold=0.09,  # threshold validation of Medusa output
        posterior_alpha=0.3, # sqrt(posterior_threshold)
    ):
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        input_ids = input_ids.clone()

        if hasattr(self, "medusa_choices") and self.medusa_choices == medusa_choices:
            medusa_buffers = self.medusa_buffers
        else:
            medusa_buffers = generate_medusa_buffers(
                medusa_choices, device=self.base_model.device
            )
        self.medusa_buffers = medusa_buffers
        self.medusa_choices = medusa_choices


        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]

        reset_medusa_mode(self)
        medusa_logits, logits = initialize_medusa(
            input_ids, self, medusa_buffers["medusa_attn_mask"], past_key_values
        )

        new_token = 0
        last_round_token = 0

        for idx in range(max_steps):
            candidates, tree_candidates = generate_candidates(
                medusa_logits,
                logits,
                medusa_buffers["tree_indices"],
                medusa_buffers["retrieve_indices"],
            )

            # Use tree attention to verify the candidates
            medusa_logits, logits, outputs = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                medusa_buffers["medusa_position_ids"],
                input_ids,
                medusa_buffers["retrieve_indices"],
            )

            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha
            )

            # Update the input_ids and logits
            input_ids, logits, medusa_logits, new_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                medusa_buffers["retrieve_indices"],
                outputs,
                logits,
                medusa_logits,
                new_token,
                past_key_values_data,
                current_length_data,
            )

            yield {
                "text": self.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            }

            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break

if __name__ == "__main__":
    import argparse
    import sys
    import torch
    from transformers import AutoTokenizer
    
    def run_tests(args):
        print(f"\n{'='*50}")
        print("Testing Medusa Model Implementation")
        print(f"{'='*50}\n")
        
        try:
            print("1. Testing model initialization...")
            device = torch.device("cuda:0")
            print(f"Using device: {device}")
            
            base_model = LlamaForCausalLM.from_pretrained(
                args.base_model_name,
                torch_dtype=torch.float16,
                device_map="cuda:0",
                # use_flash_attention_2=False,
                use_cache=True
            )
            model = MedusaModel(
                base_model=base_model,
                medusa_num_heads=args.num_heads,
                medusa_num_layers=args.num_layers,
                base_model_name_or_path=args.base_model_name
            )
            model = model.to(device)
            print("✓ Model initialized successfully")
            
            print("\n2. Testing tokenizer functionality...")
            tokenizer = model.get_tokenizer()
            test_input = "Human: What is machine learning?\nAssistant: Machine learning is"
            inputs = tokenizer(test_input, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            print(f"✓ Tokenizer working - Input shape: {inputs['input_ids'].shape}")
            
            print("\n3. Testing model forward pass...")
            with torch.no_grad():
                medusa_outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_orig=True
                )
                medusa_logits, outputs, orig_logits = medusa_outputs
            print(f"✓ Forward pass successful")
            print(f"  - Medusa logits shape: {medusa_logits.shape}")
            print(f"  - Original logits shape: {orig_logits.shape}")
            
            print("\n4. Testing individual Medusa heads...")
            print(f"Number of Medusa heads: {len(model.medusa_head)}")
            for i, head in enumerate(model.medusa_head):
                with torch.no_grad():
                    head_output = head(outputs[0])
                    print(f"✓ Head {i} output shape: {head_output.shape}")
            
            print("\n5. Testing text generation...")
            print("Input prompt:", test_input)
            print("\nGenerating continuation...")
            
            try:
                generated_text = ""
                with torch.no_grad():
                    for output in model.medusa_generate(
                        input_ids=inputs["input_ids"],
                        temperature=args.temperature,
                        max_steps=args.max_new_tokens
                    ):
                        generated_text = output["text"]
                        if args.verbose:
                            print("Current generation:", generated_text)
                
                print("\nFinal generated text:")
                print("-" * 50)
                print(test_input + generated_text)
                print("-" * 50)
                print("✓ Generation completed successfully")
                
            except Exception as e:
                print(f"Error during generation: {str(e)}")
                if args.debug:
                    raise
            
            print("\nAll tests completed successfully! 🎉")
            
        except Exception as e:
            print(f"\n❌ Error during testing: {str(e)}")
            if args.debug:
                raise
            sys.exit(1)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test Medusa model implementation")
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="lmsys/vicuna-7b-v1.3",
        help="Base model name or path"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="Number of Medusa heads"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="Number of layers per Medusa head"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with full error traceback"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print intermediate generation outputs"
    )
    
    args = parser.parse_args()
    run_tests(args)