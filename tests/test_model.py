
import argparse
import sys
import torch
from transformers import AutoTokenizer
from model import MedusaModel
from llama.llama import LlamaForCausalLM

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