from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

def export_model(model_checkpoint="theharshithh/medusa_vicuna-7b-v1.3", save_directory="onnx/"):
    ort_model = ORTModelForCausalLM.from_pretrained(model_checkpoint, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    ort_model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

def inference(model_checkpoint="theharshithh/medusa_vicuna-7b-v1.3", save_directory="onnx/"):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = ORTModelForCausalLM.from_pretrained(save_directory)
    inputs = tokenizer("My name is Arthur and I live in", return_tensors="pt")
    gen_tokens = model.generate(**inputs)
    print(tokenizer.batch_decode(gen_tokens))

if __name__ == "__main__":
    export_model()
    inference()