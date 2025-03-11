# Medusa: Fast LLM Inference with Speculative Decoding

This repository contains an implementation of Medusa, a speculative decoding approach for faster LLM inference. The project includes training, inference, and serving components.

## Project Structure

```
.
├── model.py              # Core Medusa model implementation
├── utils.py             # Utility functions for Medusa operations
├── server/             # FastAPI server implementation
│   ├── app/
│   │   ├── main.py     # FastAPI application
│   │   └── model_service.py  # Model serving logic
│   ├── requirements.txt # Server dependencies
│   └── test_service.py  # Server testing
├── inference/          # Inference related code
│   └── answer.md      # Documentation and explanations
│   └── infer_onnx.py  # Onnx compilation and inference
├── train.py           # Training script
├── dataset.py         # Dataset handling
└── setup_dataset.py   # Dataset preparation utilities
└── requirements.txt   # Project dependencies
└── start_train.sh     # Distributed training 
└── deepspeed.json     # DeepSpeed configuration
```

## Installation

We use `uv` for fast, reliable Python package installation. Here's how to get started:

1. Install `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create a virtual environment and install dependencies:
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate 

uv pip install -r requirements.txt
```

## Features

- **Speculative Decoding**: Implements multiple prediction heads for parallel token generation
- **Dynamic Batching**: Efficient handling of concurrent requests
- **FastAPI Server**: High-performance API server with async support
- **ONNX Support**: Hardware-accelerated inference with ONNX runtime
- **Distributed Training**: Supports distributed training with DeepSpeed

## Training

To train a Medusa model:

1. Prepare your dataset:
```bash
python setup_dataset.py --input_file your_data.json
```
 - This will create train.json and eval.json in the current directory.
 - ShareGPT dataset link (here)[https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json]

2. Start training:
```bash
chmod +x start_train.sh
./start_train.sh
```

## Serving

To run the inference server:

1. Start the FastAPI server:
```bash
cd server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

2. Test the server:
```bash
python test_service.py
```

## Performance

The implementation includes several optimizations:
- Dynamic batching for efficient resource utilization
- Speculative decoding for faster inference
- ONNX runtime for hardware acceleration
- Async request handling


## Inference

To compile the model to ONNX format:
```bash
python infer_onnx.py
```

## Training Logs:

Here is the (Wandb project)[https://wandb.ai/theharshithdev-exp/huggingface/runs/89ut3wkv?nw=nwusertheharshithdev] to view the traning runs. 
