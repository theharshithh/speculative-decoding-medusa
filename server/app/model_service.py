import torch
from typing import List, Dict, Any
from model.medusa import MedusaModel
from dataclasses import dataclass
from queue import Queue
import asyncio
import time
from threading import Lock

@dataclass
class InferenceRequest:
    prompt: str
    request_id: str
    temperature: float = 0.0
    max_tokens: int = 512
    response_queue: asyncio.Queue = None

class ModelService:
    def __init__(self, model_path: str, batch_size: int = 8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MedusaModel.from_pretrained(model_path)
        self.model.eval()
        self.batch_size = batch_size
        self.request_queue = Queue()
        self.lock = Lock()
        
    def _batch_tokenize(self, prompts: List[str]):
        tokenizer = self.model.get_tokenizer()
        return tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
    
    @torch.no_grad()
    def _process_batch(self, requests: List[InferenceRequest]):
        prompts = [req.prompt for req in requests]
        inputs = self._batch_tokenize(prompts)
    
        for i, request in enumerate(requests):
            single_input = {
                'input_ids': inputs['input_ids'][i:i+1],
                'attention_mask': inputs['attention_mask'][i:i+1] if 'attention_mask' in inputs else None,
                'temperature': request.temperature,
                'max_steps': request.max_tokens
            }
            
            generated_text = ""
            for output in self.model.medusa_generate(**single_input):
                generated_text = output["text"]
            
            asyncio.run_coroutine_threadsafe(
                request.response_queue.put({
                    'request_id': request.request_id,
                    'generated_text': generated_text
                }), 
                asyncio.get_event_loop()
            )
    
    async def generate_text(self, prompt, request_id, temperature=0.0, max_tokens=512):
        response_queue = asyncio.Queue()
        request = InferenceRequest(
            prompt=prompt,
            request_id=request_id,
            temperature=temperature,
            max_tokens=max_tokens,
            response_queue=response_queue
        )
        
        self.request_queue.put(request)
        
        result = await response_queue.get()
        return result
    
    def start_batch_processing(self):
        while True:
            batch = []
            while len(batch) < self.batch_size and not self.request_queue.empty():
                try:
                    request = self.request_queue.get_nowait()
                    batch.append(request)
                except Queue.Empty:
                    break
            
            if batch:
                self._process_batch(batch)
            else:
                time.sleep(0.01)