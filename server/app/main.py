from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import threading
from typing import Optional
from .model_service import ModelService
import os

class GenerationRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 512

class GenerationResponse(BaseModel):
    request_id: str
    generated_text: str

app = FastAPI(title="Medusa LLM Service")

MODEL_PATH = "lmsys/vicuna-7b-v1.3"
BATCH_SIZE = 1 

model_service = ModelService(MODEL_PATH, batch_size=BATCH_SIZE)

batch_thread = threading.Thread(target=model_service.start_batch_processing, daemon=True)
batch_thread.start()

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    try:
        request_id = str(uuid.uuid4())
        
        result = await model_service.generate_text(
            prompt=request.prompt,
            request_id=request_id,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return GenerationResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 