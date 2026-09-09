"""
FastAPI Backend Server for Tiny Transformer LLM
Provides high-performance inference, attention matrix inspection,
training metrics telemetry, and tokenization services.
"""

import os
import json
import time
from typing import Optional, List, Dict, Any
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model import TinyTransformerLM, TinyTransformerConfig
from tokenizer import CharTokenizer

# Setup Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

app = FastAPI(
    title="Tiny Transformer LLM API",
    description="Scratch-trained Causal Decoder Transformer Inference & Telemetry Engine",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
model: Optional[TinyTransformerLM] = None
tokenizer: Optional[CharTokenizer] = None
training_history: Optional[Dict[str, Any]] = None

def load_system():
    global model, tokenizer, training_history
    tok_path = os.path.join(DATA_DIR, "tokenizer.json")
    if os.path.exists(tok_path):
        tokenizer = CharTokenizer.load(tok_path)
    else:
        tokenizer = None

    ckpt_path = os.path.join(CKPT_DIR, "best_model.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(CKPT_DIR, "final_model.pt")

    if os.path.exists(ckpt_path) and tokenizer is not None:
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        cfg_dict = checkpoint.get("config", {})
        config = TinyTransformerConfig(**cfg_dict)
        model = TinyTransformerLM(config).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"Loaded model from {ckpt_path} on {DEVICE.upper()} with {model.get_num_params():,} params.")

    hist_path = os.path.join(CKPT_DIR, "training_history.json")
    if os.path.exists(hist_path):
        with open(hist_path, "r", encoding="utf-8") as f:
            training_history = json.load(f)

@app.on_event("startup")
def startup_event():
    load_system()

# Request/Response Schemas
class GenerateRequest(BaseModel):
    prompt: str = Field(default="=== LOG ENTRY", description="Input prompt text")
    max_new_tokens: int = Field(default=80, ge=1, le=256, description="Number of tokens to generate")
    temperature: float = Field(default=0.8, ge=0.0, le=2.0, description="Sampling temperature (0.0 for greedy)")
    top_k: Optional[int] = Field(default=20, ge=0, le=100, description="Top-K truncation limit (0 to disable)")
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0, description="Top-P nucleus probability threshold")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")

class AttentionRequest(BaseModel):
    text: str = Field(default="=== LOG ENTRY: The silicon lattice", description="Text to analyze attention across")

class TokenizeRequest(BaseModel):
    text: str = Field(default="", description="Text to tokenize")

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "device": DEVICE,
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "params": model.get_num_params() if model else 0
    }

@app.get("/api/info")
def model_info():
    if model is None or tokenizer is None:
        load_system()
    if model is None or tokenizer is None:
        return {"status": "Model still training or initializing"}

    return {
        "model_name": "TinyTransformer-Alchemist",
        "architecture": "Causal Decoder-Only Transformer (GPT Architecture)",
        "parameters": model.get_num_params(),
        "vocab_size": tokenizer.vocab_size,
        "block_size": model.config.block_size,
        "n_layer": model.config.n_layer,
        "n_head": model.config.n_head,
        "d_model": model.config.d_model,
        "head_dim": model.config.d_model // model.config.n_head,
        "device": DEVICE,
        "attention_type": "Multi-Head Scaled Dot-Product with Causal Mask",
        "activation": "GELU",
        "norm": "Pre-LayerNorm with Residual Skip Connections"
    }

@app.get("/api/metrics")
def get_metrics():
    global training_history
    if training_history is None:
        hist_path = os.path.join(CKPT_DIR, "training_history.json")
        if os.path.exists(hist_path):
            with open(hist_path, "r", encoding="utf-8") as f:
                training_history = json.load(f)
    if training_history is None:
        raise HTTPException(status_code=404, detail="Training telemetry not yet available.")
    return training_history

@app.get("/api/corpus")
def get_corpus_samples():
    corpus_file = os.path.join(DATA_DIR, "corpus.txt")
    if not os.path.exists(corpus_file):
        raise HTTPException(status_code=404, detail="Corpus not found.")
    with open(corpus_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    sections = text.split("=== ")
    cleaned_sections = ["=== " + s.strip() for s in sections if s.strip()][:8]
    return {
        "total_characters": len(text),
        "total_words": len(text.split()),
        "sample_excerpts": cleaned_sections
    }

@app.post("/api/tokenize")
def tokenize_text(req: TokenizeRequest):
    if tokenizer is None:
        load_system()
    if tokenizer is None:
        raise HTTPException(status_code=500, detail="Tokenizer not loaded.")
    
    tokens = tokenizer.analyze_tokens(req.text)
    token_ids = [t["id"] for t in tokens]
    return {
        "text": req.text,
        "token_count": len(tokens),
        "tokens": tokens,
        "ids": token_ids
    }

@app.post("/api/generate")
def generate_text(req: GenerateRequest):
    if model is None or tokenizer is None:
        load_system()
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is still training or not loaded.")

    if req.seed is not None:
        torch.manual_seed(req.seed)

    start_time = time.time()
    
    prompt = req.prompt if req.prompt else "=== LOG"
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [tokenizer.bos_token_id]

    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

    # Autoregressive generation
    out_tensor, step_details = model.generate(
        input_tensor,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        return_step_details=True
    )

    all_token_ids = out_tensor[0].cpu().tolist()
    full_text = tokenizer.decode(all_token_ids)
    generated_ids = all_token_ids[len(prompt_ids):]
    generated_text = tokenizer.decode(generated_ids)

    # Annotate step details with character representation
    for step in step_details:
        step["chosen_char"] = tokenizer.itos.get(step["chosen_id"], "")
        for cand in step["top_candidates"]:
            cand["char"] = tokenizer.itos.get(cand["id"], "")
            if cand["char"] == " ":
                cand["display"] = "␣ (space)"
            elif cand["char"] == "\n":
                cand["display"] = "↵ (newline)"
            else:
                cand["display"] = cand["char"]

    latency_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "full_text": full_text,
        "tokens_generated": len(generated_ids),
        "latency_ms": latency_ms,
        "tokens_per_sec": round(len(generated_ids) / max(latency_ms / 1000, 0.001), 1),
        "step_details": step_details  # Return all step details for full sequence confidence mapping
    }

@app.post("/api/attention")
def get_attention(req: AttentionRequest):
    if model is None or tokenizer is None:
        load_system()
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is still training or not loaded.")

    text = req.text[:min(len(req.text), model.config.block_size)]
    token_ids = tokenizer.encode(text)
    if not token_ids:
        token_ids = [tokenizer.bos_token_id]

    # Limit to at most 32 tokens for clean visualization matrix
    token_ids = token_ids[:32]
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)

    attn_matrix = model.get_attention_matrix(input_tensor)
    tokens_meta = tokenizer.analyze_tokens(tokenizer.decode(token_ids))

    n_layers = len(attn_matrix)
    n_heads = len(attn_matrix[0]) if attn_matrix else 0

    return {
        "text": tokenizer.decode(token_ids),
        "token_count": len(token_ids),
        "tokens": [t["display"] for t in tokens_meta],
        "token_ids": token_ids,
        "num_layers": n_layers,
        "num_heads": n_heads,
        "n_layer": n_layers,
        "n_head": n_heads,
        "attention_weights": attn_matrix,  # [layer][head][token_i][token_j]
        "weights": attn_matrix             # direct compatibility with frontend heatmap
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8008))
    uvicorn.run(app, host="127.0.0.1", port=port)
