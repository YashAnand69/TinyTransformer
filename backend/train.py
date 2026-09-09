"""
Training Pipeline for Tiny Transformer LM from Scratch
Trains the model on the niche corpus, tracks training and validation loss curves,
computes perplexity, implements cosine annealing with warmup,
and exports checkpoints and detailed telemetry history.
"""

import os
import time
import math
import json
from dataclasses import asdict
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW

from model import TinyTransformerLM, TinyTransformerConfig
from tokenizer import CharTokenizer
from corpus import get_full_corpus, save_corpus_to_file

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Linear warmup followed by cosine decay."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample random chunks of text of length block_size."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: str
) -> Tuple[float, float]:
    """Estimate average loss across multiple batches for both splits."""
    model.eval()
    out = {}
    for split_name, split_data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split_data, batch_size, block_size, device)
            _, loss, _ = model(x, targets=y)
            losses[k] = loss.item()
        out[split_name] = losses.mean().item()
    model.train()
    return out["train"], out["val"]

def train():
    print("=" * 60)
    print("🚀 TRAINING TINY TRANSFORMER FROM SCRATCH")
    print("=" * 60)

    # 1. Hardware acceleration detection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Hardware compute device selected: {device.upper()}")

    # 2. Setup directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 3. Prepare niche corpus & tokenizer
    corpus_file = os.path.join(data_dir, "corpus.txt")
    corpus_text = save_corpus_to_file(corpus_file)
    print(f"Corpus loaded: {len(corpus_text):,} characters, {len(corpus_text.split()):,} words.")

    tokenizer = CharTokenizer.train_from_text(corpus_text)
    tok_path = os.path.join(data_dir, "tokenizer.json")
    tokenizer.save(tok_path)
    print(f"Tokenizer saved with vocabulary of {tokenizer.vocab_size} tokens.")

    # 4. Tokenize corpus & split
    encoded_corpus = torch.tensor(tokenizer.encode(corpus_text), dtype=torch.long)
    n_train = int(0.9 * len(encoded_corpus))
    train_data = encoded_corpus[:n_train]
    val_data = encoded_corpus[n_train:]
    print(f"Data split: Train = {len(train_data):,} tokens | Validation = {len(val_data):,} tokens")

    # 5. Model configuration
    config = TinyTransformerConfig(
        block_size=128,
        vocab_size=tokenizer.vocab_size,
        n_layer=4,
        n_head=4,
        d_model=128,
        dropout=0.1,
        bias=False
    )
    model = TinyTransformerLM(config).to(device)
    num_params = model.get_num_params()
    print(f"Model architecture initialized: {num_params:,} parameters.")

    # 6. Training hyper-parameters
    batch_size = 32
    max_steps = 2200
    eval_interval = 50
    eval_iters = 20
    warmup_steps = 100
    max_lr = 1e-3
    min_lr = 1e-4
    weight_decay = 0.01

    optimizer = AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay, betas=(0.9, 0.95))

    history: List[Dict] = []
    sample_generations: List[Dict] = []

    start_time = time.time()
    best_val_loss = float("inf")

    print(f"Starting training run ({max_steps} steps, batch_size={batch_size}, block_size={config.block_size})...")

    # Sample generation at step 0 (untrained noise)
    prompt_starter = "User: Who are you?\nAssistant:"
    ctx = torch.tensor([tokenizer.encode(prompt_starter)], dtype=torch.long, device=device)
    out_tokens, _ = model.generate(ctx, max_new_tokens=60, temperature=0.8, top_k=10)
    untrained_gen = tokenizer.decode(out_tokens[0].tolist())
    sample_generations.append({
        "step": 0,
        "sample": untrained_gen,
        "note": "Untrained random weights initialization (pure entropy)"
    })

    for step in range(1, max_steps + 1):
        # Update learning rate
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Sample batch
        x, y = get_batch(train_data, batch_size, config.block_size, device)

        # Forward & backward pass
        logits, loss, _ = model(x, targets=y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient clipping to prevent instability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Periodic evaluation & logging
        if step % eval_interval == 0 or step == max_steps:
            train_loss, val_loss = estimate_loss(
                model, train_data, val_data, batch_size, config.block_size, eval_iters, device
            )
            perplexity = math.exp(min(val_loss, 20.0))
            elapsed = round(time.time() - start_time, 2)

            point = {
                "step": step,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "perplexity": round(perplexity, 2),
                "lr": round(lr, 6),
                "elapsed_sec": elapsed
            }
            history.append(point)

            print(
                f"Step {step:4d}/{max_steps} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Perplexity: {perplexity:6.2f} | "
                f"LR: {lr:.2e} | "
                f"Elapsed: {elapsed}s"
            )

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "best_val_loss": best_val_loss,
                }, os.path.join(ckpt_dir, "best_model.pt"))

        # Intermediate sample generation at step 400 and step 1000
        if step in (400, 1000):
            model.eval()
            out_tokens, _ = model.generate(ctx, max_new_tokens=80, temperature=0.75, top_k=15)
            mid_gen = tokenizer.decode(out_tokens[0].tolist())
            sample_generations.append({
                "step": step,
                "sample": mid_gen,
                "note": f"Emerging grammar and syntax at step {step}"
            })
            model.train()

    total_training_time = round(time.time() - start_time, 2)
    print(f"\n✅ Training completed in {total_training_time}s! Best Validation Loss: {best_val_loss:.4f}")

    # Final generation with trained model
    model.eval()
    final_out, _ = model.generate(ctx, max_new_tokens=120, temperature=0.7, top_k=15)
    final_gen = tokenizer.decode(final_out[0].tolist())
    sample_generations.append({
        "step": max_steps,
        "sample": final_gen,
        "note": "Fully trained model exhibiting structured styling and domain lexicon"
    })

    # Export training telemetry report
    report = {
        "summary": {
            "model_name": "TinyTransformer-Alchemist-4L4H",
            "parameters": num_params,
            "vocab_size": tokenizer.vocab_size,
            "block_size": config.block_size,
            "layers": config.n_layer,
            "heads": config.n_head,
            "d_model": config.d_model,
            "total_steps": max_steps,
            "batch_size": batch_size,
            "tokens_per_step": batch_size * config.block_size,
            "total_tokens_trained": max_steps * batch_size * config.block_size,
            "training_duration_sec": total_training_time,
            "device": device,
            "best_val_loss": round(best_val_loss, 4),
            "final_train_loss": history[-1]["train_loss"] if history else None,
            "final_val_loss": history[-1]["val_loss"] if history else None,
            "final_perplexity": history[-1]["perplexity"] if history else None,
        },
        "history": history,
        "sample_generations": sample_generations
    }

    report_path = os.path.join(ckpt_dir, "training_history.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Training telemetry saved to: {report_path}")

    # Also save the final model weights
    torch.save({
        "step": max_steps,
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
    }, os.path.join(ckpt_dir, "final_model.pt"))
    print(f"Model weights saved to {ckpt_dir}/best_model.pt and final_model.pt")

if __name__ == "__main__":
    train()
