"""
TinyTransformerLM: A Generative Causal Decoder Transformer Built from Scratch
Pure PyTorch implementation with zero black-box dependencies.
Includes Causal Self-Attention, Pre-LayerNorm residuals, GELU MLP,
attention weight extraction, and autoregressive generation with top-k / top-p sampling.
"""

import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict
import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class TinyTransformerConfig:
    block_size: int = 128        # Maximum sequence context window
    vocab_size: int = 100        # Vocabulary size
    n_layer: int = 4             # Number of Transformer blocks
    n_head: int = 4              # Number of attention heads
    d_model: int = 128           # Embedding dimension
    dropout: float = 0.1         # Dropout rate
    bias: bool = False           # Bias in Linear and LayerNorm layers

class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.
    Computes scaled dot-product attention with lower-triangular causal masking
    to prevent tokens from attending to future tokens.
    """
    bias: torch.Tensor

    def __init__(self, config: TinyTransformerConfig):
        super().__init__()
        assert config.d_model % config.n_head == 0, "d_model must be divisible by n_head"
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.head_dim = config.d_model // config.n_head

        # Key, Query, Value projections combined in a single linear layer
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: lower-triangular matrix of shape (1, 1, block_size, block_size)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            )
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, C = x.size()  # Batch size, Sequence length, Embedding dim

        # Calculate query, key, values for all heads in batch and move head forward to the batch dim
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)

        att_weights = None
        if return_attention:
            # Explicit path stays inspectable and portable to ONNX.
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att_weights = F.softmax(att, dim=-1)
            y = self.attn_dropout(att_weights) @ v
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )

        # Re-assemble all head outputs side-by-side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))

        return y, (att_weights if return_attention else None)

class MLP(nn.Module):
    """
    Position-wise Feed-Forward Network.
    Expands hidden dimension by 4x, applies GELU non-linearity, and projects back.
    """
    def __init__(self, config: TinyTransformerConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block using Pre-LayerNorm formulation:
    x = x + Attention(LayerNorm(x))
    x = x + MLP(LayerNorm(x))
    """
    def __init__(self, config: TinyTransformerConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, elementwise_affine=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, elementwise_affine=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-attention branch with residual connection
        attn_out, attn_weights = self.attn(self.ln_1(x), return_attention=return_attention)
        x = x + attn_out
        # Feed-forward branch with residual connection
        x = x + self.mlp(self.ln_2(x))
        return x, attn_weights

class TinyTransformerLM(nn.Module):
    """
    The complete Tiny Transformer Language Model.
    Comprises token and positional embeddings, stacked decoder blocks,
    final layer normalization, and a linear language modeling head.
    """
    def __init__(self, config: TinyTransformerConfig):
        super().__init__()
        self.config = config

        wte = nn.Embedding(config.vocab_size, config.d_model)
        wpe = nn.Embedding(config.block_size, config.d_model)
        drop = nn.Dropout(config.dropout)
        h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        ln_f = nn.LayerNorm(config.d_model, elementwise_affine=config.bias)

        self.transformer = nn.ModuleDict(dict(
            wte = wte,
            wpe = wpe,
            drop = drop,
            h = h,
            ln_f = ln_f,
        ))

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: shared representation between token embedding and LM head
        wte.weight = self.lm_head.weight

        # Initialize weights with standard Gaussian truncated at 0.02
        self.apply(self._init_weights)

        # Special scaled init for residual projections (per GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        """Return total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, f"Input sequence length {T} exceeds block size {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=device)  # shape (T)

        # Forward the transformer
        wte = self.transformer["wte"]
        wpe = self.transformer["wpe"]
        drop = self.transformer["drop"]
        h = self.transformer["h"]
        ln_f = self.transformer["ln_f"]

        assert isinstance(wte, nn.Embedding) and isinstance(wpe, nn.Embedding)
        assert isinstance(drop, nn.Dropout) and isinstance(h, nn.ModuleList)
        assert isinstance(ln_f, nn.LayerNorm)

        tok_emb = wte(idx)                     # token embeddings: (B, T, d_model)
        pos_emb = wpe(pos)                     # position embeddings: (T, d_model)
        x = drop(tok_emb + pos_emb)

        all_attentions: Optional[List[torch.Tensor]] = [] if return_attention else None
        for block in h:
            x, attn_w = block(x, return_attention=return_attention)
            if all_attentions is not None and attn_w is not None:
                all_attentions.append(attn_w)

        x = ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten across batch and sequence dimensions for cross-entropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss, all_attentions

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        return_step_details: bool = False,
        eos_token_id: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Autoregressive generation loop supporting:
        - Temperature scaling (smoothing or sharpening logits)
        - Top-K truncation (retaining only top K candidates)
        - Top-P nucleus sampling (retaining smallest set with cumulative prob >= top_p)
        - Step-by-step diagnostic breakdown for inspection
        """
        self.eval()
        step_details = []

        for _ in range(max_new_tokens):
            # Crop context if sequence exceeds block size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass
            logits, _, _ = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # shape: (B, vocab_size)

            # Pluck logits before temperature for candidate tracking
            raw_probs = F.softmax(logits, dim=-1)

            # Apply temperature and sampling or greedy argmax
            if temperature > 0:
                logits = logits / temperature

                # Optional Top-K filtering
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")

                # Optional Top-P (nucleus) filtering
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[:, indices_to_remove] = -float("inf")

                # Softmax to get probabilities
                probs = F.softmax(logits, dim=-1)

                # Sample next token with fallback to greedy if probabilities degenerate
                if torch.isnan(probs).any() or probs.sum() <= 0:
                    idx_next = torch.argmax(raw_probs, dim=-1, keepdim=True)
                else:
                    idx_next = torch.multinomial(probs.cpu() if generator is not None else probs, num_samples=1, generator=generator).to(idx.device)
            else:
                # Greedy choice
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            if return_step_details:
                # Extract top 5 candidate probabilities for inspection
                top_vals, top_inds = torch.topk(raw_probs[0], k=min(5, raw_probs.size(-1)))
                chosen_id_val = int(idx_next[0, 0].item())
                chosen_prob_val = round(float(raw_probs[0, chosen_id_val].item()), 4)
                step_details.append({
                    "chosen_id": chosen_id_val,
                    "chosen_prob": chosen_prob_val,
                    "top_candidates": [
                        {"id": int(top_inds[i].item()), "prob": round(float(top_vals[i].item()), 4)}
                        for i in range(len(top_inds))
                    ]
                })

            # Append sampled token to sequence
            idx = torch.cat((idx, idx_next), dim=1)
            if eos_token_id is not None and bool((idx_next == eos_token_id).all()):
                break

        return idx, step_details

    def get_attention_matrix(self, idx: torch.Tensor) -> List[List[List[float]]]:
        """
        Extract attention matrix for sequence idx.
        Returns: [layer][head][token_i][token_j] as float lists.
        """
        self.eval()
        with torch.no_grad():
            _, _, all_attns = self(idx, return_attention=True)
            # all_attns: list of length n_layer, each shape (B=1, n_head, T, T)
            result = []
            for layer_attn in all_attns:
                # layer_attn[0]: shape (n_head, T, T)
                heads_list = layer_attn[0].cpu().numpy().tolist()
                result.append(heads_list)
            return result
