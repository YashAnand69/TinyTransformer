"""
Custom Tokenizer for Tiny Transformer
Character-level and subword tokenizer with vocabulary persistence,
encoding, decoding, and token analysis.
"""

import json
import os
from typing import List, Dict, Tuple, Optional

class CharTokenizer:
    def __init__(self, chars: Optional[List[str]] = None):
        self.special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        if chars is not None:
            # Filter out any accidental overlap with special tokens
            unique_chars = sorted(list(set(c for c in chars if c not in self.special_tokens)))
            self.vocab = self.special_tokens + unique_chars
            self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
            self.itos = {i: ch for i, ch in enumerate(self.vocab)}
        else:
            self.vocab = []
            self.stoi = {}
            self.itos = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.stoi["<PAD>"]

    @property
    def unk_token_id(self) -> int:
        return self.stoi["<UNK>"]

    @property
    def bos_token_id(self) -> int:
        return self.stoi["<BOS>"]

    @property
    def eos_token_id(self) -> int:
        return self.stoi["<EOS>"]

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode string into list of token integers."""
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)
        for char in text:
            tokens.append(self.stoi.get(char, self.unk_token_id))
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode list of token integers back into string."""
        chars = []
        for tid in token_ids:
            if tid in self.itos:
                ch = self.itos[tid]
                if skip_special_tokens and ch in self.special_tokens:
                    continue
                chars.append(ch)
            else:
                if not skip_special_tokens:
                    chars.append("<UNK>")
        return "".join(chars)

    def analyze_tokens(self, text: str) -> List[Dict]:
        """Return list of tokens with metadata (char, id, readable display)."""
        res = []
        for char in text:
            tid = self.stoi.get(char, self.unk_token_id)
            display = char
            if char == " ":
                display = "␣"
            elif char == "\n":
                display = "↵"
            elif char == "\t":
                display = "⇥"
            res.append({
                "char": char,
                "display": display,
                "id": tid
            })
        return res

    def save(self, filepath: str):
        """Save vocabulary to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "vocab": self.vocab,
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "CharTokenizer":
        """Load tokenizer from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.vocab = data["vocab"]
        tok.special_tokens = data.get("special_tokens", ["<PAD>", "<UNK>", "<BOS>", "<EOS>"])
        tok.stoi = {ch: i for i, ch in enumerate(tok.vocab)}
        tok.itos = {i: ch for i, ch in enumerate(tok.vocab)}
        return tok

    @classmethod
    def train_from_text(cls, text: str) -> "CharTokenizer":
        """Build tokenizer directly from text corpus."""
        chars = sorted(list(set(text)))
        return cls(chars)
