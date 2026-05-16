import math
import time
import random
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap even and odd dimensions for rotary embedding."""
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin = freqs.cos(), freqs.sin()
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


def build_rope_cache(seq_len: int, dim: int, device: torch.device) -> torch.Tensor:
    """Pre compute rotary frequencies for a given sequence length."""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)               # [seq_len, dim/2]
    freqs = torch.stack([freqs, freqs], dim=-1).flatten(-2)    # [seq_len, dim]
    return freqs


class SwiGLU(nn.Module):
    """SwiGLU activation (SiLU multiplied by the split half of the input)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, seq_len: int, dropout: float = 0.0):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "rope_cache",
            build_rope_cache(seq_len, self.head_dim, torch.device("cpu")),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        rope = self.rope_cache[:T, :].unsqueeze(0).unsqueeze(0)   # (1,1,T,head_dim)
        q, k = apply_rotary_pos_emb(q, k, rope)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float("-inf"))
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = torch.matmul(attn_probs, v)            # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Feed forward block that uses SwiGLU activation."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim * 2, bias=False)   # *2 for SwiGLU split
        self.act = SwiGLU()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.fc2(x)


class TransformerBlock(nn.Module):
    """One transformer block: attention, feed forward, residual connections, and layer norms."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        seq_len: int,
        ff_hidden_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads, seq_len, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dim * ff_hidden_mult, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        h = self.attn(h, attn_mask)
        x = x + self.dropout(h)

        h = self.norm2(x)
        h = self.ff(h)
        return x + self.dropout(h)


class MicroTransformer(nn.Module):
    """
    Tiny but capable Micro Transformer (about 32M parameters).

    Default configuration (fits comfortably in 16GB VRAM):
        dim       = 560      # hidden size per token
        n_heads   = 8
        n_layers  = 6
        seq_len   = 2048     # must match the trainers seq_len
        dropout   = 0.1
    Custom values can be passed when constructing the model.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int = 2048,
        dim: int = 560,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.dim = dim
        self.vocab_size = vocab_size

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, dim))

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    n_heads=n_heads,
                    seq_len=seq_len,
                    ff_hidden_mult=4,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        idx: (B, T) token ids
        attention_mask: (B, 1, 1, T)
        Returns logits of shape (B, T, vocab_size)
        """
        B, T = idx.size()
        if T > self.seq_len:
            raise ValueError(f"Sequence length {T} exceeds model capacity {self.seq_len}")

        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :T, :]
        x = token_embeddings + position_embeddings
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, attn_mask=attention_mask)

        x = self.norm(x)
        logits = self.head(x)
        return logits

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.4,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.2,
        repetition_window: int = 256,
        tokenizer: Optional[Tokenizer] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with fixed temperature (0.4) and repetition penalty (1.2).
        """
        self.eval()
        device = idx.device
        generated = idx

        for _ in range(max_new_tokens):
            cur_idx = generated[:, -self.seq_len :]
            logits = self(cur_idx)[:, -1, :]
            logits = logits / max(temperature, 1e-8)

            if repetition_penalty != 1.0:
                recent = (
                    generated[:, -repetition_window:]
                    if repetition_window > 0
                    else generated
                )
                self._apply_rep_penalty(logits, recent, repetition_penalty)

            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                top_vals, _ = torch.topk(logits, top_k, dim=-1)
                cutoff = top_vals[:, -1].unsqueeze(1)
                logits = torch.where(logits < cutoff,
                                    torch.full_like(logits, -float("inf")),
                                    logits)

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                mask = cumulative_probs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_logits[mask] = -float("inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_indices, sorted_logits)

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)

        self.train()
        return generated

    def _apply_rep_penalty(self, logits: torch.Tensor, recent: torch.Tensor, penalty: float) -> None:
        """Apply repetition penalty to logits."""
        if penalty == 1.0 or recent.numel() == 0:
            return
        B, vocab = logits.shape
        mask = torch.zeros((B, vocab), dtype=torch.bool, device=logits.device)

        batch_offsets = (torch.arange(B, device=logits.device) * vocab).unsqueeze(1)
        flat = (recent + batch_offsets).view(-1)
        uniq = torch.unique(flat)
        batch_idx = uniq // vocab
        token_idx = uniq % vocab
        mask[batch_idx, token_idx] = True

        logits[mask] = torch.where(
            logits[mask] > 0,
            logits[mask] / penalty,
            logits[mask] * penalty,
        )

    def generate_multi_prompt(
        self,
        initial_idx: torch.Tensor,
        secondary_idx: torch.Tensor,
        switch_step: int = 500,
        max_new_tokens: int = 1000,
        temperature: float = 0.4,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.2,
        repetition_window: int = 256,
    ) -> torch.Tensor:

        self.eval()
        device = initial_idx.device
        generated = initial_idx
        steps_generated = 0

        def apply_rep(logits: torch.Tensor, recent: torch.Tensor) -> None:
            if repetition_penalty == 1.0 or recent.numel() == 0:
                return
            B, vocab = logits.shape
            mask = torch.zeros((B, vocab), dtype=torch.bool, device=logits.device)

            batch_offsets = (torch.arange(B, device=logits.device) * vocab).unsqueeze(1)
            flat = (recent + batch_offsets).view(-1)
            uniq = torch.unique(flat)
            batch_idx = uniq // vocab
            token_idx = uniq % vocab
            mask[batch_idx, token_idx] = True
            logits[mask] = logits[mask] / repetition_penalty

        while steps_generated < max_new_tokens:
            if steps_generated == switch_step:
                generated = torch.cat((generated, secondary_idx), dim=1)

            cur_idx = generated[:, -self.seq_len :]
            logits = self(cur_idx)[:, -1, :]
            logits = logits / max(temperature, 1e-8)

            if repetition_penalty != 1.0:
                recent = (
                    generated[:, -repetition_window:]
                    if repetition_window > 0
                    else generated
                )
                apply_rep(logits, recent)

            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, top_k, dim=-1)
                cutoff = values[:, -1].unsqueeze(1)
                logits = torch.where(logits < cutoff,
                                    torch.full_like(logits, -float("inf")),
                                    logits)

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                mask = cumulative_probs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_logits[mask] = -float("inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_indices, sorted_logits)

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            steps_generated += 1

        self.train()
        return generated


def load_tokenizer(tokenizer_path: str | Path) -> Tokenizer:
    """Load a tokenizers JSON file."""
    path = Path(tokenizer_path)
    if not path.is_file():
        raise FileNotFoundError(f"Tokenizer file not found at {path}")
    return Tokenizer.from_file(str(path))


class LuauFileDataset(Dataset):
    """Dataset that reads *.lua files, tokenizes them, and returns fixed length tensors."""

    def __init__(self, root_dir: str | Path, tokenizer: Tokenizer, max_seq_len: int):
        self.root = Path(root_dir)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Directory {self.root} does not exist.")
        self.filepaths = sorted(self.root.glob("*.lua"))
        if not self.filepaths:
            raise RuntimeError(f"No *.lua files found in {self.root}")

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_id = tokenizer.token_to_id("<pad>")
        if self.pad_id is None:
            self.pad_id = tokenizer.token_to_id("<unk>") or 0

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        script_path = self.filepaths[idx]
        raw = script_path.read_text(encoding="utf-8")
        token_ids = self.tokenizer.encode(raw).ids

        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[: self.max_seq_len]
        else:
            token_ids = token_ids + [self.pad_id] * (self.max_seq_len - len(token_ids))

        return torch.tensor(token_ids, dtype=torch.long)


def get_train_dataloader(
    batch_size: int,
    max_seq_len: int,
    tokenizer_path: str | Path = "data/tokenizer.json",
    clean_root: str | Path = "data/clean/train",
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    tokenizer = load_tokenizer(tokenizer_path)
    dataset = LuauFileDataset(root_dir=clean_root, tokenizer=tokenizer, max_seq_len=max_seq_len)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return loader


if __name__ == "__main__":
    try:
        dl = get_train_dataloader(
            batch_size=4,
            max_seq_len=1024,
            tokenizer_path="data/tokenizer.json",
            clean_root="data/clean/train",
            num_workers=0,
        )
        batch = next(iter(dl))
        print("DataLoader works! Batch shape:", batch.shape)
        print("First 10 token IDs of the first sample:", batch[0, :10].tolist())
    except Exception as e:
        print("Error while building the DataLoader:")
        print(e)

    try:
        tokenizer = load_tokenizer("data/tokenizer.json")
        vocab_sz = tokenizer.get_vocab_size()
        model = MicroTransformer(vocab_size=vocab_sz, seq_len=2048).to("cpu")
        logits = model(batch[:, :-1])
        print("Model forward OK - logits shape:", logits.shape)
    except Exception as e:
        print("Model sanity check failed:")
        print(e)