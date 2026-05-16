import os, math, time, sys, argparse, random, hashlib
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import nullcontext
from collections import Counter
from typing import List, Set, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from tokenizers import Tokenizer
from tqdm.auto import tqdm

# --------------------------------------------------------------
# Import the (improved) transformer implementation
# --------------------------------------------------------------
try:
    from micro_transformer_v2 import MicroTransformer   # RoPE + SwiGLU version
except ImportError:
    from micro_transformer import MicroTransformer


def clean_bpe(text: str) -> str:
    """Replace GPT-2 BPE space/newline markers with actual whitespace."""
    return text.replace("Ġ", " ").replace("Ċ", "\n")


def setup_cpu_optimizations() -> None:
    """Configure PyTorch for maximum CPU throughput."""
    if torch.cuda.is_available():
        return
    num_cores = os.cpu_count() or 4
    torch.set_num_threads(num_cores)
    torch.set_num_interop_threads(max(1, num_cores // 2))
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("medium")
    print(
        f"Optimized for CPU: {num_cores} threads, "
        f"{max(1, num_cores // 2)} interop threads"
    )


setup_cpu_optimizations()


@dataclass
class Config:
    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    data_dir: Path = Path.cwd() / "data" / "clean"
    tokenizer_path: Path = Path.cwd() / "data" / "tokenizer.json"
    ckpt_dir: Path = Path.cwd() / "checkpoints"
    cache_dir: Path = Path.cwd() / ".cache"
    sample_dir: Path = Path.cwd() / "samples"

    # ------------------------------------------------------------------
    # Model / training hyper-params
    # ------------------------------------------------------------------
    seq_len: int = 8192
    vocab_size: int = 32_000

    # size knobs that give ~38M parameters
    n_layers: int = 12
    d_model: int = 300
    n_heads: int = 6
    d_ff: int = 2048

    # ------------------------------------------------------------------
    # Training hyper-params
    # ------------------------------------------------------------------
    batch_size: int = 4
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    max_steps: int = 40_000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    save_every: int = 500          # checkpoint interval
    sample_every: int = 500        # sample interval
    max_samples: int = 2           # how many snippets to write each time

    device: str = "auto"
    dtype: str = "bfloat16"
    compile_model: bool = False
    num_workers: int = 0

    def __post_init__(self) -> None:
        self.tokenizer_path = Path(self.tokenizer_path)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)


class TokenizedLuauDataset(Dataset):
    def __init__(self, cfg: Config, tokenizer: Tokenizer):
        self.cfg = cfg
        self.seq_len = cfg.seq_len
        self.tokenizer = tokenizer

        cache_file = cfg.cache_dir / "tokens_train.pt"
        if cache_file.is_file():
            print(f"Loaded cached tokens from {cache_file}")
            self.tokens = torch.load(cache_file, weights_only=False)
        else:
            print("Building token cache …")
            self.tokens = self._build_corpus()
            torch.save(self.tokens, cache_file)
            print(f"Cached {len(self.tokens):,} tokens → {cache_file}")

        self.n_samples = max(0, len(self.tokens) - self.seq_len - 1)

    def _build_corpus(self) -> torch.Tensor:
        files = list(self.cfg.data_dir.rglob("*.lua"))
        if not files:
            print(f"No *.lua files in {self.cfg.data_dir}")
            return torch.empty(0, dtype=torch.long)

        files = sorted(files)
        eos_id = (
            self.tokenizer.token_to_id("</s>")
            or self.tokenizer.token_to_id("[EOS]")
            or self.tokenizer.token_to_id("[SEP]")
            or self.tokenizer.token_to_id("[PAD]")
            or 0
        )
        all_ids: List[int] = []
        for fp in tqdm(files, desc="Tokenising training data", leave=False):
            try:
                txt = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            ids = self.tokenizer.encode(txt).ids
            all_ids.extend(ids)
            all_ids.append(eos_id)
        return torch.tensor(all_ids, dtype=torch.long)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return x, y


def get_device(cfg: Config) -> torch.device:
    if cfg.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(cfg.device)


def get_dtype(cfg: Config, device: torch.device):
    if device.type != "cuda":
        return torch.float32
    if cfg.dtype == "bfloat16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if cfg.dtype == "float16":
        return torch.float16
    return torch.float32


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def log(msg: str) -> None:
    print(msg, flush=True)


def count_params_on_device(cfg: Config, device: str = "cpu") -> int:
    cfg_device_backup = cfg.device
    cfg.device = device
    import inspect
    sig = inspect.signature(MicroTransformer.__init__)
    allowed = set(sig.parameters.keys())
    model_kwargs = {"vocab_size": cfg.vocab_size, "seq_len": cfg.seq_len}
    def maybe_add(name, value):
        if name in allowed:
            model_kwargs[name] = value
    maybe_add("dim", cfg.d_model)
    maybe_add("d_model", cfg.d_model)
    maybe_add("hidden_dim", cfg.d_model)
    maybe_add("n_layers", cfg.n_layers)
    maybe_add("depth", cfg.n_layers)
    maybe_add("num_layers", cfg.n_layers)
    maybe_add("n_heads", cfg.n_heads)
    maybe_add("heads", cfg.n_heads)
    maybe_add("num_heads", cfg.n_heads)
    maybe_add("dropout", getattr(cfg, "dropout", 0.1))
    model = MicroTransformer(**model_kwargs).to(get_device(cfg))
    total = count_params(model)
    cfg.device = cfg_device_backup
    return total


def generate_sample(
    model: nn.Module,
    tokenizer: Tokenizer,
    device: torch.device,
    prompt_text: str = "local function",
    max_new: int = 80,
    temperature: float = 0.4,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
) -> str:
    model.eval()
    ids = tokenizer.encode(prompt_text).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)

    def sample_next(logits: torch.Tensor) -> torch.Tensor:
        if repetition_penalty != 1.0:
            token_counts = torch.bincount(x.view(-1), minlength=logits.size(-1))
            penalty_mask = (token_counts > 0).float() * (repetition_penalty - 1.0)
            logits = logits - penalty_mask
        logits = logits / max(temperature, 1e-8)
        if top_k > 0:
            top_k_ = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k_)
            cutoff = values[:, -1].unsqueeze(1)
            logits = torch.where(logits < cutoff,
                                torch.full_like(logits, -float("inf")),
                                logits)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            mask = cum_probs > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_logits[mask] = -float("inf")
            logits = torch.zeros_like(logits).scatter_(1, sorted_indices, sorted_logits)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    if hasattr(model, "generate"):
        try:
            out = model.generate(
                x,
                max_new_tokens=max_new,
                temperature=temperature,
            )
            if isinstance(out, tuple):
                out = out[0]
            raw_text = tokenizer.decode(out[0].tolist())
            model.train()
            return clean_bpe(raw_text)
        except TypeError:
            pass

    for _ in range(max_new):
        with torch.no_grad():
            logits = model(x[:, -256:])
            if isinstance(logits, tuple):
                logits = logits[0]
            nxt = sample_next(logits[:, -1, :])
            x = torch.cat([x, nxt], dim=1)

    raw_text = tokenizer.decode(x[0].tolist())
    model.train()
    return clean_bpe(raw_text)


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = get_device(cfg)
        self.dtype = get_dtype(cfg, self.device)
        self.is_cpu = self.device.type == "cpu"

        log(f"Running on {self.device} with dtype {self.dtype}")
        if self.is_cpu:
            log(
                f"CPU mode - seq_len={cfg.seq_len}, "
                f"batch={cfg.batch_size}, workers={cfg.num_workers}"
            )

        self.tokenizer = Tokenizer.from_file(str(cfg.tokenizer_path))
        actual_vocab = self.tokenizer.get_vocab_size()
        if actual_vocab != cfg.vocab_size:
            cfg.vocab_size = actual_vocab
            log(f"Tokenizer vocab size detected: {actual_vocab}")

        self.pad_id = -100
        weight = torch.ones(self.cfg.vocab_size, device=self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id, weight=weight)

        train_ds = TokenizedLuauDataset(cfg, self.tokenizer)
        log(f"Training samples available: {len(train_ds):,}")

        use_pin_memory = not self.is_cpu
        self.train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=use_pin_memory,
            drop_last=True,
        )

        import inspect
        sig = inspect.signature(MicroTransformer.__init__)
        allowed = set(sig.parameters.keys())
        model_kwargs = {"vocab_size": cfg.vocab_size, "seq_len": cfg.seq_len}
        def maybe_add(name, value):
            if name in allowed:
                model_kwargs[name] = value
        maybe_add("dim", cfg.d_model)
        maybe_add("d_model", cfg.d_model)
        maybe_add("hidden_dim", cfg.d_model)
        maybe_add("n_layers", cfg.n_layers)
        maybe_add("depth", cfg.n_layers)
        maybe_add("num_layers", cfg.n_layers)
        maybe_add("n_heads", cfg.n_heads)
        maybe_add("heads", cfg.n_heads)
        maybe_add("num_heads", cfg.n_heads)
        maybe_add("dropout", getattr(cfg, "dropout", 0.1))

        if "dim" in model_kwargs and ("n_heads" in model_kwargs or "heads" in model_kwargs):
            dim_val = model_kwargs["dim"]
            heads_key = "n_heads" if "n_heads" in model_kwargs else "heads"
            heads_val = model_kwargs[heads_key]
            if dim_val % heads_val != 0:
                new_heads = max([h for h in range(1, heads_val + 1) if dim_val % h == 0])
                log(
                    f"Adjusting heads from {heads_val} to {new_heads} "
                    f"to match hidden dimension {dim_val}"
                )
                model_kwargs[heads_key] = new_heads

        self.model = MicroTransformer(**model_kwargs).to(self.device)
        log(f"Model built with {count_params(self.model):,} parameters")

        if cfg.compile_model and hasattr(torch, "compile"):
            log("torch.compile enabled")
            self.model = torch.compile(self.model)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            betas=(0.9, 0.95),
            weight_decay=cfg.weight_decay,
        )

        def lr_lambda(step: int) -> float:
            if step < cfg.warmup_steps:
                return step / max(1, cfg.warmup_steps)
            prog = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
            return cfg.min_lr / cfg.lr + (1 - cfg.min_lr / cfg.lr) * 0.5 * (
                1 + math.cos(math.pi * prog)
            )
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        self.amp_ctx = (
            lambda: torch.amp.autocast(device_type="cuda", dtype=self.dtype)
            if self.device.type == "cuda"
            else nullcontext
        )

        self.step = 0
        self.best_val_loss = float("inf")
        self.tokens_processed = 0
        self.start_time = time.time()

        self._maybe_resume()

    def _maybe_resume(self) -> None:
        if not self.cfg.ckpt_dir.is_dir():
            log("No checkpoint folder - starting fresh")
            return
        ckpt_files = list(self.cfg.ckpt_dir.glob("*.pt"))
        if not ckpt_files:
            log("No checkpoints found starting fresh")
            return
        best_state = None
        best_step = -1
        for ckpt_path in ckpt_files:
            try:
                state = torch.load(ckpt_path, map_location=self.device)
                step = state.get("step", -1)
                if step > best_step:
                    best_step = step
                    best_state = state
            except Exception as e:
                log(f"Could not read {ckpt_path}: {e}")
        if best_state is None:
            log("No valid checkpoints - starting fresh")
            return
        log(f"Resuming from step {best_step}")
        self.model.load_state_dict(best_state["model"])
        self.optimizer.load_state_dict(best_state["optim"])
        self.scheduler.load_state_dict(best_state["sched"])
        self.step = best_state["step"]
        self.best_val_loss = best_state.get("best_val_loss", float("inf"))
        self.tokens_processed = best_state.get("tokens_processed", 0)

    def _train_one_step(self, xb: torch.Tensor, yb: torch.Tensor) -> None:
        xb = xb.to(self.device)
        yb = yb.to(self.device)

        with self.amp_ctx():
            logits = self.model(xb)
            loss = self.criterion(logits.view(-1, logits.size(-1)), yb.view(-1))

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.scheduler.step()

        self.step += 1
        self.tokens_processed += xb.numel()

        if self.step % 10 == 0:
            elapsed = time.time() - self.start_time
            speed = self.tokens_processed / elapsed / 1e6
            log(
                f"Step {self.step:05d} - loss {loss.item():.4f} - "
                f"{speed:.2f} M tok/s - elapsed {fmt_time(elapsed)}"
            )
        if self.step % self.cfg.sample_every == 0:
            self._save_samples()
        if self.step % self.cfg.save_every == 0:
            self._save_checkpoint()

    def train(self) -> None:
        log("Starting training")
        pbar = tqdm(
            total=self.cfg.max_steps,
            desc="Training",
            unit="step",
            leave=True,
        )
        pbar.n = self.step
        pbar.refresh()
        while self.step < self.cfg.max_steps:
            for xb, yb in self.train_loader:
                self._train_one_step(xb, yb)
                pbar.update(1)
                if self.step >= self.cfg.max_steps:
                    break
        pbar.close()
        log("Training complete")
        self._save_checkpoint(final=True)

    def _save_checkpoint(self, best: bool = False, final: bool = False) -> None:
        ckpt_name = (
            "best.pt"
            if best
            else ("final.pt" if final else f"ckpt_step_{self.step}.pt")
        )
        ckpt_path = self.cfg.ckpt_dir / ckpt_name
        torch.save(
            {
                "model": self.model.state_dict(),
                "optim": self.optimizer.state_dict(),
                "sched": self.scheduler.state_dict(),
                "step": self.step,
                "best_val_loss": self.best_val_loss,
                "tokens_processed": self.tokens_processed,
            },
            ckpt_path,
        )
        log(f"Checkpoint saved: {ckpt_path}")

    def _save_samples(self) -> None:
        samples = [
            generate_sample(
                self.model,
                self.tokenizer,
                self.device,
                prompt_text="-- generate something",
                max_new=120,
                temperature=0.4,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
            )
            for _ in range(self.cfg.max_samples)
        ]
        for i, txt in enumerate(samples):
            out_path = self.cfg.sample_dir / f"sample_step{self.step:06d}_{i}.lua"
            out_path.write_text(txt, encoding="utf-8")
        log(f"Saved {len(samples)} sample(s) to {self.cfg.sample_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CPU optimized Micro Transformer trainer (no validation)"
    )
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (default: 4, good for CPU)")
    parser.add_argument("--seq_len", type=int, default=2048,
                        help="Sequence length (default: 2048)")
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_every", type=int, default=500,
                        help="Checkpoint interval")
    parser.add_argument("--sample_every", type=int, default=500,
                        help="Sample generation interval")
    parser.add_argument("--max_samples", type=int, default=2,
                        help="How many generation samples to save each time")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile (requires PyTorch 2.0+)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (0 = main thread, best for CPU)")

    parser.add_argument("--n_layers", type=int, default=12,
                        help="Number of transformer blocks (default 12 → ~40M params)")
    parser.add_argument("--d_model", type=int, default=300,
                        help="Hidden dimension per token (default 300)")
    parser.add_argument("--n_heads", type=int, default=6,
                        help="Number of attention heads (must divide d_model; default 6)")
    parser.add_argument("--d_ff", type=int, default=2048,
                        help="Feed-forward inner dimension (default 2048)")

    parser.add_argument(
        "--show_params",
        action="store_true",
        help="Print total trainable parameters and exit (no training)",
    )

    args = parser.parse_args()

    cfg = Config(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_steps=args.max_steps,
        lr=args.lr,
        save_every=args.save_every,
        sample_every=args.sample_every,
        max_samples=args.max_samples,
        compile_model=args.compile,
        num_workers=args.num_workers,
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
    )

    if args.show_params:
        param_cnt = count_params_on_device(cfg, device="cpu")
        log(f"Model has {param_cnt:,} trainable parameters")
        sys.exit(0)

    trainer = Trainer(cfg)
    trainer.train()