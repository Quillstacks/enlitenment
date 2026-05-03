"""Offline generator for lecture-09 precomputed weights.

Trains the larger ('next scale up') variants of the char-level transformer
on Schiller's *Die Raeuber* and saves the weights as .npz so the pyodide
notebook can load them without needing torch in the browser.

The recipe matches what students implement in the notebook (1-block
transformer, char vocab, causal or bidirectional attention) but with a
larger d_model, a longer context, and many more training steps than would
fit in a JupyterLite session. The student's tiny in-browser model and
these precomputed weights share the SAME forward-pass implementation, so
loading the precomputed weights drops in transparently.

Usage (from repo root):
    python3 -m venv .venv
    source .venv/bin/activate
    pip install torch numpy
    python content/unsupervised-deep-learning/lecture-09/generate_precomputed.py

Outputs (alongside this script):
    vocab.json                    char <-> index map
    bidirectional_lm_weights.npz  encoder-only, single-token masking (BERT)
    span_lm_weights.npz           encoder-only, span masking (SpanBERT/T5)
    causal_lm_weights.npz         decoder-only, next-token (GPT)
    jepa_weights.npz              JEPA student weights (teacher = EMA copy)
    training_curves.npz           loss curves for the four runs
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE   = Path(__file__).resolve().parent
SEED   = 0
DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# Hyperparameters: 'next scale up' from what the notebook trains in-browser
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Architecture matches the notebook's numpy ``forward()``: one attention
    # block, single head, no head reshaping. d_model and context are larger
    # than the in-browser tiny model (8 / 16) so the loaded weights actually
    # demonstrate "what a properly-trained version looks like".
    d_model:    int = 32
    n_heads:    int = 1
    d_ff:       int = 64
    context:    int = 32
    n_layers:   int = 1
    dropout:    float = 0.0     # off so the saved weights match a clean forward
    batch_size: int = 64
    n_steps:    int = 1500
    lr:         float = 3e-4
    mask_frac:  float = 0.15
    span_mean:  float = 3.0
    ema_m:      float = 0.99


# ---------------------------------------------------------------------------
# Corpus & tokeniser
# ---------------------------------------------------------------------------

def load_corpus(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    """Char-level vocab. Reserves index 0 for [MASK]."""
    chars = sorted(set(text))
    # Slot 0 reserved for [MASK]; real chars start at 1
    stoi = {"\x00": 0}                                 # [MASK] sentinel
    for i, c in enumerate(chars, start=1):
        stoi[c] = i
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> np.ndarray:
    return np.array([stoi[c] for c in text], dtype=np.int64)


def sample_batch(ids: np.ndarray, context: int, batch_size: int,
                 rng: np.random.Generator) -> torch.Tensor:
    """Random contiguous windows."""
    starts = rng.integers(0, len(ids) - context - 1, size=batch_size)
    batch = np.stack([ids[s:s + context] for s in starts])
    return torch.from_numpy(batch).long()


# ---------------------------------------------------------------------------
# Tiny transformer (causal or bidirectional, controlled by a mask)
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.n_heads = cfg.n_heads
        self.d_head  = cfg.d_model // cfg.n_heads

    def attend(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)                    # (B, T, H, Dh)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if attn_mask is not None:
            scores = scores + attn_mask                # (1, 1, T, T) broadcast
        weights = F.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.attend(self.ln1(x), attn_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context, cfg.d_model)
        self.blocks  = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f    = nn.LayerNorm(cfg.d_model)
        self.head    = nn.Linear(cfg.d_model, vocab_size, bias=False)
        # Cache the causal mask
        causal = torch.full((cfg.context, cfg.context), -1e9)
        causal = torch.triu(causal, diagonal=1)
        self.register_buffer("causal_mask", causal[None, None, :, :])

    def hidden(self, ids: torch.Tensor, causal: bool) -> torch.Tensor:
        B, T = ids.shape
        positions = torch.arange(T, device=ids.device)[None, :].expand(B, T)
        x = self.tok_emb(ids) + self.pos_emb(positions)
        attn_mask = self.causal_mask[:, :, :T, :T] if causal else None
        for block in self.blocks:
            x = block(x, attn_mask)
        return self.ln_f(x)

    def forward(self, ids: torch.Tensor, causal: bool = False) -> torch.Tensor:
        return self.head(self.hidden(ids, causal))


# ---------------------------------------------------------------------------
# Masking strategies
# ---------------------------------------------------------------------------

def single_token_mask(ids: torch.Tensor, mask_id: int, mask_frac: float,
                      vocab_size: int, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """BERT 80/10/10 mixture. Returns (corrupted_ids, mask_positions)."""
    B, T = ids.shape
    probs  = torch.rand(B, T, generator=rng)
    chosen = probs < mask_frac                                  # bool mask of positions to predict
    sub    = torch.rand(B, T, generator=rng)
    is_mask  = chosen & (sub < 0.8)
    is_rand  = chosen & (sub >= 0.8) & (sub < 0.9)
    corrupted = ids.clone()
    corrupted[is_mask] = mask_id
    rand_ids = torch.randint(1, vocab_size, ids.shape, generator=rng)  # avoid mask_id (0)
    corrupted[is_rand] = rand_ids[is_rand]
    return corrupted, chosen


def span_mask(ids: torch.Tensor, mask_id: int, mask_frac: float, mean_span: float,
              vocab_size: int, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Geometric-length contiguous spans replaced with [MASK]."""
    B, T = ids.shape
    target_n = int(mask_frac * T)
    chosen = torch.zeros(B, T, dtype=torch.bool)
    p = 1.0 / mean_span
    for b in range(B):
        masked = 0
        attempts = 0
        while masked < target_n and attempts < 50:
            attempts += 1
            length = max(1, int(torch.distributions.Geometric(p).sample().item()) + 1)
            length = min(length, target_n - masked)
            start = int(torch.randint(0, T - length, (1,), generator=rng).item())
            if chosen[b, start:start + length].any():
                continue
            chosen[b, start:start + length] = True
            masked += length
    corrupted = ids.clone()
    corrupted[chosen] = mask_id
    return corrupted, chosen


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_masked(model: TinyLM, ids: np.ndarray, vocab_size: int,
                 cfg: Config, mask_strategy: str, tag: str) -> list[float]:
    rng_np  = np.random.default_rng(SEED)
    rng_pt  = torch.Generator(); rng_pt.manual_seed(SEED)
    opt     = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    losses  = []
    for step in range(cfg.n_steps):
        batch = sample_batch(ids, cfg.context, cfg.batch_size, rng_np)
        if mask_strategy == "single":
            corrupted, mask_pos = single_token_mask(batch, 0, cfg.mask_frac, vocab_size, rng_pt)
        elif mask_strategy == "span":
            corrupted, mask_pos = span_mask(batch, 0, cfg.mask_frac, cfg.span_mean, vocab_size, rng_pt)
        else:
            raise ValueError(mask_strategy)
        logits = model(corrupted, causal=False)
        # Cross-entropy at masked positions only
        flat_logits = logits[mask_pos]
        flat_targets = batch[mask_pos]
        loss = F.cross_entropy(flat_logits, flat_targets)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss))
        if (step + 1) % 100 == 0:
            print(f"  [{tag}] step {step+1:4d}/{cfg.n_steps}  loss={np.mean(losses[-100:]):.4f}")
    return losses


def train_causal(model: TinyLM, ids: np.ndarray, cfg: Config, tag: str) -> list[float]:
    rng_np = np.random.default_rng(SEED)
    opt    = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    losses = []
    for step in range(cfg.n_steps):
        batch = sample_batch(ids, cfg.context + 1, cfg.batch_size, rng_np)
        x, y  = batch[:, :-1], batch[:, 1:]
        logits = model(x, causal=True)
        loss   = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss))
        if (step + 1) % 100 == 0:
            print(f"  [{tag}] step {step+1:4d}/{cfg.n_steps}  loss={np.mean(losses[-100:]):.4f}")
    return losses


def train_jepa(student: TinyLM, teacher: TinyLM, ids: np.ndarray,
               vocab_size: int, cfg: Config, tag: str) -> list[float]:
    """Predict the teacher's hidden states at masked positions in feature space.

    Teacher sees the unmasked input; student sees the masked input. Loss is MSE
    between student hidden states and teacher hidden states at masked positions.
    Teacher is updated as an EMA of the student.
    """
    rng_np = np.random.default_rng(SEED)
    rng_pt = torch.Generator(); rng_pt.manual_seed(SEED)
    opt    = torch.optim.AdamW(student.parameters(), lr=cfg.lr)
    for p in teacher.parameters():
        p.requires_grad_(False)
    losses = []
    for step in range(cfg.n_steps):
        batch = sample_batch(ids, cfg.context, cfg.batch_size, rng_np)
        corrupted, mask_pos = single_token_mask(batch, 0, cfg.mask_frac, vocab_size, rng_pt)
        with torch.no_grad():
            target_h = teacher.hidden(batch, causal=False)
        student_h = student.hidden(corrupted, causal=False)
        # MSE at masked positions only
        diff = (student_h - target_h)[mask_pos]
        loss = (diff ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        # EMA update on teacher
        with torch.no_grad():
            for tp, sp in zip(teacher.parameters(), student.parameters()):
                tp.data.mul_(cfg.ema_m).add_(sp.data, alpha=1 - cfg.ema_m)
        losses.append(float(loss))
        if (step + 1) % 100 == 0:
            print(f"  [{tag}] step {step+1:4d}/{cfg.n_steps}  loss={np.mean(losses[-100:]):.6f}")
    return losses


# ---------------------------------------------------------------------------
# Save weights as a flat .npz
# ---------------------------------------------------------------------------

def model_to_dict(model: TinyLM) -> dict[str, np.ndarray]:
    """Save weights with names that mirror the notebook's numpy implementation."""
    out = {}
    out["tok_emb"] = model.tok_emb.weight.detach().cpu().numpy().astype(np.float32)
    out["pos_emb"] = model.pos_emb.weight.detach().cpu().numpy().astype(np.float32)
    for i, blk in enumerate(model.blocks):
        out[f"blk{i}_ln1_w"] = blk.ln1.weight.detach().cpu().numpy().astype(np.float32)
        out[f"blk{i}_ln1_b"] = blk.ln1.bias.detach().cpu().numpy().astype(np.float32)
        out[f"blk{i}_qkv_w"] = blk.qkv.weight.detach().cpu().numpy().astype(np.float32)
        out[f"blk{i}_proj_w"] = blk.proj.weight.detach().cpu().numpy().astype(np.float32)
        out[f"blk{i}_ln2_w"] = blk.ln2.weight.detach().cpu().numpy().astype(np.float32)
        out[f"blk{i}_ln2_b"] = blk.ln2.bias.detach().cpu().numpy().astype(np.float32)
        out[f"blk{i}_ffn1_w"] = blk.ffn[0].weight.detach().cpu().numpy().astype(np.float32)
        out[f"blk{i}_ffn1_b"] = blk.ffn[0].bias.detach().cpu().numpy().astype(np.float32)
        out[f"blk{i}_ffn2_w"] = blk.ffn[2].weight.detach().cpu().numpy().astype(np.float32)
        out[f"blk{i}_ffn2_b"] = blk.ffn[2].bias.detach().cpu().numpy().astype(np.float32)
    out["ln_f_w"] = model.ln_f.weight.detach().cpu().numpy().astype(np.float32)
    out["ln_f_b"] = model.ln_f.bias.detach().cpu().numpy().astype(np.float32)
    out["head_w"] = model.head.weight.detach().cpu().numpy().astype(np.float32)
    out["__meta__"] = np.array([model.cfg.d_model, model.cfg.n_heads,
                                 model.cfg.context, model.cfg.n_layers], dtype=np.int32)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(SEED)

    cfg  = Config()
    text = load_corpus(HERE / "die_raeuber.txt")
    stoi, itos = build_vocab(text)
    vocab_size = len(stoi)
    print(f"Corpus: {len(text)} chars  Vocab: {vocab_size}  (slot 0 = [MASK])")

    ids = encode(text, stoi)

    # Save vocab
    (HERE / "vocab.json").write_text(
        json.dumps({"stoi": stoi, "itos": {str(k): v for k, v in itos.items()},
                    "mask_id": 0}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    curves = {}

    # 1. Bidirectional masked LM (BERT)
    print("\n=== Bidirectional masked LM ===")
    bd = TinyLM(vocab_size, cfg)
    curves["bidirectional"] = train_masked(bd, ids, vocab_size, cfg, "single", "bd-lm")
    np.savez(HERE / "bidirectional_lm_weights.npz", **model_to_dict(bd))

    # 2. Span-masked LM (SpanBERT/T5-corruption)
    print("\n=== Span-masked LM ===")
    sp = TinyLM(vocab_size, cfg)
    curves["span"] = train_masked(sp, ids, vocab_size, cfg, "span", "span-lm")
    np.savez(HERE / "span_lm_weights.npz", **model_to_dict(sp))

    # 3. Causal LM (GPT)
    print("\n=== Causal LM ===")
    cs = TinyLM(vocab_size, cfg)
    curves["causal"] = train_causal(cs, ids, cfg, "causal-lm")
    np.savez(HERE / "causal_lm_weights.npz", **model_to_dict(cs))

    # 4. JEPA: shared-encoder student/teacher in feature space
    print("\n=== JEPA (text) ===")
    student = TinyLM(vocab_size, cfg)
    teacher = TinyLM(vocab_size, cfg)
    teacher.load_state_dict(student.state_dict())     # init copy
    curves["jepa"] = train_jepa(student, teacher, ids, vocab_size, cfg, "jepa")
    np.savez(HERE / "jepa_weights.npz", **model_to_dict(student))

    # Loss curves for the notebook to plot
    np.savez(HERE / "training_curves.npz",
             bidirectional=np.array(curves["bidirectional"], dtype=np.float32),
             span=np.array(curves["span"], dtype=np.float32),
             causal=np.array(curves["causal"], dtype=np.float32),
             jepa=np.array(curves["jepa"], dtype=np.float32))

    print("\nAll precomputed weights saved alongside this script.")


if __name__ == "__main__":
    main()
