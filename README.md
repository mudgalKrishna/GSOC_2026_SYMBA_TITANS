# SYMBA — Titans + MiRAS: Feynman Amplitude to Squared Amplitude

> Sequence-to-sequence symbolic regression on Feynman diagram amplitudes using four Titans neural memory architectures enhanced with MiRAS (Mixture of Recurrent and Attentive Sequences) routing.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square)](https://pytorch.org)
[![Physics](https://img.shields.io/badge/Physics-QED%20%7C%20QCD-purple?style=flat-square)](#)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](#license)

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
  - [1. Preprocessing & Tokenization](#1-preprocessing--tokenization)
  - [2. Pretraining](#2-pretraining)
  - [3. Fine-tuning](#3-fine-tuning)
  - [4. Evaluation & Ablation](#4-evaluation--ablation)
- [Architectures](#architectures)
- [Results](#results)
- [Configuration](#configuration)
- [Requirements](#requirements)
- [Citation](#citation)

---

## Overview

This project implements a physics-aware seq2seq pipeline that learns the mapping:

```
Feynman amplitude  →  squared amplitude
```

across two quantum field theory regimes:

| Model | Domain | Sequences | Avg source length |
|-------|--------|-----------|-------------------|
| **QED** | Quantum Electrodynamics | 360 | ~127 tokens |
| **QCD** | Quantum Chromodynamics | 234 | ~483 tokens |

Four distinct Titans encoder architectures (MAC, MAG, MAL, LMM) are each equipped with a **MiRAS soft router** — a learned per-token gate that dynamically blends the neural memory path and the attention path for every token. This yields 8 fine-tuned models (4 architectures × 2 physics models) that are compared in a full ablation study.

---

## Project Structure

```
symba-titans-miras/
│
├── config.yaml                        # All hyperparameters and paths
│
├── data/
│   ├── raw/                           # Raw .txt files (QED and QCD)
│   └── processed/
│       └── processed_data.pkl         # Preprocessed output (tokenized + encoded)
│
├── ckpt/                              # Saved model checkpoints
│   ├── pretrained_encoder.pth         # Phase 1 encoder LM checkpoint
│   ├── pretrained_tgt_embed.pth       # Phase 2 decoder embed checkpoint
│   ├── MAC_QEDbest.pth
│   ├── MAC_QCDbest.pth
│   ├── MAG_QEDbest.pth
│   ├── MAG_QCDbest.pth
│   ├── MAL_QEDbest.pth
│   ├── MAL_QCDbest.pth
│   ├── LMM_QEDbest.pth
│   └── LMM_QCDbest.pth
│
├── notebooks/
│   ├── 01_preprocessing.ipynb         # Task 1.2 — data preprocessing & tokenization
│   ├── 02_pretraining.ipynb           # Phase 1 + 2 shared pretraining
│   ├── 03_finetune_MAC_QED.ipynb      # Fine-tuning: MAC on QED
│   ├── 03_finetune_MAC_QCD.ipynb      # Fine-tuning: MAC on QCD
│   ├── 03_finetune_MAG_QED.ipynb      # Fine-tuning: MAG on QED
│   ├── 03_finetune_MAG_QCD.ipynb      # Fine-tuning: MAG on QCD
│   ├── 03_finetune_MAL_QED.ipynb      # Fine-tuning: MAL on QED
│   ├── 03_finetune_MAL_QCD.ipynb      # Fine-tuning: MAL on QCD
│   ├── 03_finetune_LMM_QED.ipynb      # Fine-tuning: LMM on QED
│   ├── 03_finetune_LMM_QCD.ipynb      # Fine-tuning: LMM on QCD
│   └── 04_evaluation_ablation.ipynb   # Final ablation study & results
│
├── docs/
│   ├── SYMBA_Preprocessing_Documentation.docx
│   └── SYMBA_Training_Setup_Documentation.docx
│
└── eval_results/
    ├── ablation_results.csv
    ├── ablation_results_pretty.csv
    ├── all_results.json
    ├── ablation_bar_charts.png
    ├── radar_chart.png
    ├── heatmap.png
    ├── curves_QED.png
    └── curves_QCD.png
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install einops editdistance tqdm pandas numpy scikit-learn matplotlib
```

### 2. Place raw data

Copy the 17 SYMBA `.txt` files into `data/raw/`. Files should be named with `QED` or `QCD` in the filename.

### 3. Run preprocessing

Open and run `notebooks/01_preprocessing.ipynb`. This produces `data/processed/processed_data.pkl`.

### 4. Run pretraining

Open and run `notebooks/02_pretraining.ipynb`. This produces:
- `ckpt/pretrained_encoder.pth`
- `ckpt/pretrained_tgt_embed.pth`

### 5. Run fine-tuning (8 notebooks)

Run each of the 8 fine-tuning notebooks. Each notebook only requires changing two lines at the top:

```python
ARCH  = 'MAC'   # MAC | MAG | MAL | LMM
LABEL = 'QED'   # QED | QCD
```

### 6. Run evaluation

Open and run `notebooks/04_evaluation_ablation.ipynb` to generate the full ablation study, plots, and qualitative examples.

---

## Dataset

### Format

Each raw `.txt` file contains one interaction per line:

```
Interaction : Feynman_diagram : amplitude : squared_amplitude
```

### Preprocessing pipeline

| Stage | Description |
|-------|-------------|
| **File parsing** | Split on ` : ` delimiter; rejoin squared amplitude if it contains the delimiter |
| **Index normalization** | Replace arbitrary numeric suffixes (`tau_576`) with canonical pool names (`INDEX_0`) |
| **Momentum protection** | `p_1`, `s_12` etc. are protected from normalization — they carry physical meaning |
| **Tokenization** | Word-level; each physics atom (operator, symbol, normalized index) becomes one token |
| **Vocabulary** | 4 separate vocabs: QED-src, QED-tgt, QCD-src, QCD-tgt — built from training split only |
| **Encoding** | `[BOS] + token_ids + [EOS]`; truncated at 97th-percentile length |
| **Half-splits** | Amplitude and squared amplitude token lists split at midpoint for pretraining |

### Special tokens

| Token | Index | Purpose |
|-------|-------|---------|
| `<PAD>` | 0 | Batch padding |
| `<UNK>` | 1 | Unknown tokens |
| `<BOS>` | 2 | Begin of sequence |
| `<EOS>` | 3 | End of sequence |
| `<SEP>` | 4 | Expression separator |
| `<TERM0>` | 5 | Term boundary |
| `<TERM1>` | 6 | Term boundary variant |

---

## Pipeline

### 1. Preprocessing & Tokenization

**Notebook:** `01_preprocessing.ipynb`

The `AmplitudeTokenizer` normalizes all raw expressions before tokenization:

```python
tokenizer = AmplitudeTokenizer(index_pool_size=1000, to_replace=True)

# Source sequences (amplitudes)
amp_tokens = tokenizer.tokenize_amplitude(amplitude_str)

# Target sequences (squared amplitudes)
sq_tokens  = tokenizer.tokenize_squared(squared_amplitude_str)
```

Key normalization: arbitrary tensor indices like `gamma_576` → `INDEX_0`, spinor labels like `i_12345` → `PINDEX_0`, while physical labels `p_1`, `s_12`, `m_b`, `reg_prop` are preserved exactly.

---

### 2. Pretraining

**Notebook:** `02_pretraining.ipynb`

Two-phase pretraining before any fine-tuning:

**Phase 1 — Encoder language model:**
- Data: amplitude token halves (H1 + H2) from QED + QCD combined
- Objective: next-token causal LM (predict next token given previous tokens)
- Architecture: temporary MAC encoder with LM head
- Epochs: 30 | LR: 3e-4 | Effective batch: 16

**Phase 2 — Decoder embedding pretext:**
- Data: squared amplitude halves (SH1 → SH2)
- Objective: seq2seq — predict second half from first half
- Why: forces decoder embedding to learn Mandelstam polynomial structure before fine-tuning
- Epochs: 30 | LR: 3e-4 | Effective batch: 16

Both phases save to `ckpt/` and are loaded by all 8 fine-tuning runs.

---

### 3. Fine-tuning

**Notebooks:** `03_finetune_{ARCH}_{LABEL}.ipynb` (8 total)

Each fine-tuning run:

1. Builds model with **per-label vocabulary sizes** (not combined)
2. Loads pretrained encoder (shape-filtered — non-embedding weights transfer; embedding reinitialised)
3. Loads pretrained `tgt_embed` (shape-filtered)
4. Fine-tunes end-to-end: `amplitude → squared amplitude`
5. Saves best checkpoint by validation loss

```
Training: 120 epochs | batch=4 | grad_accum=4 → eff_batch=16
LR: 3e-4 cosine decay to 1e-6 | warmup: 10% | label_smoothing: 0.1
Mixed precision: bfloat16 | grad_clip: 1.0
```

---

### 4. Evaluation & Ablation

**Notebook:** `04_evaluation_ablation.ipynb`

Loads all 8 checkpoints and evaluates on held-out test sets. Metrics:

| Metric | Description |
|--------|-------------|
| CE Loss | Cross-entropy on test set |
| PPL | `exp(per-token NLL)` — lower is better |
| Token Accuracy | % of individual tokens predicted correctly |
| Sequence Accuracy | % of complete expressions predicted exactly |
| Mean Edit Distance | Average token-level edit distance to reference |
| Exact Matches | Count of completely correct predictions |

> **Evaluation note:** NeuralLTM performs live memory updates during evaluation. `torch.no_grad()` is NOT used — only output logits are `.detach()`-ed before metric computation.

---

## Architectures

All four architectures share the same NeuralLTM, FFN (SwiGLU), RMSNorm, RoPE, and MiRAS router. Only the wiring between memory and attention differs.

### MAC — Memory as Context

Memory is retrieved and **prepended as a prefix** before attention runs. The attention head attends over `[persistent_tokens ‖ memory ‖ input]` simultaneously.

```
x → retrieve(memory) → norm → prefix
x → norm1 → CausalMHA(prefix + x) → MiRAS(mem, attn) → update(memory) → FFN → out
```

Best for: long-range retrieval tasks; QED (full causal attention tractable for short sequences).

---

### MAG — Memory as Gate

Sliding-window attention and NeuralLTM memory run **in parallel** on the same input. The MiRAS router learns a per-token soft blend.

```
x → SlidingWindowMHA(norm1(x))  ─┐
x → NeuralLTM(x, state)         ─┤→ MiRAS gate → FFN → out
                                  └─ (per-token dynamic blend)
```

Best for: adaptive routing; sequences where local and global patterns are separable.

---

### MAL — Memory as Layer

Memory **preprocesses the input** first. The MiRAS router blends memory output with the original input. Sliding-window attention then runs on the blended result.

```
x → NeuralLTM(x, state) → MiRAS(mem, x) → x_hat
x_hat → norm1 → SlidingWindowMHA → FFN → out
```

Best for: structured inputs where memory can normalize the representation before local processing.

---

### LMM — Long-term Memory Module

**No attention at all.** The MiRAS router blends memory output with the raw input identity. All sequence modeling comes from NeuralLTM's gradient-descent memory updates.

```
x → norm1 → NeuralLTM(x, state) → MiRAS(mem, x_identity) → FFN → out
```

Best for: very long sequences (QCD); lowest compute cost; O(n) vs O(n²) attention.

---

### Architecture Comparison

| Property | MAC | MAG | MAL | LMM |
|----------|-----|-----|-----|-----|
| Attention type | CausalMHA | SlidingWindow | SlidingWindow | None |
| Memory–attention order | Memory first | Parallel | Memory first | Memory only |
| Persistent tokens | Yes (4) | No | No | No |
| MiRAS 'attn' path | `attn_out` | `attn_out` | `attn_out` | identity `x` |
| Total params | ~1.64M | ~1.71M | ~1.64M | ~1.51M |
| Relative compute | High | Med-High | Medium | Low |

---

## Results

*Fill in your ablation study results here after running `04_evaluation_ablation.ipynb`.*

### QED Test Set

| Architecture | CE Loss | PPL | Token Acc | Seq Acc | Mean ED |
|-------------|---------|-----|-----------|---------|---------|
| MAC | — | — | — | — | — |
| MAG | — | — | — | — | — |
| MAL | — | — | — | — | — |
| LMM | — | — | — | — | — |

### QCD Test Set

| Architecture | CE Loss | PPL | Token Acc | Seq Acc | Mean ED |
|-------------|---------|-----|-----------|---------|---------|
| MAC | — | — | — | — | — |
| MAG | — | — | — | — | — |
| MAL | — | — | — | — | — |
| LMM | — | — | — | — | — |

---

## Configuration

All hyperparameters live in `config.yaml`. See the file for full documentation of every field.

```yaml
# Quick reference — key values
model:
  dim: 128
  num_heads: 8
  num_encoder_layers: 2
  num_decoder_layers: 2

training:
  epochs: 120
  batch_size: 4
  grad_accum_steps: 4       # effective batch = 16
  optimizer_lr: 3.0e-4
  pretrain_epochs: 30
```

---

## Requirements

```
python >= 3.10
torch >= 2.0
numpy
pandas
scikit-learn
matplotlib
einops
editdistance
tqdm
```

GPU: NVIDIA T4 (15.6 GB VRAM) or equivalent. All models are under 2M parameters and fit comfortably.

---

## Citation

If you use this code or methodology, please cite:

```bibtex
@misc{symba_titans_miras_2025,
  title   = {SYMBA Titans+MiRAS: Neural Memory Architectures for Feynman Amplitude Symbolic Regression},
  year    = {2025},
  note    = {Based on Titans (arXiv:2501.00663) and SYMBA dataset}
}
```

**References:**
- Titans: Learning to Memorize at Test Time — arXiv:2501.00663
- SYMBA: Symbolic Mathematics Benchmark for Amplitudes

---

## License

MIT License — see `LICENSE` for details.
