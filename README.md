# SYMBA — Titans + MiRAS:  Amplitude → Squared Amplitude

> Sequence-to-sequence amplitude to squared amplitude
> using four Titans neural memory architectures enhanced with MiRAS routing.
> Built for GSoC 2026 — ML4Sci organization.

---

## Repository Structure
```
GSOC_2026_SYMBA_TITANS/
│
├── SYMBA_preprocessing_tokenization_Common...ipynb   # Task 1.2 — preprocessing & tokenization
├── 2.4Specifictask_pretrain.ipynb                    # Phase 1 + 2 shared pretraining
├── 2.4Specifictask_Titan_Finetuning(QCD+QED).ipynb   # All 8 fine-tuning runs (4 arch × 2 models)
├── README.md
└── LICENSE
```

---

## What this project does

Maps Feynman amplitude expressions to their squared amplitudes for two physics models:

| Model | Domain | Avg sequence length |
|-------|--------|-------------------|
| QED | Quantum Electrodynamics | ~127 tokens |
| QCD | Quantum Chromodynamics | ~483 tokens |

---

## How to run

### 1. Preprocessing
Open and run `SYMBA_preprocessing_tokenization_Common...ipynb`
- Parses all 17 raw SYMBA `.txt` files
- Normalizes tensor indices, tokenizes expressions, builds vocabularies
- Saves `processed_data.pkl`

### 2. Pretraining
Open and run `2.4Specifictask_pretrain.ipynb`
- Phase 1: encoder LM pretraining on amplitude token halves
- Phase 2: decoder embedding pretraining on squared amplitude halves
- Saves `pretrained_encoder.pth` and `pretrained_tgt_embed.pth`

### 3. Fine-tuning & Evaluation
Open and run `2.4Specifictask_Titan_Finetuning(QCD+QED).ipynb`
- Fine-tunes all 4 architectures on both QED and QCD (8 runs total)
- Runs ablation study and generates evaluation plots

---

## The Four Architectures

| Architecture | Key idea | Attention type |
|---|---|---|
| **MAC** | Memory prepended as context prefix before attention | Full causal |
| **MAG** | Memory and attention run in parallel, blended by MiRAS gate | Sliding window |
| **MAL** | Memory preprocesses input first, then attention runs | Sliding window |
| **LMM** | No attention at all — pure neural memory only | None |

All four use a **MiRAS router** — a learned per-token gate that dynamically
blends the memory path and attention path for every token.

---

## Results (Test Set)

| Arch | Label | CE Loss | PPL | Token Acc | Seq Acc | Exact |
|------|-------|---------|-----|-----------|---------|-------|
| MAC | QED | 0.0896 | 1.0936 | 99.58% | 66.67% | 24/36 |
| MAG | QED | 0.0963 | 1.1009 | 99.22% | 47.22% | 17/36 |
| MAL | QED | 0.1018 | 1.1070 | 99.07% | 44.44% | 16/36 |
| LMM | QED | 0.1105 | 1.1164 | 98.89% | 41.67% | 15/36 |
| MAC | QCD | 0.6407 | 1.9515 | 84.57% | 0.00% | 0/24 |
| MAG | QCD | 0.6360 | 1.9498 | 84.35% | 0.00% | 0/24 |
| MAL | QCD | 0.4065 | 1.5426 | 90.30% | 8.33% | 2/24 |
| LMM | QCD | 0.6197 | 1.9144 | 83.81% | 0.00% | 0/24 |

---

## Requirements
```
torch >= 2.0
numpy, pandas, scikit-learn
matplotlib, einops, editdistance, tqdm
```

GPU recommended — tested on Colab T4 (15.6 GB VRAM).

---

## References

- Titans: Learning to Memorize at Test Time — arXiv:2501.00663
- SYMBA: Symbolic computation of squared amplitudes — Alnuqaydan et al. 2023
- MiRAS: It's All Connected — arXiv:2504.13173

---

## License

MIT — see `LICENSE` for details.s.
