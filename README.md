# DualMark

**A Novel Dual Watermarking Approach for Large Language Models**

> Zihao Qiang, Jifei Hao, Jipeng Qiang, Yi Zhu, Chaowei Zhang, Yan Liu, Wei Li
> *Information Processing & Management*, 63 (2026) 104707 · [DOI: 10.1016/j.ipm.2026.104707](https://doi.org/10.1016/j.ipm.2026.104707)

Official implementation of the **DualMark** framework, an ensemble-inspired **dual green–red watermarking** scheme for detecting machine-generated text. The full paper is provided in [`2026-dualwark.pdf`](2026-dualwark.pdf).

---

## Overview

Watermarking embeds imperceptible, machine-detectable patterns into generated text, and has become a leading approach for detecting LLM-generated content. The dominant *green–red* strategy (e.g., KGW) dynamically partitions the vocabulary into a *green list* and a *red list* at each decoding step and biases sampling toward green tokens. Detection then measures the excess proportion of green tokens (a `z`-score).

However, green–red methods remain vulnerable in two ways:

1. **Token replacement** — paraphrasing or synonym substitution replaces green tokens, shrinking the detectable signal.
2. **Context sensitivity** — the green/red partition depends on preceding tokens, so editing those tokens reconfigures the partition and breaks alignment with the original watermark.

**DualMark** addresses both problems by embedding **two independent green–red watermarks** that use **separate context windows and partitioning strategies**, then combining them in an ensemble:

- **Tiered bias at generation** — the strongest bias is applied to tokens in the *intersection* of both green lists, while a standard bias is applied to tokens unique to each list.
- **OR fusion at detection** — the text is flagged as watermarked if *either* watermark yields a statistically significant signal.

Because the two watermarks are independent, an attacker must suppress *both* signals simultaneously, which requires modifying a substantially larger fraction of the text. The design incurs only minimal overhead (one extra green–red list per step).

---

## Key Contributions

- **Ensemble of green–red watermarks.** First method to integrate multiple *overlapping* green–red partitions, specifically targeting the context-sensitivity limitations of the green–red family (unlike heterogeneous dual schemes such as Duwak).
- **Robustness.** Improves detection **AUROC by over 11%** under severe word-deletion attacks and **doubles the true-positive rate (TP@1%)** against adaptive baselines under heavy paraphrasing.
- **Minimal overhead.** Adds only ~5% generation latency (e.g., KGW+TSW adds 0.28 ms/token over KGW on a Qwen3-8B backbone).
- **Flexible and extensible.** Two branches can be two different methods (KGW + TSW) or two instances of the same family, reusing existing watermarking procedures.
- **Preserves text quality.** Comparable perplexity, coherence, and semantic loss to the best single-watermark baselines.

---

## Method

At each decoding step `i`, two independent green–red partitions are constructed from different context windows:

```
G1, R1 = Method1(V, t_{i-1})
G2, R2 = Method2(V, t_{i-2})
```

where `Method1` / `Method2` can be two different watermarking methods or the same method. The partitions are fully decoupled and can be computed in parallel.

**Generation (tiered bias).** With `G∩ = G1 ∩ G2`, the logits are adjusted as:

```
l̃_k = l_k + δ1 + δ2,   if k ∈ G∩
l̃_k = l_k + δ1,        if k ∈ G1 \ G∩
l̃_k = l_k + δ2,        if k ∈ G2 \ G∩
l̃_k = l_k,             otherwise
```

**Detection (OR fusion).** The detector reconstructs both partitions and computes independent `z`-scores:

```
z_j = (N_j − γ_j·T) / √(γ_j·(1−γ_j)·T),   j ∈ {1, 2}
```

The final decision is:

```
watermarked ⇔ (z1 > τ1) OR (z2 > τ2)
```

**Why it works.** For green-list ratio `γ = 0.5`, a random replacement erases a single watermark with probability `1 − γ = 0.5`, but erases *both* DualMark watermarks with probability only `1 − (2γ − γ²) = 0.25` — a 75% failure rate per replacement attempt. Context-window decoupling further ensures that editing one context token disrupts only one watermark's partitions, preventing error propagation across the whole signal.

---

## Implemented Combinations

DualMark composes two base green–red methods. This repository provides the three pairwise combinations evaluated in the paper (base methods: **KGW**, **UPV**, **TSW**):

| Algorithm | Composition | Config |
|-----------|-------------|--------|
| `Twins`   | KGW + TSW (token-specific watermarking) | [`config/Twins.json`](config/Twins.json) |
| `Twint`   | KGW + UPV (unforgeable public-verifiable) | [`config/Twint.json`](config/Twint.json) |
| `Twinu`   | TSW + UPV | [`config/Twinu.json`](config/Twinu.json) |

Each config file holds two halves of parameters (suffixed `_1` / `_2`), one per watermark branch (e.g., `gamma_1` / `gamma_2`, `delta_1` / `delta_2`).

---

## Repository Structure

```
DualMark/
├── 2026-dualwark.pdf          # The paper
├── demo.py                    # End-to-end embed + detect demo
├── requirements.txt           # Python dependencies
├── config/
│   ├── Twins.json             # KGW + TSW
│   ├── Twint.json             # KGW + UPV
│   └── Twinu.json             # TSW + UPV
├── watermark/
│   ├── twins/                 # KGW + TSW implementation
│   ├── twint/                 # KGW + UPV implementation
│   └── twinu/                 # TSW + UPV implementation
├── utils/
│   ├── base_attacker.py       # Attacker factory (AutoAttacker)
│   └── attacker/
│       ├── word_subs.py       # Word-substitution attack (BERT)
│       ├── word_delete.py     # Word-deletion attack
│       └── dipper.py          # Document-level paraphrase attack (Dipper/T5)
└── result/
    ├── baseline/              # Single-method & pairwise baselines
    └── dualmark/              # DualMark (Twins/Twint/Twinu) outputs
```

---

## Requirements & Installation

The dependencies are listed in [`requirements.txt`](requirements.txt). Install them with:

```bash
pip install -r requirements.txt
```

### Base Code

The code is written as a plugin for the open-source watermarking toolkit **[MarkLLM][MarkLLM-code]** and reuses its framework (e.g., `watermark.base`, `watermark.auto_watermark`, `watermark.ts`, `watermark.upv`, `utils.transformers_config`). To run it:

1. Clone MarkLLM and set up its environment.
2. Copy the `watermark/`, `config/`, and `utils/` folders from this repository into the MarkLLM project root.
3. Register the three algorithms in MarkLLM's `watermark/auto_watermark.py` by adding `Twins`, `Twint`, and `Twinu` to `WATERMARK_MAPPING_NAMES` and to the class/config mapping.

> The base methods use pretrained components from their respective repositories: [TSW][TSW-code] (the TSW/TS checkpoint) and [UPV][UPV-code] (the UPV generator/detector models). Fill in the `ckpt_path` / `generator_model_name` / `detector_model_name` fields in the config files with your local model paths.

### Base Model

The paper uses **Qwen3-8B** for watermarked generation (`temperature = 0.7`), **OPT-6.7b** for perplexity, and **bert-base-cased** for word-substitution attacks. You can use any Hugging Face causal LM by setting `generate_model_path`.

---

## Usage

### Embedding & Detecting a Watermark

The example below works for any of `Twins`, `Twint`, or `Twinu` (change `watermark_name` accordingly).

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from watermark.auto_watermark import AutoWatermark, WATERMARK_MAPPING_NAMES
from utils.transformers_config import TransformersConfig

watermark_name = 'Twins'          # 'Twins' | 'Twint' | 'Twinu'

assert watermark_name in WATERMARK_MAPPING_NAMES, f"{watermark_name} not in WATERMARK_MAPPING_NAMES"

device = "cuda:3" if torch.cuda.is_available() else "cpu"
generate_model_path = "facebook/opt-1.3b"

transformers_config = TransformersConfig(
    model=AutoModelForCausalLM.from_pretrained(generate_model_path).to(device),
    tokenizer=AutoTokenizer.from_pretrained(generate_model_path),
    device=device,
    max_new_tokens=200,
    min_length=230,
    do_sample=True,
    no_repeat_ngram_size=4,
)

watermark_model = AutoWatermark.load(
    watermark_name,
    transformers_config=transformers_config,
    algorithm_config=f'./config/{watermark_name}.json',
)

# Embed the watermark
prompt = "Please input some text to watermark model."
watermarked_text = watermark_model.generate_watermarked_text(prompt)

# Detect the watermark
result = watermark_model.detect_watermark(watermarked_text)
print(result)
```

A ready-to-run script is provided in [`demo.py`](demo.py).

### Running Adversarial Attacks

The evaluation applies word-level and document-level attacks to the watermarked text, then re-runs detection. The attackers are exposed via `AutoAttacker` in [`utils/base_attacker.py`](utils/base_attacker.py):

```python
from utils.base_attacker import AutoAttacker

attacker = AutoAttacker.load_attacker('WordSubs')    # word substitution
# attacker = AutoAttacker.load_attacker('WordDelete') # word deletion
# attacker = AutoAttacker.load_attacker('Dipper')     # document-level paraphrase

attacked = attacker.attack(watermarked_text)
detect_result = watermark_model.detect_watermark(attacked.text_attacked)
```

---

## Experimental Results

Experiments use 1,000 samples from the C4 dataset (via MarkLLM), each 200 ± 20 tokens, with half watermarked and half unwatermarked. Each run is repeated 10 times (rounding off the best and worst) and averaged. Detection is measured by **AUROC** and **TP@1% / TP@5%** (true-positive rate at fixed 1%/5% false-positive rates).

**Clean (no-attack) setting.** All methods — single watermarks and DualMark combinations — separate watermarked from unwatermarked text essentially perfectly (AUROC ≈ 0.99–1.00, TP@1%/TP@5% ≈ 0.99–1.00).

**Word-level attacks** (substitution and deletion, attack rates 0.1–0.5):

- Detection degrades as the attack rate increases for all methods, but DualMark combinations consistently retain substantially higher scores than single methods, especially at larger attack rates.
- Under a 0.5 substitution attack, `KGW+TSW` reaches an AUROC of **0.87**, outperforming the heterogeneous dual baseline Duwak (0.84).
- Word substitution is more damaging to high-precision detection (TP@1%) than deletion, since substitutions actively replace signal-carrying tokens.

**Document-level attacks** (GPT-4o-mini paraphrase, probabilities 0.2–1.0):

- DualMark combinations lead in detectability across all attack probabilities; `KGW+TSW` is the most robust across almost all metrics and attack levels.
- Under full paraphrasing (rate 1.0) of rewritten sentences, `KGW+TSW` maintains a TP@1% of **0.46** — nearly double the adaptive TSW baseline (0.23) and above KGW (0.44).

**Text quality.** DualMark is comparable to the best existing methods on perplexity and semantic loss; the ensemble incurs only a slight perplexity cost in exchange for the large robustness gains.

**Latency.** On a Qwen3-8B backbone, DualMark adds negligible per-token cost (e.g., KGW+TSW: 5.16 ms/token vs. 4.88 ms for KGW alone) — a ~5% increase.

> Full tables, figures, ablations (effect of `γ` and `δ`), and the removal-attack analysis (Theorem 1) are in [`2026-dualwark.pdf`](2026-dualwark.pdf).

---

## Citation

If you use this code or the DualMark method in your research, please cite:

```bib
@article{qiang2026dualmark,
  title   = {DualMark: A Novel Dual Watermarking Approach for Large Language Models},
  author  = {Qiang, Zihao and Hao, Jifei and Qiang, Jipeng and Zhu, Yi and Zhang, Chaowei and Liu, Yan and Li, Wei},
  journal = {Information Processing \& Management},
  volume  = {63},
  pages   = {104707},
  year    = {2026},
  publisher = {Elsevier},
  doi     = {10.1016/j.ipm.2026.104707}
}
```

---

## References

- KGW — Kirchenbauer et al., *A Watermark for Large Language Models*, ICML 2023.
- UPV — Liu et al., *An Unforgeable Publicly Verifiable Watermark for Large Language Models*, ICLR 2024.
- TSW — Huo et al., *Token-Specific Watermarking with Enhanced Detectability and Semantic Coherence for Large Language Models*, ICML 2024.
- MarkLLM — Pan et al., *MarkLLM: An Open-Source Toolkit for LLM Watermarking*, EMNLP 2024 (System Demonstrations).

---

For questions about running the code, please open an issue in this repository.

[MarkLLM-code]: https://github.com/THU-BPM/MarkLLM
[TSW-code]: https://github.com/mignonjia/TS_watermark
[UPV-code]: https://github.com/THU-BPM/unforgeable_watermark
