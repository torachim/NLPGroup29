<div align="center">

# 🎙️ CLARITY

### *When Politicians Speak — but Don't Answer*
**Detecting Evasion in Political Statements via Fine-Grained Evasion Taxonomy**

---

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-FFD21E)](https://huggingface.co/transformers)
[![XLNet](https://img.shields.io/badge/Model-XLNet--base--cased-9B59B6)](https://huggingface.co/xlnet-base-cased)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![University](https://img.shields.io/badge/University%20of%20Bonn-CAISA%20Lab-003366)](https://www.uni-bonn.de)

<br/>

> *"The art of evasion is not silence — it is fluency without substance. This system learns to hear the difference."*

<br/>

**Introduction to NLP · Winter Semester 2025/2026**  
*<a href="https://github.com/Imado30">Imad Azizi</a> · <a href="https://github.com/SnehpreetDh">Snehpreet Kaur Dhinsa</a> · <a href="https://github.com/rafikfarhane">Rafik Farhane</a>  · <a href="https://github.com/torachim">Torge Rau</a>*

</div>

---

## 🔍 Overview

**CLARITY** tackles one of the most subtle challenges in computational linguistics: automatically detecting *when politicians evade questions*. Evasion is deliberate, fluent, and rhetorically sophisticated — standard NLP systems, trained on cooperative language, consistently misclassify it as a genuine answer.

We introduce a **hierarchical classification framework** that leverages fine-grained evasion labels as intermediate supervision to improve macro-level **response clarity detection**. Rather than asking *"Did this politician answer the question?"*, we first ask *"How exactly are they avoiding it?"* — and use that structure to make a sharper final judgment.

The task: classify each political interview response as one of three **clarity labels**:

| Label | Description |
|---|---|
| ✅ **Clear Reply** | Direct, substantive answer |
| 🔄 **Ambivalent** | Strategic vagueness — neither answering nor refusing |
| ❌ **Clear Non-Reply** | Explicit refusal or deflection |

Benchmarked on the [*"I Never Said That"*](https://konstantinosftw.github.io/CLARITY-SemEval-2026/) dataset — U.S. presidential interviews spanning **17 years (2006–2023)**.

---

## 🧠 The Problem

Political evasion is deceptively fluent. A politician who says *"We need to focus on the real issues facing families..."* in response to *"Did you lie to the Senate?"* is technically speaking — but saying nothing responsive. Key challenges:

- **Long sequences:** Answers average **312 words**, pushing past the 512-token hard limit of most transformers
- **Semantic ambiguity:** The line between *Ambivalent* and *Clear Non-Reply* is pragmatic, not lexical
- **Class imbalance:** 59.2% of responses are *Ambivalent*, causing naive models to collapse onto the majority class
- **Non-cooperative language:** No standard NLP corpus teaches that "addressing" a question ≠ "answering" it

---

## 📊 Dataset

The <a href="https://huggingface.co/datasets/ailsntua/QEvasion">`ailsntua/QEvasion`</a> dataset from the <a href="https://semeval.github.io/SemEval2026/tasks.html">**CLARITY SemEval 2026**</a> challenge:

| Split | Samples |
|---|---|
| Training | 3,448 |
| Test | 308 (3 annotators/sample) |

**Interesting findings from our EDA:**
- *Ambivalent* responses are significantly longer (avg. 342 words) than *Clear Non-Replies* (143 words) — brevity signals refusal (Kruskal-Wallis H=195.36, p<0.001)
- Barack Obama had the highest *Ambivalent* rate (68.1%), Trump the highest *Clear Non-Reply* rate (11.8%), Biden the highest *Clear Reply* rate (37.3%)
- *Clear Non-Reply* rates consistently **drop in election years** — politicians speak more directly under electoral pressure

---

## 🌳 Hierarchical Taxonomy

A two-level hierarchy maps 9 fine-grained rhetorical strategies to 3 clarity classes:

```
✅ Clear Reply        →  Explicit
🔄 Ambivalent         →  Implicit · General · Dodging · Deflection · Partial/half-answer
❌ Clear Non-Reply    →  Declining to answer · Claims ignorance · Clarification
```

We also test a **reduced 5-class taxonomy** (k=5) that merges semantically similar micro-labels (*Dodging* + *Deflection* → *Active Evasion*, etc.) to reduce data sparsity.

---

## ⚙️ Methodology

Five model configurations are evaluated across three label granularities:

| Model | Approach |
|---|---|
| **Logistic Regression (Direct)** | TF-IDF → predict clarity directly (k=3) |
| **Logistic Regression (Hierarchical)** | TF-IDF → predict evasion (k=9) → map to clarity |
| **XLNet Direct** | Fine-tune on clarity labels (k=3) |
| **XLNet Hierarchical** | Fine-tune on evasion (k=9) → map to clarity |
| **XLNet Reduced** | Fine-tune on reduced evasion (k=5) → map to clarity |

**Why XLNet?** Its Transformer-XL recurrence mechanism handles long political responses better than BERT/RoBERTa, preserving long-range rhetorical context without information loss from truncation.

**Architecture:** `[Q] <sep> [A]` → XLNet-base-cased → Mean Pooling → Dropout(0.1) → Linear Head

**Training:** AdamW (lr=2e-5), 15 epochs with early stopping, batch size 8, balanced class weights via Cross-Entropy loss, NVIDIA T4 GPU (Google Colab).

---

## 📁 Project Structure

```
CLARITY/
├── src/
│   ├── model.py                # XLNet architecture (SingleHeadXLNet)
│   ├── dataset.py              # PyTorch Dataset, label maps, voting logic
│   ├── train_direct.py         # Direct clarity training (k=3)
│   ├── train_hierarchical.py   # Hierarchical training (k=9)
│   ├── train_reduced.py        # Reduced taxonomy training (k=5)
│   ├── baseline_models.py      # TF-IDF + Logistic Regression baselines
│   └── evaluation.py           # Metrics computation
├── notebooks/
│   ├── NLP_XLNet_training.ipynb    # Full training pipeline (Colab-ready)
│   ├── data_plots.ipynb            # EDA & visualizations
│   └── final_evaluation.ipynb      # Results, confusion matrices
├── data/raw/                    # Place train.csv & test.csv here
├── models/                      # Saved checkpoints (auto-created)
└── requirements.txt
```

---

## 🚀 Installation & Usage

```bash
# Clone & install
git clone https://github.com/Imado30/CLARITY.git](https://github.com/torachim/NLPGroup29.git
cd CLARITY
pip install -r requirements.txt

# Download dataset
python -c "
from datasets import load_dataset
import os; os.makedirs('data/raw', exist_ok=True)
ds = load_dataset('ailsntua/QEvasion')
ds['train'].to_csv('data/raw/train.csv', index=False)
ds['test'].to_csv('data/raw/test.csv', index=False)
"

# Run baselines
python -m src.baseline_models

# Train XLNet (choose one)
python -m src.train_direct          # Direct clarity (k=3)
python -m src.train_hierarchical    # Hierarchical evasion (k=9)
python -m src.train_reduced         # Reduced taxonomy (k=5)
```

---

## 📈 Results

### Global Performance

| Model | Accuracy | Macro F1 | **Weighted F1** |
|---|---|---|---|
| 🏆 **XLNet Direct (k=3)** | **0.708** | **0.658** | **0.714** |
| XLNet Reduced (k=5) | 0.643 | 0.609 | 0.654 |
| XLNet Hierarchical (k=9) | 0.627 | 0.617 | 0.642 |
| Baseline Hierarchical | 0.562 | 0.436 | 0.554 |
| Baseline Direct | 0.468 | 0.419 | 0.493 |

### Benchmark vs. Prior Work

| System | Macro F1 |
|---|---|
| RoBERTa-base — Thomas et al. (2024) | 0.530 |
| XLNet Evasion Pipeline — Thomas et al. (2024) | 0.546 |
| **CLARITY: XLNet Direct (Ours)** | **0.658** |
| Llama-70B (Instruction-tuned) | 0.680 |

> Our encoder-based system achieves **near-parity with a 70B instruction-tuned LLM** at a fraction of the computational cost.

### Notable Class-Level Findings

- **XLNet Direct** dominates on *Ambivalent* (F1: 0.781) and overall accuracy
- **XLNet Hierarchical (k=9)** achieves the best *Clear Non-Reply* precision (0.607) — fine-grained supervision creates sharper refusal boundaries
- **Baselines** misclassify 72.2% of true *Clear Replies* as *Ambivalent* — XLNet reduces this to 34.2%

---

## 💡 Key Findings

1. **Deep semantics beat lexical features** — The 16 pp Weighted F1 gap between XLNet and TF-IDF proves evasion detection requires contextual understanding, not keyword matching
2. **Direct > Hierarchical for transformers (RQ1)** — Data sparsity fragments the training signal across 9 micro-classes; balanced loss functions already handle minority classes effectively
3. **Hierarchical helps classical models (RQ2)** — Taxonomy structure acts as an inductive bias that compensates for Logistic Regression's limited capacity
4. **Less granularity is better than more (RQ3)** — The 5-class reduced taxonomy outperforms the 9-class scheme, confirming that merging ambiguous labels reduces noise

**Practical recommendation:** Use *XLNet Direct* for general monitoring; use *XLNet Hierarchical (k=9)* when minority-class precision (refusal detection) matters most.

---

## 📚 References

- Thomas et al. (2024) — *"I Never Said That": A Dataset, Taxonomy, and Baselines for Political Evasion*
- Yang et al. (2019) — *XLNet: Generalized Autoregressive Pretraining for Language Understanding*
- Grice (1975) — *Logic and Conversation*
- Bull (2003) — *The Microanalysis of Political Communication*
- [CLARITY SemEval 2026](https://konstantinosftw.github.io/CLARITY-SemEval-2026/)

---

<div align="center">

**University of Bonn · CAISA Lab · Introduction to NLP · WS 2025/26**

*Imad Azizi · Snehpreet Dhinsa · Rafik Farhane · Torge Rau*

---

*"Not all speech is communication. This project teaches machines to know the difference."*

</div>
