# Hallucination as Signal Decay
**When Models Know but Don’t Say: Detecting Truth-Signal Decay and Fixing It with Conditional Decoding**

We examine why LLMs fail to generate facts they "know" (Pairwise Accuracy > 90%), identifying a "Top-k Decoding Gap" where the correct entity is present in the logits but loses to generic tokens. We propose a lightweight **Top-k Monitor** and **Logit Mixing** intervention to recover these errors.

![Gap Analysis](assets/composite_gap.png)

> **See [ANALYSIS.md](ANALYSIS.md) for a deep-dive visual report of the findings.**

## 💻 Interactive Demo

You can easily use this link to run the code interactively without installing anything:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arshiahemmat/truth-signal-decay/blob/main/Hallucination_Analysis.ipynb)

## 📂 Repository Structure

The codebase is organized into five main modules under `src/`:

### `src/1_baseline/` (The Phenomenon)
Quantifies the Knowledge-Generation Gap.
*   `01_baseline.py`: Evaluates Pairwise vs. Top-1 Accuracy.
*   `12_prior_frequency_baseline.py`: Checks prior frequency bias.
*   `15_topk_gap_plot.py`: Plots the Top-k Recall Curve.

### `src/2_analysis/` (Mechanistic & Error Modes)
Investigates *why* the gap exists.
*   `02_logit_lens.py`: Traces logit evolution across layers.
*   `03_causal_trace.py`: Ablation study of suppressor heads.
*   `11_rank_mass_decomposition.py`: Decomposes errors (False-Selected vs Generic-Collapse).
*   `16_winner_breakdown.py`: Analyzes which tokens win (e.g., stopwords).

### `src/3_monitor/` (Detection)
Trains and evaluates the decay detector.
*   `04_train_probe.py`: Trains Logistic Regression probes.
*   `07_stats_metrics.py`: Computes False Positive Rates (FPR).
*   `09_ood_split.py`: Tests OOD generalization on held-out relations.

### `src/4_intervention/` (Mitigation)
Methods to fix the decay.
*   `05_intervention.py`: Baseline intervention tradeoffs.
*   `08_logit_mixing.py`: **Logit Mixing** (Soft Steering) implementation.
*   `14_stoplist_intervention.py`: Stoplist constraint implementation.

### `src/5_plotting/` (Figures)
Generates publication-ready figures.
*   `21_paper_figures_final.py`: **Master Script** - Generates Figures 1-8.
*   `20_generate_composites.py`: Creates unified composite panels.

## 🚀 Usage

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Experiments**:
    Navigate to the relevant folder and run the script.
    *Example: Reproduce the Baseline Gap*
    ```bash
    python src/1_baseline/01_baseline.py
    ```

3.  **Generate Figures**:
    ```bash
    python src/5_plotting/21_paper_figures_final.py
    ```
    *Figures will be saved to `plots/` (created automatically).*

## 📊 Key Findings

*   **The Gap**: Llama-3.1-8B has 92% Knowledge (Pairwise) but only 24% Generation (Top-1).
*   **Signal Decay**: 92% of failures are "Signal Decay" (True-Competitive or Generic-Collapse), not confident false beliefs.
*   **Intervention**: Monitor-gated Logit Mixing recovers errors with positive net gain (+3.0), outperforming random baselines (-7.0).

## 📝 License
MIT License.
