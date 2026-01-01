# Hallucination as Signal Decay
**Investigating and Mitigating Knowledge-Generation Gaps in LLMs**

This repository contains the code for the paper "Hallucination as Signal Decay". We investigate why LLMs fail to generate facts they "know" (Pairwise Accuracy > 90%), identifying a "Top-k Decoding Gap" where the correct entity is present in the logits but loses to generic tokens during final selection. We propose a lightweight **Top-k Monitor** and **Logit Mixing** intervention to recover these errors.

## 📂 Repository Structure

### 1. Baseline & Phenomenon (Section 4.1 & 4.2)
*   `01_baseline.py`: Evaluates Pairwise vs. Top-1 Accuracy on CounterFact.
*   `02_logit_lens.py`: Traces logit evolution across layers (Logit Lens).
*   `11_rank_mass_decomposition.py`: Decomposes errors into "False-Selected", "True-Competitive", and "Generic-Collapse".
*   `15_topk_gap_plot.py`: Generates the Top-k Recall Curve.

### 2. Mechanistic Analysis (Section 4.3)
*   `03_causal_trace.py`: Performs specific-head ablation ("Suppressor Heads").
*   `06_mechanistic_diff.py`: Analyzes the "Overwriting" signature (Peak - Final).
*   `10_narrative_checks.py`: Entropy analysis and qualitative audits.

### 3. Monitor & Intervention (Section 4.4 & 4.5)
*   **Probing (Monitor)**
    *   `04_train_probe.py`: Trains Logistic Regression probes on mid-layer activations.
    *   `07_stats_metrics.py`: Computes rigorous False Positive Rates and Confidence Intervals.
    *   `09_ood_split.py`: Evaluates OOD Generalization on held-out relations.
*   **Intervention**
    *   `05_intervention.py`: Evaluates Monitor-Gated Intervention trade-offs.
    *   `08_logit_mixing.py`: Implements "Logit Mixing" (Soft Steering).
    *   `14_stoplist_intervention.py`: Implements "Stoplist Constraints" (Hard Blocking).

### 4. Robustness Checks
*   `12_prior_frequency_baseline.py`: Checks if "Prior Advantage" explains the error (Control).
*   `13_prefix_robustness.py`: Tests if errors recover given the first token (Prefix).

### 5. Visualization (Section 4.6 & Figures)
*   `21_paper_figures_final.py`: **Main Plotting Script**. Generates Figures 1-8 for the paper.
*   `20_generate_composites.py`: Generates composite panels (Mechanistic, Error Modes).
*   `19_paper_plots.py`: High-resolution individual plots.

## 🚀 Usage

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Experiments**:
    Scripts are numbered sequentially. To reproduce the core "Knowledge Gap" finding:
    ```bash
    python 01_baseline.py
    ```

3.  **Train Monitor**:
    ```bash
    python 04_train_probe.py
    ```

4.  **Generate Paper Figures**:
    ```bash
    python 21_paper_figures_final.py
    ```

## 📊 Key Findings

*   **The Gap**: Llama-3.1-8B has 92% Knowledge (Pairwise) but only 24% Generation (Top-1).
*   **Signal Decay**: 92% of failures are "Signal Decay" (True-Competitive or Generic-Collapse), not confident false beliefs.
*   **Intervention**: Monitor-gated Logit Mixing recovers errors with positive net gain (+3.0), outperforming random baselines (-7.0).

## 📝 License
MIT License.
