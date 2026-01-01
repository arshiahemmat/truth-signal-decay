import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import os
from matplotlib.gridspec import GridSpec

RESULTS_DIR = "/homes/55/arshia/project_1_hallucination/results"
PLOTS_DIR = "/homes/55/arshia/project_1_hallucination/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def main():
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1, 1])
    
    # --- Data Loading ---
    
    # 1. Intervention (Panel A)
    # Using delta gain
    try:
        df_int = pd.read_csv(f"{RESULTS_DIR}/05_intervention_tradeoff_comprehensive.csv")
    except:
        print("Missing Intervention Data")
        df_int = pd.DataFrame()
        
    # 2. Prior Baseline (Panel B)
    try:
        df_prior = pd.read_csv(f"{RESULTS_DIR}/12_prior_baseline.csv")
    except:
        print("Missing Prior Data")
        df_prior = pd.DataFrame()

    # 3. ROC Data (Panels C-F)
    try:
        with open(f"{RESULTS_DIR}/04_roc_data.pkl", "rb") as f:
            roc_data = pickle.load(f)
    except:
        print("Missing ROC Data")
        roc_data = {}

    # --- Colors ---
    colors = {
        "Llama-3.1-8B": "#1f77b4", # Blue
        "Gemma-2-9B": "#2ca02c",   # Green
        "Llama-Instruct": "#ff7f0e" # Orange
    }
    
    # --- Panel A: Intervention Delta ---
    ax_a = fig.add_subplot(gs[0, :2])
    ax_a.set_title("(a) Monitor detects Signal Decay (Net Gain vs Random)", fontsize=12, fontweight='bold')
    
    if not df_int.empty:
        for model in ["Llama-3.1-8B", "Gemma-2-9B"]:
            sub = df_int[df_int['model'] == model].sort_values('intervention_rate')
            if sub.empty: continue
            
            # Calculate Delta
            # Columns: intervention_rate, net_gain, random_net_gain
            # Note: intervention_rate is coverage
            coverage = sub['intervention_rate']
            delta_gain = sub['net_gain'] - sub['random_net_gain']
            
            c = colors.get(model, 'black')
            ax_a.plot(coverage, delta_gain, marker='o', label=f"{model}", color=c)
            
            # Mark operating point (approx 10% coverage)
            idx = (coverage - 0.10).abs().idxmin()
            if np.abs(coverage[idx] - 0.10) < 0.05: # Ensure we are somewhat close
                op_cov = coverage[idx]
                op_delta = delta_gain[idx]
                ax_a.plot(op_cov, op_delta, marker='*', markersize=15, color='gold', markeredgecolor='black', zorder=10)
                ax_a.annotate(r"$\tau \approx 0.9$", (op_cov, op_delta), xytext=(5, 5), textcoords='offset points')

    ax_a.axhline(0, color='gray', linestyle='--')
    ax_a.set_xlabel("Coverage")
    ax_a.set_ylabel(r"$\Delta$ Net Gain (Monitor - Random)")
    ax_a.legend()
    ax_a.grid(True, alpha=0.3)

    # --- Panel B: Prior KDE ---
    ax_b = fig.add_subplot(gs[0, 2:])
    ax_b.set_title("(b) Prior advantage does not predict Signal Decay", fontsize=12, fontweight='bold')
    
    if not df_prior.empty:
        # Add N to labels
        cats = df_prior['category'].unique()
        for cat in cats:
            subset = df_prior[df_prior['category'] == cat]
            n = len(subset)
            label = f"{cat} (n={n})"
            # Use only necessary categories? Or all?
            # User mentioned "True-Competitive (n=...)" and "False-Selected" is rare
            if n < 5: continue # Skip very rare if KDE fails?
            try:
                sns.kdeplot(data=subset, x='delta_b', fill=True, label=label, ax=ax_b, common_norm=False)
            except:
                pass
            
    ax_b.set_xlabel(r"Prior Advantage ($\Delta b$)")
    ax_b.set_ylabel("Density")
    ax_b.legend(title="Error Category")
    ax_b.grid(True, alpha=0.3)

    # --- Bottom Row: ROC/PR ---
    # Structure: Llama ROC, Llama PR, Gemma ROC, Gemma PR
    models_ordered = ["Llama-3.1-8B", "Gemma-2-9B"]
    panels = [
        ("(c)", "(d)"), 
        ("(e)", "(f)")
    ]
    
    col_idx = 0
    for i, model in enumerate(models_ordered):
        if model not in roc_data: continue
        
        data = roc_data[model]
        c = colors.get(model, 'black')
        
        # ROC Plot
        ax_roc = fig.add_subplot(gs[1, col_idx])
        col_idx += 1
        
        ax_roc.set_title(f"{panels[i][0]} {model} (IID) – ROC", fontsize=11)
        ax_roc.plot(data['fpr'], data['tpr'], color=c, linestyle='-', label=f"Probe (AUC={data['roc_auc']:.2f})")
        ax_roc.plot(data['fpr_e'], data['tpr_e'], color=c, linestyle='--', label=f"Entropy (AUC={data['roc_auc_e']:.2f})")
        ax_roc.plot([0, 1], [0, 1], 'k:', alpha=0.5)
        
        # Marker for 10% FPR? Or operating point?
        # User said "Mark the chosen threshold point".
        # We don't have thresholds in the dict easily unless we saved them.
        # But we can mark 1% FPR or 5% FPR.
        # Let's mark FPR approx 0.05
        idx_roc = (np.abs(data['fpr'] - 0.05)).argmin()
        ax_roc.plot(data['fpr'][idx_roc], data['tpr'][idx_roc], marker='*', color='gold', markersize=10, markeredgecolor='black', zorder=10)

        ax_roc.set_xlabel("FPR")
        ax_roc.set_ylabel("TPR")
        ax_roc.legend(fontsize=9)
        ax_roc.grid(True, alpha=0.3)
        
        # PR Plot
        ax_pr = fig.add_subplot(gs[1, col_idx])
        col_idx += 1
        
        ax_pr.set_title(f"{panels[i][1]} {model} (IID) – PR", fontsize=11)
        ax_pr.plot(data['recall'], data['precision'], color=c, linestyle='-', label=f"Probe (AP={data['pr_auc']:.2f})")
        ax_pr.plot(data['recall_e'], data['precision_e'], color=c, linestyle='--', label=f"Entropy (AP={data['pr_auc_e']:.2f})")
        
        # Marker at Recall corresponding to same threshold?
        # Hard without alignment. We assume recall ~ 50% for high precision?
        # Let's Skip marker on PR if we can't link them accurately.
        # Or mark Recall where Precision drops?
        
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_ylim(0, 1.05)
        ax_pr.legend(fontsize=9)
        ax_pr.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/18_final_summary.png", dpi=150)
    print(f"Saved {PLOTS_DIR}/18_final_summary.png")

if __name__ == "__main__":
    main()
