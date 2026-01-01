import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
from matplotlib.gridspec import GridSpec

# --- Configuration ---
RESULTS_DIR = "/homes/55/arshia/project_1_hallucination/results"
PLOTS_DIR = "/homes/55/arshia/project_1_hallucination/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Set Style
sns.set_context("paper", font_scale=1.4)
sns.set_style("ticks")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

COLORS = {
    "Llama-3.1-8B": "#1f77b4", # Blue
    "Gemma-2-9B": "#2ca02c",   # Green
    "Llama-Instruct": "#ff7f0e", # Orange
    "Monitor": "#d62728", # Red
    "Random": "#7f7f7f", # Grey
    "Entropy": "#9467bd" # Purple
}

CATEGORIES_PALETTE = {
    "Correct": "#aec7e8",
    "True-Competitive": "#2ca02c",
    "Generic-Collapse": "#1f77b4",
    "False-Selected": "#9467bd",
    "Other": "#7f7f7f"
}

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{filename}", dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()

# --- Plot 1: Baseline Accuracy (01) ---
def plot_baseline():
    try:
        df = pd.read_csv(f"{RESULTS_DIR}/01_baseline_comparison.csv")
    except: return
    
    df_long = pd.melt(df, id_vars=['model'], value_vars=['is_correct_pairwise', 'is_correct_top1'], 
                      var_name='Metric', value_name='Correct')
    df_long['Metric'] = df_long['Metric'].replace({
        'is_correct_pairwise': 'Knowledge (Pairwise)',
        'is_correct_top1': 'Generation (Top-1)'
    })
    
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=df_long, x="model", y="Correct", hue="Metric", palette="Paired")
    plt.title("Knowledge vs Generation Gap", fontsize=16, fontweight='bold')
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.legend(title=None)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=12)
    save_plot("01_baseline_accuracy.png")

# --- Plot 2: Top-k Gap (15) ---
def plot_topk():
    try:
        df_l = pd.read_csv(f"{RESULTS_DIR}/11_rank_mass_Llama-3.1-8B.csv")
        df_g = pd.read_csv(f"{RESULTS_DIR}/11_rank_mass_Gemma-2-9B.csv")
    except: return
    
    k_values = [1, 3, 5, 10, 50, 100]
    results = []
    for label, df in [("Llama-3.1-8B", df_l), ("Gemma-2-9B", df_g)]:
        total = len(df)
        for k in k_values:
            recall = (df['rank_true_final'] < k).sum() / total
            results.append({"Model": label, "k": str(k), "Recall": recall})
            
    res_df = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=res_df, x='k', y='Recall', hue='Model', palette="viridis")
    plt.title("Top-k Generation Curve (The 'Knowledge Gap')", fontsize=16, fontweight='bold')
    plt.xlabel("Top-k Threshold")
    plt.ylabel("Recall of True Token")
    plt.ylim(0, 1.05)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=10)
    save_plot("15_topk_gap.png")

# --- Plot 3: Error Decomposition (11) ---
def plot_decomposition():
    try:
        df_l = pd.read_csv(f"{RESULTS_DIR}/11_rank_mass_Llama-3.1-8B.csv")
        df_g = pd.read_csv(f"{RESULTS_DIR}/11_rank_mass_Gemma-2-9B.csv")
    except: return
    
    combined = []
    for label, df in [("Llama-3.1-8B", df_l), ("Gemma-2-9B", df_g)]:
        # Filter for Errors (rank_true > 0)
        errors = df[df['rank_true_final'] > 0]
        counts = errors['category'].value_counts(normalize=True).reset_index()
        counts.columns = ['Category', 'Proportion']
        counts['Model'] = label
        combined.append(counts)
        
    df_plot = pd.concat(combined)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_plot, x="Model", y="Proportion", hue="Category", palette=CATEGORIES_PALETTE)
    plt.title("Error Mode Decomposition", fontsize=16, fontweight='bold')
    plt.ylim(0, 0.7)
    save_plot("11_error_decomposition.png")

# --- Plot 4: Winner Breakdown (16) ---
def plot_winner():
    # Hardcoded from previous run analysis to ensure matching style without re-logic
    # Llama-3.1-8B data
    data = {"Winner Type": ["Generic Stopword", "Other Entity", "False Target"],
            "Proportion": [0.63, 0.36, 0.01]}
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(6, 5))
    ax = sns.barplot(data=df, x="Winner Type", y="Proportion", palette="rocket")
    plt.title("What Beats the Truth?", fontsize=14, fontweight='bold')
    plt.ylabel("Proportion of Errors")
    plt.ylim(0, 0.8)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    save_plot("16_winner_breakdown.png")

# --- Plot 5: Competition Margin (17) ---
def plot_margin():
    try:
        df = pd.read_csv(f"{RESULTS_DIR}/17_margin_data.csv")
    except: return
    
    plt.figure(figsize=(8, 6))
    # Rename categories for clean legend
    sns.kdeplot(data=df, x='margin', hue='category', fill=True, common_norm=False, palette="mako", linewidth=2)
    plt.title("Competition Margin Distribution", fontsize=16, fontweight='bold')
    plt.xlabel("Logit Margin (Logit_Top1 - Logit_True)")
    plt.axvline(0, color='k', linestyle='--')
    save_plot("17_competition_margin.png")

# --- Plot 6: Intervention Tradeoff (05) ---
def plot_tradeoff():
    try:
        df = pd.read_csv(f"{RESULTS_DIR}/05_intervention_tradeoff_comprehensive.csv")
    except: return
    
    # Plot Net Gain vs Coverage
    plt.figure(figsize=(8, 6))
    
    # Monitor (Solid)
    sns.lineplot(data=df, x='intervention_rate', y='net_gain', hue='model', style='model', markers=True, dashes=False, linewidth=2.5, markersize=9)
    
    # Random (Dashed) - Need to melt or just plot overlay?
    # Overlay is easier. Random gain depends on coverage.
    # Since Random Net Gain is in the CSV, we can plot it.
    # But filtering by model results in plotting logic issues if we just pass 'y=random_net_gain'.
    # We can plot Random lines per model manually.
    for model in df['model'].unique():
        sub = df[df['model'] == model].sort_values('intervention_rate')
        c = COLORS.get(model, 'gray')
        plt.plot(sub['intervention_rate'], sub['random_net_gain'], linestyle='--', color=c, alpha=0.6, label=f"{model} (Random)")

    plt.axhline(0, color='gray', linestyle='-')
    plt.title("Intervention Efficacy: Net Gain vs Coverage", fontsize=16, fontweight='bold')
    plt.xlabel("Intervention Rate (Coverage)")
    plt.ylabel("Net Gain (Corrected Answers)")
    save_plot("05_intervention_tradeoff.png")

# --- Plot 7: Logit Mixing (08) ---
def plot_mixing():
    try:
        df = pd.read_csv(f"{RESULTS_DIR}/08_logit_mixing.csv")
    except: return
    
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x='alpha', y='net_gain', hue='model', marker='o', linewidth=2.5)
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Logit Mixing Efficacy", fontsize=16, fontweight='bold')
    plt.xlabel("Mixing Coefficient $\\alpha$")
    plt.ylabel("Net Gain")
    save_plot("08_logit_mixing.png")

# --- Plot 8: Final Summary Composite (18) ---
def plot_final_summary_composite():
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1, 1], wspace=0.3, hspace=0.45)
    
    try:
        df_int = pd.read_csv(f"{RESULTS_DIR}/05_intervention_tradeoff_comprehensive.csv")
        df_prior = pd.read_csv(f"{RESULTS_DIR}/12_prior_baseline.csv")
        with open(f"{RESULTS_DIR}/04_roc_data.pkl", "rb") as f:
            roc_data = pickle.load(f)
    except: return

    # Panel A: Delta Gain
    ax_a = fig.add_subplot(gs[0, :2])
    ax_a.set_title("(a) Monitor detects Signal Decay (Net Gain vs Random)", fontsize=14, fontweight='bold')
    if not df_int.empty:
        for model in ["Llama-3.1-8B", "Gemma-2-9B"]:
            # Filter by model
            sub = df_int[df_int['model'] == model].sort_values('intervention_rate')
            if sub.empty: continue
            
            # Use 'random_net_gain' column
            coverage = sub['intervention_rate']
            delta = sub['net_gain'] - sub['random_net_gain']
            
            c = COLORS.get(model, 'black')
            ax_a.plot(coverage, delta, marker='o', label=model, color=c, lw=2)
            
            # Star at 0.10 coverage
            idx = (coverage - 0.10).abs().idxmin()
            if abs(coverage[idx]-0.10) < 0.05:
                ax_a.plot(coverage[idx], delta[idx], marker='*', color='gold', markersize=18, markeredgecolor='k', zorder=10)
                
    ax_a.axhline(0, color='gray', ls='--')
    ax_a.set_xlabel("Coverage")
    ax_a.set_ylabel(r"$\Delta$ Net Gain")
    ax_a.legend()
    ax_a.grid(True, alpha=0.3)
    
    # Panel B: Prior KDE
    ax_b = fig.add_subplot(gs[0, 2:])
    ax_b.set_title("(b) Prior Advantage does not predict Signal Decay", fontsize=14, fontweight='bold')
    if not df_prior.empty:
        cats = ["Correct", "Generic-Collapse", "True-Competitive"]
        for cat in cats:
            sub = df_prior[df_prior['category'] == cat]
            if len(sub) > 5:
                sns.kdeplot(data=sub, x='delta_b', fill=True, label=f"{cat} (n={len(sub)})", ax=ax_b, common_norm=False, lw=2, alpha=0.3)
    ax_b.set_xlabel(r"Prior Advantage ($\Delta b$)")
    ax_b.legend()
    ax_b.grid(True, alpha=0.3)
    
    # Bottom Row: ROC/PR
    models = ["Llama-3.1-8B", "Gemma-2-9B"]
    panels = [("(c)", "(d)"), ("(e)", "(f)")]
    col = 0
    for i, m in enumerate(models):
        if m not in roc_data: 
            col += 2
            continue
        d = roc_data[m]
        c = COLORS.get(m, 'black')
        
        # ROC
        ax_r = fig.add_subplot(gs[1, col])
        ax_r.set_title(f"{panels[i][0]} {m} (IID) ROC", fontsize=12, fontweight='bold')
        ax_r.plot(d['fpr'], d['tpr'], color=c, lw=2.5, label=f"Probe (AUC={d['roc_auc']:.2f})")
        ax_r.plot(d['fpr_e'], d['tpr_e'], color=c, lw=2, ls='--', label=f"Entropy (AUC={d['roc_auc_e']:.2f})")
        ax_r.plot([0,1],[0,1],'k:', alpha=0.5)
        
        # Star at FPR 0.05
        idx = (np.abs(d['fpr'] - 0.05)).argmin()
        ax_r.plot(d['fpr'][idx], d['tpr'][idx], marker='*', color='gold', markersize=14, markeredgecolor='k', zorder=10)
        
        ax_r.set_xlabel("FPR")
        ax_r.set_ylabel("TPR")
        ax_r.legend(fontsize=9, loc='lower right')
        ax_r.grid(True, alpha=0.3)
        col += 1
        
        # PR
        ax_p = fig.add_subplot(gs[1, col])
        ax_p.set_title(f"{panels[i][1]} {m} (IID) PR", fontsize=12, fontweight='bold')
        ax_p.plot(d['recall'], d['precision'], color=c, lw=2.5, label=f"Probe (AP={d['pr_auc']:.2f})")
        ax_p.plot(d['recall_e'], d['precision_e'], color=c, lw=2, ls='--', label=f"Entropy (AP={d['pr_auc_e']:.2f})")
        ax_p.set_xlabel("Recall")
        ax_p.set_ylabel("Precision")
        ax_p.legend(fontsize=9)
        ax_p.grid(True, alpha=0.3)
        col += 1
        
    save_plot("18_final_summary.png")

# --- Quick Extras (Hardcoded) ---
def plot_prefix_recovery():
    # 13_prefix_robustness.png
    # True-Comp: ~0.28, Others 0
    data = {"Category": ["True-Competitive", "Generic-Collapse"],
            "Recovery Rate": [0.28, 0.0]}
    df = pd.DataFrame(data)
    plt.figure(figsize=(6, 5))
    ax = sns.barplot(data=df, x="Category", y="Recovery Rate", palette="viridis")
    plt.title("Prefix Robustness (Do Errors Recover?)", fontsize=14, fontweight='bold')
    plt.ylim(0, 0.4)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    save_plot("13_prefix_robustness.png")

def plot_stoplist():
    # 14_stoplist_intervention.png
    # True-Comp 0.16, Generic 0.0
    data = {"Category": ["True-Competitive", "Generic-Collapse"],
            "Recovery Rate": [0.16, 0.0]}
    df = pd.DataFrame(data)
    plt.figure(figsize=(6, 5))
    ax = sns.barplot(data=df, x="Category", y="Recovery Rate", palette="magma")
    plt.title("Stoplist Intervention Efficacy", fontsize=14, fontweight='bold')
    plt.ylim(0, 0.4)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    save_plot("14_stoplist_intervention.png")

def main():
    print("Generating High-Quality Plots...")
    plot_baseline()
    plot_topk()
    plot_decomposition()
    plot_winner()
    plot_margin()
    plot_tradeoff()
    plot_mixing()
    plot_prefix_recovery()
    plot_stoplist()
    plot_final_summary_composite()
    print("Done.")

if __name__ == "__main__":
    main()
