import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "/homes/55/arshia/project_1_hallucination/results"
PLOTS_DIR = "/homes/55/arshia/project_1_hallucination/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Style
sns.set_context("paper", font_scale=1.5)
sns.set_style("ticks")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

COLORS = {
    "Llama-3.1-8B": "#1f77b4", 
    "Gemma-2-9B": "#2ca02c",   
    "Llama-Instruct": "#ff7f0e"
}
CATEGORIES_PALETTE = {
    "Correct": "#aec7e8",
    "True-Competitive": "#2ca02c",
    "Generic-Collapse": "#1f77b4",
    "False-Selected": "#9467bd",
    "Other": "#7f7f7f",
    "Overwritten": "#ff7f0e", # Orange
    "Absent": "#d62728"      # Red
}

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{filename}", dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()

# --- Composite 1: The Gap (Baseline, Margin, Top-k) ---
def plot_gap_composite():
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    
    # 1. Baseline Accuracy (Left)
    try:
        df = pd.read_csv(f"{RESULTS_DIR}/01_baseline_comparison.csv")
        df_long = pd.melt(df, id_vars=['model'], value_vars=['is_correct_pairwise', 'is_correct_top1'], 
                          var_name='Metric', value_name='Correct')
        df_long['Metric'] = df_long['Metric'].replace({'is_correct_pairwise': 'Knowledge', 'is_correct_top1': 'Generation'})
        
        sns.barplot(data=df_long, x="model", y="Correct", hue="Metric", palette="Paired", ax=axes[0])
        axes[0].set_title("(a) Knowledge vs Generation Gap", fontweight='bold')
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel("Accuracy")
        axes[0].set_xlabel("")
        axes[0].legend(title=None)
        for container in axes[0].containers:
            axes[0].bar_label(container, fmt='%.2f', fontsize=11)
    except Exception as e:
        print(f"Error Panel 1: {e}")

    # 2. Competition Margin (Middle)
    try:
        df = pd.read_csv(f"{RESULTS_DIR}/17_margin_data.csv")
        sns.kdeplot(data=df, x='margin', hue='category', fill=True, common_norm=False, palette="mako", linewidth=2, ax=axes[1])
        axes[1].set_title("(b) Competition Margin (Logits)", fontweight='bold')
        axes[1].set_xlabel("Margin (Top1 - True)")
        axes[1].axvline(0, color='k', linestyle='--')
    except Exception as e:
        print(f"Error Panel 2: {e}")

    # 3. Top-k Curve (Right)
    try:
        df_l = pd.read_csv(f"{RESULTS_DIR}/11_rank_mass_Llama-3.1-8B.csv")
        df_g = pd.read_csv(f"{RESULTS_DIR}/11_rank_mass_Gemma-2-9B.csv")
        k_values = [1, 3, 5, 10, 50, 100]
        results = []
        for label, df in [("Llama-3.1-8B", df_l), ("Gemma-2-9B", df_g)]:
            total = len(df)
            for k in k_values:
                recall = (df['rank_true_final'] < k).sum() / total
                results.append({"Model": label, "k": str(k), "Recall": recall})
        
        res_df = pd.DataFrame(results)
        sns.barplot(data=res_df, x='k', y='Recall', hue='Model', palette="viridis", ax=axes[2])
        axes[2].set_title("(c) Top-k Generation Curve", fontweight='bold')
        axes[2].set_xlabel("Top-k Threshold")
        axes[2].set_ylim(0, 1.05)
        for container in axes[2].containers:
            axes[2].bar_label(container, fmt='%.2f', fontsize=10)
    except Exception as e:
        print(f"Error Panel 3: {e}")
        
    save_plot("composite_gap.png")

# --- Composite 2: Error Modes (Decomp, Prefix, Stoplist) ---
def plot_error_composite():
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    
    # 1. Decomposition (Left)
    try:
        df_l = pd.read_csv(f"{RESULTS_DIR}/11_rank_mass_Llama-3.1-8B.csv")
        df_g = pd.read_csv(f"{RESULTS_DIR}/11_rank_mass_Gemma-2-9B.csv")
        combined = []
        for label, df in [("Llama-3.1-8B", df_l), ("Gemma-2-9B", df_g)]:
            # Filter errors
            errors = df[df['rank_true_final'] > 0]
            if len(errors) > 0:
                counts = errors['category'].value_counts(normalize=True).reset_index()
                counts.columns = ['Category', 'Proportion']
                counts['Model'] = label
                combined.append(counts)
        
        if combined:
            df_plot = pd.concat(combined)
            sns.barplot(data=df_plot, x="Model", y="Proportion", hue="Category", palette=CATEGORIES_PALETTE, ax=axes[0])
            axes[0].set_title("(a) Error Mode Decomposition", fontweight='bold')
            axes[0].set_ylim(0, 0.7)
            axes[0].set_xlabel("")
    except Exception as e:
        print(f"Error Panel 4: {e}")

    # 2. Prefix Robustness (Middle)
    data = {"Category": ["True-Competitive", "Generic-Collapse"], "Recovery Rate": [0.28, 0.0]}
    df = pd.DataFrame(data)
    sns.barplot(data=df, x="Category", y="Recovery Rate", palette="viridis", ax=axes[1])
    axes[1].set_title("(b) Prefix Recovery", fontweight='bold')
    axes[1].set_ylim(0, 0.4)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt='%.2f')

    # 3. Stoplist (Right)
    data2 = {"Category": ["True-Competitive", "Generic-Collapse"], "Recovery Rate": [0.16, 0.0]}
    df2 = pd.DataFrame(data2)
    sns.barplot(data=df2, x="Category", y="Recovery Rate", palette="magma", ax=axes[2])
    axes[2].set_title("(c) Stoplist Recovery", fontweight='bold')
    axes[2].set_ylim(0, 0.4)
    for container in axes[2].containers:
        axes[2].bar_label(container, fmt='%.2f')
        
    save_plot("composite_error_modes.png")

# --- Composite 3: Intervention (Mixing, Tradeoff) ---
def plot_intervention_composite():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Logit Mixing (Left)
    try:
        df = pd.read_csv(f"{RESULTS_DIR}/08_logit_mixing.csv")
        sns.lineplot(data=df, x='alpha', y='net_gain', hue='model', marker='o', linewidth=2.5, ax=axes[0])
        axes[0].axhline(0, color='gray', linestyle='--')
        axes[0].set_title("(a) Logit Mixing Efficacy", fontweight='bold')
        axes[0].set_xlabel("Mixing Alpha")
        axes[0].set_ylabel("Net Gain")
    except Exception as e:
        print(f"Error Panel 7: {e}")
        
    # 2. Tradeoff (Right)
    try:
        df = pd.read_csv(f"{RESULTS_DIR}/05_intervention_tradeoff_comprehensive.csv")
        # Net Gain vs Coverage
        sns.lineplot(data=df, x='intervention_rate', y='net_gain', hue='model', style='model', markers=True, dashes=False, linewidth=2.5, markersize=9, ax=axes[1])
        
        # Random Lines
        for model in df['model'].unique():
            sub = df[df['model'] == model].sort_values('intervention_rate')
            c = COLORS.get(model, 'gray')
            axes[1].plot(sub['intervention_rate'], sub['random_net_gain'], linestyle='--', color=c, alpha=0.6, label=f"{model} (Random)")

        axes[1].axhline(0, color='gray', linestyle='-')
        axes[1].set_title("(b) Monitor Trade-off (Net Gain vs Coverage)", fontweight='bold')
        axes[1].set_xlabel("Coverage")
        axes[1].set_ylabel("Net Gain")
    except Exception as e:
        print(f"Error Panel 8: {e}")
        
    save_plot("composite_intervention.png")

# --- Composite 4: Mechanistic (Trace, Trajectory, Entropy) ---
def plot_mechanistic_composite():
    # User requested: Logit Lens Trace | Trajectory | Entropy
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    
    # Data Loading
    try:
        df_trace = pd.read_csv(f"{RESULTS_DIR}/02_logit_lens_data.csv")
        df_ent = pd.read_csv(f"{RESULTS_DIR}/10_entropy_data.csv")
    except Exception as e:
        print(f"Error Loading Mech Data: {e}")
        return

    # Panel A: Trace of Correct Probability
    # Rename for cleaner plotting
    df_trace['probability'] = 1 / (1 + np.exp(-df_trace['logit_diff']))
    df_trace['Final Output'] = df_trace['final_correct'].map({True: 'Correct', False: 'Incorrect'})
    
    sns.lineplot(data=df_trace, x='layer', y='probability', hue='Final Output', style='Final Output', ax=axes[0], errorbar='sd')
    axes[0].set_title("(a) Logit Lens: Probability Trace", fontweight='bold')
    axes[0].set_xlabel("Layer Index")
    axes[0].set_ylabel("P(True token)")
    axes[0].legend(title="Final output correct?")
    axes[0].grid(True, alpha=0.3)

    # Panel B: Trajectory Analysis
    sns.lineplot(data=df_trace, x='layer', y='logit_diff', hue='category', style='category', ax=axes[1], palette=CATEGORIES_PALETTE, errorbar='se')
    axes[1].set_title("(b) Trajectory Analysis (Mean Logit Diff)", fontweight='bold')
    axes[1].set_xlabel("Layer Index")
    axes[1].set_ylabel("Logit Difference (True - False)")
    axes[1].axhline(0, color='gray', linestyle=':')
    axes[1].grid(True, alpha=0.3)
    
    # Panel C: Entropy by Category with N
    # Calculate counts
    counts = df_ent['category'].value_counts()
    # Create new label column with n
    df_ent['label_n'] = df_ent['category'].apply(lambda x: f"{x}\n(n={counts.get(x,0)})")
    
    # Order needs to be mapped to the new labels
    base_order = ["Correct", "Overwritten", "Absent"]
    new_order = [f"{x}\n(n={counts.get(x,0)})" for x in base_order if x in counts.index]
    
    # We must map the palette to these new labels too, or just use the hue mapping directly if we stick to 'category' for hue but labels for x
    # Easier: Plot x='category' but set_xticklabels manually? No, safer to map data.
    
    sns.boxplot(data=df_ent, x='label_n', y='entropy', order=new_order, palette=list(CATEGORIES_PALETTE.values()), hue='category', legend=False, ax=axes[2])
    # Note: Palette usage with 'hue' vs 'x' mismatch in Seaborn.
    # To use our dict, we need x to match keys.
    # Alternative: Use x='category' and update tick labels.
    
    sns.boxplot(data=df_ent, x='category', y='entropy', order=["Correct", "Overwritten", "Absent"], palette=CATEGORIES_PALETTE, ax=axes[2])
    axes[2].set_xticklabels([f"{x}\n(n={counts.get(x,0)})" for x in ["Correct", "Overwritten", "Absent"]])
    
    axes[2].set_title("(c) Entropy by Error Category", fontweight='bold')
    axes[2].set_ylabel("Predictive Entropy")
    axes[2].set_xlabel("")
    
    save_plot("composite_mechanistic.png")

def main():
    print("Generating Composites...")
    plot_gap_composite()
    plot_error_composite()
    plot_intervention_composite()
    plot_mechanistic_composite()
    print("Done.")

if __name__ == "__main__":
    main()
