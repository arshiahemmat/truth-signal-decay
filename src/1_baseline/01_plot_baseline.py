
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def plot_baseline():
    csv_path = f"{RESULTS_DIR}/01_baseline_comparison.csv"
    if not os.path.exists(csv_path):
        print(f"No results found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # 1. Accuracy Plot (Pairwise vs Top-1)
    df_long = pd.melt(df, id_vars=['model'], value_vars=['is_correct_pairwise', 'is_correct_top1'], 
                      var_name='Metric', value_name='Correct')
    
    # Rename for readability
    df_long['Metric'] = df_long['Metric'].replace({
        'is_correct_pairwise': 'Pairwise (Knowledge)',
        'is_correct_top1': 'Top-1 (Exact)'
    })
    
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=df_long, x="model", y="Correct", hue="Metric", palette="Paired")
    plt.title("Baseline Accuracy: Pairwise (Knowledge) vs Top-1 (Exact)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
        
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/01_baseline_accuracy.png", dpi=150)
    plt.close()
    
    # 2. Entropy Distribution (Split by Correctness)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="model", y="entropy", hue="is_correct_pairwise", palette="Set2")
    plt.title("Predictive Entropy by Correctness (Pairwise)")
    plt.xlabel("Model")
    plt.ylabel("Entropy (nats)")
    plt.legend(title="Correct (Pairwise)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/01_baseline_entropy.png")
    plt.close()
    
    print(f"Plots saved to {PLOTS_DIR}")

if __name__ == "__main__":
    plot_baseline()
