import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "/homes/55/arshia/project_1_hallucination/results"
PLOTS_DIR = "/homes/55/arshia/project_1_hallucination/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def main():
    # Load data
    try:
        df_llama = pd.read_csv(f"{RESULTS_DIR}/11_rank_mass_Llama-3.1-8B.csv")
        df_gemma = pd.read_csv(f"{RESULTS_DIR}/11_rank_mass_Gemma-2-9B.csv")
    except:
        print("Could not load 11_rank_mass CSVs. Run 11 first.")
        return

    # Check Columns
    # We need 'rank_true_final'
    
    # Define k thresholds
    k_values = [1, 3, 5, 10, 50, 100]
    
    results = []
    
    for label, df in [("Llama-3.1-8B", df_llama), ("Gemma-2-9B", df_gemma)]:
        total = len(df)
        for k in k_values:
            # rank is 0-indexed. Top-1 means rank 0.
            # Top-k means rank < k.
            count = (df['rank_true_final'] < k).sum()
            recall = count / total
            
            results.append({
                "Model": label,
                "k": str(k), # String for categorical plot
                "Recall": recall
            })
            
    res_df = pd.DataFrame(results)
    print("Top-k Recall Results:")
    print(res_df)
    
    # Plot
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=res_df, x='k', y='Recall', hue='Model', palette="viridis")
    
    # Add values
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
        
    plt.title("Top-k Generation Curve (The 'Knowledge Gap')")
    plt.xlabel("Top-k Threshold")
    plt.ylabel("Recall of True Entity")
    plt.ylim(0, 1.0)
    
    plt.savefig(f"{PLOTS_DIR}/15_topk_gap.png")
    print(f"Saved to {PLOTS_DIR}/15_topk_gap.png")

if __name__ == "__main__":
    main()
