import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

RESULTS_DIR = "/homes/55/arshia/project_1_hallucination/results"
PLOTS_DIR = "/homes/55/arshia/project_1_hallucination/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def calculate_overwriting_strength(csv_path, label):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return None
        
    df = pd.read_csv(csv_path)
    
    # Define "Peak" range (Mid layers)
    # Llama-3.1-8B has 32 layers. Mid = 15-25.
    mid_df = df[(df['layer'] >= 15) & (df['layer'] <= 25)]
    final_df = df[df['layer'] == df['layer'].max()]
    
    # Calculate Peak Logit Diff per prompt
    peaks = mid_df.groupby('prompt_idx')['logit_diff'].max()
    
    # Calculate Final Logit Diff per prompt
    finals = final_df.set_index('prompt_idx')['logit_diff']
    
    # Merge
    merged = pd.DataFrame({'peak': peaks, 'final': finals}).dropna()
    
    # Metric: Overwriting Strength = Peak - Final
    # If Peak (Correct) is high and Final is low (Wrong/Suppressed), strength is High.
    merged['strength'] = merged['peak'] - merged['final']
    merged['label'] = label
    
    return merged

def main():
    base_csv = f"{RESULTS_DIR}/02_logit_lens_data.csv"
    instruct_csv = f"{RESULTS_DIR}/02_logit_lens_data_instruct.csv"
    
    df_base = calculate_overwriting_strength(base_csv, "Llama-3.1-8B (Base)")
    df_instruct = calculate_overwriting_strength(instruct_csv, "Llama-3.1-8B-Instruct")
    
    if df_base is None or df_instruct is None:
        return
        
    final_df = pd.concat([df_base, df_instruct])
    
    # Stats
    print("\nOverwriting Strength (Peak - Final) Stats:")
    print(final_df.groupby('label')['strength'].describe())
    
    # Plot KDE
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=final_df, x="strength", hue="label", fill=True, common_norm=False, palette={"Llama-3.1-8B (Base)": "blue", "Llama-3.1-8B-Instruct": "orange"})
    
    plt.title("Distribution of Overwriting Strength (Δ_peak - Δ_final)")
    plt.xlabel("Strength (Logit Diff Drop)")
    plt.axvline(0, color='black', linestyle=':')
    plt.grid(True, alpha=0.3)
    
    output_path = f"{PLOTS_DIR}/06_mechanistic_diff.png"
    plt.savefig(output_path)
    print(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    main()
