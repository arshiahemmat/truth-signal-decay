import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

RESULTS_DIR = "/homes/55/arshia/project_1_hallucination/results"
PLOTS_DIR = "/homes/55/arshia/project_1_hallucination/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def main():
    # 1. Load Data
    # 01 has metadata (prompt, entropy, logits)
    df_base = pd.read_csv(f"{RESULTS_DIR}/01_baseline_comparison.csv")
    
    # 02 has category (Overwritten vs Absent)
    # Note: 02 data might be a subset (n~1000). We need to align.
    # 02 usually has 'prompt_idx' which maps to... row index of what?
    # In earlier scripts, we saw 02 was generated from a subset. 
    # Let's check 02 columns.
    df_lens = pd.read_csv(f"{RESULTS_DIR}/02_logit_lens_data.csv")
    
    # 02 is "long form" (one row per layer per prompt).
    # We want prompt-level categories.
    # Get mapping: prompt_idx -> category
    cat_map = df_lens[['prompt_idx', 'category']].drop_duplicates().set_index('prompt_idx')
    
    # Define df_llama
    df_llama = df_base[df_base['model'] == "Llama-3.1-8B"].reset_index(drop=True)

    # Reconstruct 02 subset logic to map prompt_idx (0..N) to original data
    # 02 logic: n_samples=200 target.
    # correct = head(100), wrong = head(100).
    
    n_samples_target = 200
    n_half = n_samples_target // 2
    
    # Ensure strict sorting/state
    # 01 usually saved sequentially.
    
    correct_df = df_llama[df_llama['is_correct_pairwise'] == True].head(n_half)
    wrong_df = df_llama[df_llama['is_correct_pairwise'] == False].head(n_half)
    
    subset_df = pd.concat([correct_df, wrong_df])
    
    # subset_df is now the exact sequence processed by 02 loop
    # prompt_idx in 02 corresponds to iloc in this subset_df
    
    # Create mapping: 02_prompt_idx -> 01_row
    subset_df = subset_df.reset_index(drop=False) # Keep 'index' col from 01
    subset_df['reconstructed_idx'] = subset_df.index # 0, 1, 2...
    
    # Join 02 data with this subset on reconstructed_idx
    df_lens_reduced = cat_map.reset_index()
    
    joined = df_lens_reduced.merge(subset_df, left_on='prompt_idx', right_on='reconstructed_idx', how='inner')
    
    print(f"Joined {len(joined)} rows via Reconstruction.")
    
    # Check consistency again
    consistent = 0
    inconsistent = 0
    for i, row in joined.iterrows():
        cat = row['category']
        is_correct = row['is_correct_pairwise']
        
        # Logic:
        # Correct -> is_correct=True
        # Absent -> is_correct=False
        # Overwritten -> is_correct=False
        
        if cat == "Correct" and is_correct: consistent += 1
        elif cat != "Correct" and not is_correct: consistent += 1
        else: 
            inconsistent += 1
            # print(f"Mismatch: Cat {cat} vs Correct {is_correct}")
            
    print(f"Alignment Check: {consistent} consistent, {inconsistent} inconsistent")
    
    if inconsistent > 5: # Allow small margin for edge cases
        print("WARNING: Merge still suspect. Check N_SAMPLES logic.")
        return
        
    # Save Joined Data for Composite Plotting
    joined.to_csv(f"{RESULTS_DIR}/10_entropy_data.csv", index=False)
    print(f"Saved entropy data to {RESULTS_DIR}/10_entropy_data.csv")

    # 2. Entropy Analysis
    print("\n=== Experiment 2: Entropy Analysis ===")
    print("Hypothesis: Overwritten (Conflict) has Low Entropy (Confident), Absent (Unknown) has High Entropy.")
    
    stats = joined.groupby('category')['entropy'].describe()
    print(stats)
    
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=joined, x='category', y='entropy', order=["Correct", "Overwritten", "Absent"])
    plt.title("Entropy by Error Category")
    plt.savefig(f"{PLOTS_DIR}/10_entropy_by_category.png")
    
    # 3. Qualitative Audit
    print("\n=== Experiment 3: Qualitative Audit ===")
    overwritten_df = joined[joined['category'] == "Overwritten"]
    
    if len(overwritten_df) > 0:
        sample = overwritten_df.sample(min(10, len(overwritten_df)), random_state=42)
        print(f"Sampling {len(sample)} Overwritten examples:\n")
        
        for i, row in sample.iterrows():
            print(f"Q: {row['prompt']}")
            print(f"True: {row['target_true']} | False: {row['target_false']}")
            print(f"Pred: {row['prediction']}")
            print(f"Entropy: {row['entropy']:.3f} | LogitDiff: {row['logit_diff']:.3f}")
            # Heuristic check for Prior Bias
            # Is False answer a 'common' word or related entity?
            print("-" * 40)
            
    # 4. Trivial Baseline (Logit Gap)
    # Check if a simple threshold on `logit_diff` (final) predicts Hallucination better than Probe.
    # Note: `logit_diff` dictates the answer (by definition, since prediction = argmax).
    # But `logit_diff` magnitude implies confidence.
    # If magnitude is small -> Uncertain.
    # Probe uses MID layer.
    # If Probe AUC > Final Logit Magnitude AUC, then mid layer has MORE info than "Final Confidence".
    
if __name__ == "__main__":
    main()
