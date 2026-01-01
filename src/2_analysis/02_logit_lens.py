
import torch
import pandas as pd
import numpy as np
from transformer_lens import HookedTransformer
from datasets import load_dataset
from tqdm import tqdm
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
MODELS = {
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
    "Gemma-2-9B": "google/gemma-2-9b" 
}
# We might run this one model at a time due to memory
DATASET_NAME = "NeelNanda/counterfact-tracing"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_model(model_name_hf):
    print(f"Loading model: {model_name_hf}")
    try:
        model = HookedTransformer.from_pretrained(
            model_name_hf, 
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        return model
    except Exception as e:
        print(f"Failed to load {model_name_hf}: {e}")
        return None

def get_data_subset(model_name, n_samples=50):
    baseline_path = f"{RESULTS_DIR}/01_baseline_comparison.csv"
    if os.path.exists(baseline_path):
        df = pd.read_csv(baseline_path)
        # Filter for the specific model
        df = df[df['model'] == model_name]
        
        correct_df = df[df['is_correct_pairwise'] == True].head(n_samples // 2)
        wrong_df = df[df['is_correct_pairwise'] == False].head(n_samples // 2)
        return pd.concat([correct_df, wrong_df])
    else:
        # ... logic ...
        return pd.DataFrame() # Simplification 

# ...

def main():
    model_name = "meta-llama/Llama-3.1-8B-Instruct" 
    
    # Load data from baseline 
    df_data = get_data_subset(model_name, n_samples=200) 
    
    if len(df_data) == 0:
        print(f"No baseline data found for {model_name}. Run 01 first.")
        return

    model = load_model(model_name)
    
    if model:
        prompts = df_data['prompt'].tolist()
        t_true = df_data['target_true'].tolist()
        t_false = df_data['target_false'].tolist()
        
        lens_df = run_logit_lens(model, prompts, t_true, t_false)
        lens_df.to_csv(f"{RESULTS_DIR}/02_logit_lens_data_instruct.csv", index=False)
def run_logit_lens(model, prompts, answers, answers_false):
    # prompts: list of strings
    # answers: list of strings (correct)
    # answers_false: list of strings (hallucination target)
    
    n_layers = model.cfg.n_layers
    results = []
    
    print(f"Running Logit Lens on {len(prompts)} prompts...")
    
    for i, (prompt, ans, ans_false) in tqdm(enumerate(zip(prompts, answers, answers_false))):
        try:
            tokens = model.to_tokens(prompt)
            target_id = model.to_single_token(ans)
            target_false_id = model.to_single_token(ans_false)
            
            if isinstance(target_id, list) or isinstance(target_false_id, list): 
                continue

            with torch.no_grad():
                logits, cache = model.run_with_cache(tokens, names_filter=lambda x: x.endswith("resid_post"))
                
                final_logits = logits[0, -1, :]
                final_logit_diff = (final_logits[target_id] - final_logits[target_false_id]).item()
                is_correct = (final_logit_diff > 0)
                
                # Trajectory
                layer_diffs = []
                
                for layer in range(n_layers):
                    resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]
                    ln_resid = model.ln_final(resid)
                    layer_logits = model.unembed(ln_resid)
                    
                    diff = (layer_logits[target_id] - layer_logits[target_false_id]).item()
                    layer_diffs.append(diff)
                    
                    results.append({
                        "prompt_idx": i,
                        "layer": layer,
                        "logit_diff": diff,
                        "final_correct": is_correct
                    })
                
                # Classification (Per Prompt)
                max_diff = max(layer_diffs)
                if is_correct:
                    category = "Correct"
                elif max_diff > 0:
                    category = "Overwritten"
                else:
                    category = "Absent"
                    
                # Store category back into results (efficient way is post-processing, but let's just add it)
                # Actually, easier to store prompts metadata and merge
                
        except Exception as e:
            print(f"Error {e}")
            continue
            
    df = pd.DataFrame(results)
    
    # Classify Prompts
    # Group by prompt_idx
    prompt_stats = df.groupby('prompt_idx')['logit_diff'].max().reset_index().rename(columns={'logit_diff': 'max_diff'})
    final_stats = df[df['layer'] == n_layers-1][['prompt_idx', 'logit_diff']].rename(columns={'logit_diff': 'final_diff'})
    
    meta = pd.merge(prompt_stats, final_stats, on='prompt_idx')
    
    def classify(row):
        if row['final_diff'] > 0: return "Correct"
        if row['max_diff'] > 0: return "Overwritten"
        return "Absent"
        
    meta['category'] = meta.apply(classify, axis=1)
    
    # Merge back
    df = pd.merge(df, meta[['prompt_idx', 'category']], on='prompt_idx')
    
    # Print prevalence
    print("\nFailure Mode Decomposition:")
    print(meta['category'].value_counts(normalize=True))
    
    return df

def plot_logit_lens(df, model_name):
    # Plot Mean Logit Diff Trajectory split by Category
    
    plt.figure(figsize=(10, 6))
    
    # Palette
    palette = {"Correct": "green", "Overwritten": "orange", "Absent": "red"}
    
    sns.lineplot(data=df, x="layer", y="logit_diff", hue="category", style="category", 
                 palette=palette, markers=True, errorbar=('ci', 95))
                 
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title(f"Trajectory Analysis: {model_name}\n(Mean Logit Difference: True - False)")
    plt.ylabel("Logit Difference (True - False)")
    plt.xlabel("Layer Index")
    plt.legend(title="Mode")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{PLOTS_DIR}/logit_lens_trajectory_{model_name.replace('/', '_')}.png")
    plt.close()

def main():
    model_label = "Llama-3.1-8B-Instruct"
    model_hf = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Load data from baseline using LABEL
    df_data = get_data_subset(model_label, n_samples=200) 
    
    if len(df_data) == 0:
        print(f"No baseline data found for {model_label}. Run 01 first.")
        # Debug
        if os.path.exists(f"{RESULTS_DIR}/01_baseline_comparison.csv"):
            df = pd.read_csv(f"{RESULTS_DIR}/01_baseline_comparison.csv")
            print(f"Available models: {df['model'].unique()}")
        return

    model = load_model(model_hf)
    
    if model:
        prompts = df_data['prompt'].tolist()
        t_true = df_data['target_true'].tolist()
        t_false = df_data['target_false'].tolist()
        
        lens_df = run_logit_lens(model, prompts, t_true, t_false)
        lens_df.to_csv(f"{RESULTS_DIR}/02_logit_lens_data_instruct.csv", index=False)
        
        plot_logit_lens(lens_df, model_label)
        print("Logit Lens analysis complete.")

if __name__ == "__main__":
    main()
