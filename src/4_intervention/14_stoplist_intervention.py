import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from datasets import load_dataset
from tqdm import tqdm
import os

RESULTS_DIR = "/homes/55/arshia/project_1_hallucination/results"
PLOTS_DIR = "/homes/55/arshia/project_1_hallucination/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data_and_targets():
    # Load 11 CSV
    try:
        df = pd.read_csv(f"{RESULTS_DIR}/11_rank_mass_Llama-3.1-8B.csv")
    except:
        print("Could not load 11_rank_mass CSV. Please run 11 first.")
        return None
        
    ds = load_dataset("NeelNanda/counterfact-tracing", split="train")
    prompt_map = {}
    for item in ds:
        prompt_map[item['prompt']] = item['target_true']
    
    t_trues = []
    for prompt in df['prompt']:
        t_trues.append(prompt_map.get(prompt, None))
            
    df['target_true'] = t_trues
    df = df.dropna(subset=['target_true'])
    return df

def main():
    df = load_data_and_targets()
    if df is None: return

    # Load Model
    model_name = "meta-llama/Llama-3.1-8B"
    print(f"Loading {model_name}...")
    try:
        model = HookedTransformer.from_pretrained(
            model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Define Stoplist
    # Common generic start tokens
    stop_words = [" the", " a", " of", " in", " to", " is", " was", " The", " A", " an", " an"]
    stop_ids = []
    for w in stop_words:
        try:
            # Note: prepend space is crucial for Llama tokenizer?
            # to_single_token handles it if key matches. 
            # transformer_lens `to_single_token` typically expects the exact string key in vocab.
            token = model.to_single_token(w)
            if token is not None:
                stop_ids.append(token)
        except:
            pass
            
    # Add other variants if needed (e.g. without space)
    # Llama 3 tokenizer is tricky. Let's inspect a few common ones dynamically?
    # No, keep it simple.
    print(f"Stoplist ({len(stop_ids)} ids): {stop_words}")

    # Categories
    categories = ["True-Competitive", "Generic-Collapse"]
    
    results = []
    
    for cat in categories:
        subset = df[df['category'] == cat]
        if len(subset) == 0: continue
        
        sample_size = min(50, len(subset))
        subset = subset.sample(sample_size, random_state=42)
        
        print(f"Testing {len(subset)} samples for category: {cat}")
        
        for i, row in tqdm(subset.iterrows(), total=len(subset)):
            prompt = row['prompt']
            target_true = row['target_true']
            
            # Run Model
            with torch.no_grad():
                logits = model(prompt)
                next_token_logits = logits[0, -1, :]
                
                # Check Baseline (No intervention) - Rank 1
                base_id = torch.argmax(next_token_logits).item()
                base_str = model.to_string(base_id)
                
                # Intervention: Suppress Stoplist through -inf
                intervened_logits = next_token_logits.clone()
                intervened_logits[stop_ids] = -float('inf')
                
                int_id = torch.argmax(intervened_logits).item()
                int_str = model.to_string(int_id)
                
                # Check correctness
                # Strict Exact Match?
                # target_true often has space " Paris".
                # generated might be "Paris" (if suppression forced it).
                
                t_clean = target_true.strip()
                b_clean = base_str.strip()
                i_clean = int_str.strip()
                
                match_base = (t_clean == b_clean) # Should be False for these error cats
                match_int = (t_clean == i_clean) or (i_clean.startswith(t_clean)) # Allow substring start? "Parisian"
                
                results.append({
                    "category": cat,
                    "prompt": prompt,
                    "target": t_clean,
                    "base_pred": b_clean,
                    "int_pred": i_clean,
                    "match_base": match_base,
                    "match_int": match_int
                })
                
    # Analysis
    res_df = pd.DataFrame(results)
    
    stats = res_df.groupby('category')[['match_base', 'match_int']].mean().reset_index()
    print("\nRecovery Rates (Exact@1) with Stoplist Intervention:")
    print(stats)
    
    # Save Plot
    # Bar chart comparing Base vs Intervened
    melted = res_df.melt(id_vars=['category'], value_vars=['match_base', 'match_int'], var_name='Type', value_name='Correct')
    
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=melted, x='category', y='Correct', hue='Type', palette="coolwarm")
    
    # Add labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
        
    plt.title("Stoplist Intervention Efficacy\n(Recovery of True Entity via 'the/a' Suppression)")
    plt.ylabel("Exact@1 Accuracy")
    plt.ylim(0, 1.0)
    plt.savefig(f"{PLOTS_DIR}/14_stoplist_intervention.png")
    
    # Print examples
    successes = res_df[(res_df['match_base'] == False) & (res_df['match_int'] == True)]
    print(f"\nSuccessfully Recovered {len(successes)} examples:")
    if len(successes) > 0:
        print(successes[['category', 'target', 'base_pred', 'int_pred']].head(10))

if __name__ == "__main__":
    main()
