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
    # Load 11 CSV for categories
    try:
        df = pd.read_csv(f"{RESULTS_DIR}/11_rank_mass_Llama-3.1-8B.csv")
    except:
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
    
    # Filter for Errors
    errors = df[df['category'].isin(["True-Competitive", "Generic-Collapse"])]
    return errors

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
    except:
        return

    # Compute Margins
    margins = []
    
    print(f"Computing margins for {len(df)} errors...")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row['prompt']
        target = row['target_true']
        
        with torch.no_grad():
            logits = model(prompt)[0, -1, :]
            
            # Top-1 Logit
            top1_val, top1_id = torch.max(logits, dim=0)
            
            # True Logit
            true_id = model.to_single_token(target)
            if true_id is None:
                margins.append(np.nan)
                continue
                
            true_val = logits[true_id]
            
            # Margin = Top1 - True
            margin = (top1_val - true_val).item()
            margins.append(margin)
            
    df['margin'] = margins
    df = df.dropna(subset=['margin'])
    
    # Save for Paper Plots
    df.to_csv(f"{RESULTS_DIR}/17_margin_data.csv", index=False)
    print(f"Saved margin data to {RESULTS_DIR}/17_margin_data.csv")
    
    # Plot Distribution by Category
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=df, x='margin', hue='category', fill=True, common_norm=False, palette="mako")
    plt.title("Competition Margin Distribution\n(Logit_Top1 - Logit_True)")
    plt.xlabel("Margin (Logits)")
    plt.axvline(x=0, color='k', linestyle='--') # Should be all positive for errors
    plt.savefig(f"{PLOTS_DIR}/17_competition_margin.png")
    
    # Stats
    print("Mean Margin by Category:")
    print(df.groupby('category')['margin'].mean())

if __name__ == "__main__":
    main()
