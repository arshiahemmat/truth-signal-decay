import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
import os

RESULTS_DIR = "/homes/55/arshia/project_1_hallucination/results"
PLOTS_DIR = "/homes/55/arshia/project_1_hallucination/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data_and_targets():
    # Load 11 CSV
    try:
        df_ranks = pd.read_csv(f"{RESULTS_DIR}/11_rank_mass_Llama-3.1-8B.csv")
    except:
        print("Could not load 11_rank_mass CSV. Please run 11 first.")
        return None, None
        
    # Load Dataset for Targets
    ds = load_dataset("NeelNanda/counterfact-tracing", split="train")
    
    # Init target cols
    df_ranks['target_true'] = str(np.nan)
    df_ranks['target_false'] = str(np.nan)
    
    # Create dictionary mapping from prompt to (true, false)
    # This is faster than iterrows
    prompt_map = {}
    print("Building Prompt Map...")
    for item in ds:
        prompt_map[item['prompt']] = (item['target_true'], item['target_false'])
        
    print("Mapping Targets...")
    found_count = 0
    t_trues = []
    t_falses = []
    for prompt in df_ranks['prompt']:
        if prompt in prompt_map:
            t_t, t_f = prompt_map[prompt]
            t_trues.append(t_t)
            t_falses.append(t_f)
            found_count += 1
        else:
            t_trues.append(None)
            t_falses.append(None)
            
    df_ranks['target_true'] = t_trues
    df_ranks['target_false'] = t_falses
    
    print(f"Matched {found_count} / {len(df_ranks)} prompts.")
    df_ranks = df_ranks.dropna(subset=['target_true', 'target_false'])
    return df_ranks

def main():
    df = load_data_and_targets()
    if df is None or len(df) == 0: return

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

    # Check for Unembedding Bias
    # Llama usually folds bias or has none.
    # W_U is [d_model, d_vocab]. b_U is [d_vocab].
    
    using_bos_proxy = False
    
    if model.b_U is not None:
        # Check if it's all zeros
        if torch.allclose(model.b_U, torch.zeros_like(model.b_U)):
            print("Model.b_U is all zeros. Using BOS-token logits as Prior Proxy.")
            using_bos_proxy = True
        else:
            print("Using Model.b_U directly.")
            bias_vector = model.b_U
    else:
        print("Model.b_U is None. Using BOS-token logits as Prior Proxy.")
        using_bos_proxy = True
        
    if using_bos_proxy:
        # Run BOS token through model
        # bos_token = model.tokenizer.bos_token_id # or just pass list
        # We want the output logits on "Nothing" or "Start".
        # Let's run on BOS.
        with torch.no_grad():
            # Run on empty string or BOS
            # model.to_tokens("") usually gives [BOS]
            logits = model(model.to_tokens("")) # shape [1, 1, vocab]
            bias_vector = logits[0, 0, :]
            
    # Compute Delta B
    print("Computing Delta B...")
    delta_bs = []
    
    for i, row in df.iterrows():
        t_true = row['target_true']
        t_false = row['target_false']
        
        try:
            id_true = model.to_single_token(t_true)
            id_false = model.to_single_token(t_false)
            
            b_true = bias_vector[id_true].item()
            b_false = bias_vector[id_false].item()
            
            delta_bs.append(b_false - b_true)
        except:
            delta_bs.append(np.nan)
            
    df['delta_b'] = delta_bs
    df = df.dropna(subset=['delta_b'])
    
    # Save Results
    df.to_csv(f"{RESULTS_DIR}/12_prior_baseline.csv", index=False)
    
    # Analysis
    # Compare "Correct" vs "Generic-Collapse" (aka Overwritten-ish)
    # The user asked for "Overwritten vs Correct".
    # In my taxonomy, "Generic-Collapse" + "True-Competitive" + "False-Selected" are the "Wrong" modes.
    # Let's group all Wrong as "Overwritten" for this top-level check, or split by category.
    # Let's do "Correct" vs "Wrong" first.
    
    df['is_correct'] = df['category'] == "Correct"
    
    # Plot KDE
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='delta_b', hue='category', common_norm=False, fill=True)
    plt.title(f"Prior Advantage (Delta B) by Error Category\nUsing {'BOS Logits' if using_bos_proxy else 'Unembedding Bias'}")
    plt.xlabel("Prior Advantage (Bias_False - Bias_True)")
    plt.savefig(f"{PLOTS_DIR}/12_prior_advantage_kde.png")
    
    # Metrics
    # AUC of Delta B predicting Wrong
    # Invert label: Target is "Wrong" (Overwritten).
    # Does High Delta B predict Error?
    auc = roc_auc_score(1 - df['is_correct'].astype(int), df['delta_b'])
    print(f"\nAUC of Delta Bias predicting Error: {auc:.3f}")
    
    # Distribution Stats
    print("\nMean Delta B by Category:")
    print(df.groupby('category')['delta_b'].mean())
    
    # Truth Seeking: Counterexamples
    # Find Overwritten cases where Delta B < 0 (Favors True)
    # i.e. Prior said "True", but Model said "False/Generic".
    # These are "Strong Overwriting".
    
    strong_overwrites = df[(df['category'] != "Correct") & (df['delta_b'] < -2.0)] # 2 logit systematic bias for True
    print(f"\nFound {len(strong_overwrites)} Strong Overwrites (Prior favors True, but Model failed):")
    if len(strong_overwrites) > 0:
        print(strong_overwrites[['prompt', 'target_true', 'target_false', 'delta_b', 'category', 'top1_final']].head(5))
        
if __name__ == "__main__":
    main()
