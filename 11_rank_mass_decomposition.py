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
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_model(model_name_hf):
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

def get_ranks_and_probs(model, prompt, target_true, target_false, target_layer):
    # Convert targets to tokens
    try:
        t_true_id = model.to_single_token(target_true)
        t_false_id = model.to_single_token(target_false)
    except:
        return None

    with torch.no_grad():
        hook_name = f"blocks.{target_layer}.hook_resid_post"
        logits_final, cache = model.run_with_cache(prompt, names_filter=lambda x: x == hook_name)
        
        # Final Layer Stats
        final_logits = logits_final[0, -1, :]
        probs_final = torch.softmax(final_logits, dim=0)
        
        # Mid Layer Stats (Virtual)
        resid = cache[hook_name][0, -1, :]
        ln_resid = model.ln_final(resid)
        mid_logits = model.unembed(ln_resid)
        probs_mid = torch.softmax(mid_logits, dim=0)
        
        # Compute Ranks (descending sort order)
        # We want the rank of the specific token ID.
        # sort indices
        sorted_ids_final = torch.argsort(final_logits, descending=True)
        rank_true_final = (sorted_ids_final == t_true_id).nonzero().item()
        rank_false_final = (sorted_ids_final == t_false_id).nonzero().item()
        top1_final_id = sorted_ids_final[0].item()
        top1_final_str = model.to_string(top1_final_id)
        
        sorted_ids_mid = torch.argsort(mid_logits, descending=True)
        rank_true_mid = (sorted_ids_mid == t_true_id).nonzero().item()
        rank_false_mid = (sorted_ids_mid == t_false_id).nonzero().item()
        top1_mid_id = sorted_ids_mid[0].item()
        top1_mid_str = model.to_string(top1_mid_id)
        
        return {
            "rank_true_final": rank_true_final,
            "rank_false_final": rank_false_final,
            "p_true_final": probs_final[t_true_id].item(),
            "p_false_final": probs_final[t_false_id].item(),
            "top1_final": top1_final_str,
            "rank_true_mid": rank_true_mid,
            "rank_false_mid": rank_false_mid,
            "p_true_mid": probs_mid[t_true_id].item(),
            "p_false_mid": probs_mid[t_false_id].item(),
            "top1_mid": top1_mid_str
        }

def analyze_model(model_name, label, layer, n_samples=300):
    print(f"Analyzing {label}...")
    model = load_model(model_name)
    if not model: return
    
    # Load dataset
    ds = load_dataset("NeelNanda/counterfact-tracing", split="train")
    
    results = []
    
    # Process n_samples
    count = 0
    for item in tqdm(ds):
        if count >= n_samples: break
        
        prompt = item['prompt']
        target_true = item['target_true']
        target_false = item['target_false']
        
        stats = get_ranks_and_probs(model, prompt, target_true, target_false, layer)
        if stats:
            stats['label'] = label
            stats['prompt'] = prompt
            results.append(stats)
            count += 1
            
    df = pd.DataFrame(results)
    
    # Categorize Errors
    # 1. False-selected: rank_false <= 5 and rank_true > rank_false
    # 2. True-competitive but lost: rank_true <= 10 (but not top-1?) 
    #    Actually "Wrong" implies top-1 != true. 
    #    Metric: Rank True in Top 10.
    # 3. Generic-collapse: rank_true > 50 and rank_false > 50
    
    def categorize(row):
        # We care about Final Layer Errors
        # Was it Correct? (Top-1 == True is one def, or Pairwise True > False)
        # Let's use Top-1 correctness for strict "Overwriting" definition, 
        # or Pairwise for consistency with report.
        # User defined types for *wrong* examples. 
        # Let's verify "Wrong first".
        
        # Strict Wrong: True is NOT Top-1.
        if row['rank_true_final'] == 0:
            return "Correct"
        
        r_true = row['rank_true_final']
        r_false = row['rank_false_final']
        
        if r_true > 50 and r_false > 50:
            return "Generic-Collapse"
        elif r_false <= 5 and r_true > r_false:
            return "False-Selected"
        elif r_true <= 10:
            return "True-Competitive"
        else:
            return "Other" # E.g. True Rank 20, False Rank 30 (Weak Collapse)

    df['category'] = df.apply(categorize, axis=1)
    
    # Save CSV
    df.to_csv(f"{RESULTS_DIR}/11_rank_mass_{label}.csv", index=False)
    
    # Plotting Stacked Bar
    wrong_df = df[df['category'] != "Correct"]
    counts = wrong_df['category'].value_counts(normalize=True)
    print(f"\nError Distribution for {label}:")
    print(counts)
    
    # Save Generic Tokens
    generic_df = wrong_df[wrong_df['category'] == "Generic-Collapse"]
    print(f"\nTop Generic Tokens for {label}:")
    print(generic_df['top1_final'].value_counts().head(10))
    
    del model
    torch.cuda.empty_cache()
    return df

def main():
    # Run for Llama and Gemma Base
    df_llama = analyze_model("meta-llama/Llama-3.1-8B", "Llama-3.1-8B", 20)
    df_gemma = analyze_model("google/gemma-2-9b", "Gemma-2-9B", 25)
    
    # Generate combined plots
    if df_llama is not None and df_gemma is not None:
        plot_error_breakdown(df_llama, df_gemma)

def plot_error_breakdown(df1, df2):
    # Combine only errors
    err1 = df1[df1['category'] != "Correct"].copy()
    err2 = df2[df2['category'] != "Correct"].copy()
    
    combined = pd.concat([err1, err2])
    
    # Calculate percentages
    props = combined.groupby(['label', 'category']).size().reset_index(name='count')
    totals = props.groupby('label')['count'].transform('sum')
    props['percentage'] = props['count'] / totals
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=props, x='label', y='percentage', hue='category', palette="viridis")
    plt.title("Error Mode Decomposition (Generic Collapse vs False Belief)")
    plt.ylabel("Proportion of Errors")
    plt.ylim(0, 1.0)
    plt.savefig(f"{PLOTS_DIR}/11_error_mode_decomposition.png")
    print(f"Main figure saved to {PLOTS_DIR}/11_error_mode_decomposition.png")

if __name__ == "__main__":
    main()
