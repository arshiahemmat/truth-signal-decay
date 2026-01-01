
print("DEBUG: Starting 03_causal_trace.py imports...")
import torch
import pandas as pd
import numpy as np
from transformer_lens import HookedTransformer
from tqdm import tqdm
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
print("DEBUG: Imports complete.")

# Configuration
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_model(model_name_hf):
    print(f"DEBUG: Loading model: {model_name_hf}")
    try:
        model = HookedTransformer.from_pretrained(
            model_name_hf, 
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        print("DEBUG: Model loaded successfully.")
        return model
    except Exception as e:
        print(f"DEBUG: Failed to load {model_name_hf}: {e}")
        return None

def analyze_logit_attribution(model, prompt, target_true, target_false):
    tokens = model.to_tokens(prompt)
    target_true_id = model.to_single_token(target_true)
    target_false_id = model.to_single_token(target_false)
    
    if isinstance(target_true_id, list) or isinstance(target_false_id, list):
        return None, None

    with torch.no_grad():
        # Use hook_z (output of heads before W_O)
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x.endswith("hook_z")) 
        
    W_U = model.W_U # [d_model, d_vocab]
    W_O = model.W_O # [n_layers, n_heads, d_head, d_model]
    
    # Directions in d_model/residual space
    dir_true = W_U[:, target_true_id] # [d_model]
    dir_false = W_U[:, target_false_id] # [d_model]
    
    layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    
    head_attr_true = torch.zeros((layers, n_heads))
    head_attr_false = torch.zeros((layers, n_heads))
    
    for layer in range(layers):
        # hook_z: [batch, seq, head, d_head]
        # We take last token position: -1
        # shape: [head, d_head]
        z = cache[f"blocks.{layer}.attn.hook_z"][0, -1, :, :] 
        
        # Project through W_O for this layer
        # W_O[layer]: [head, d_head, d_model]
        # z: [head, d_head]
        # result = einsum(z, W_O) -> [head, d_model]
        
        # Manually:
        W_O_layer = W_O[layer] # [head, d_head, d_model]
        
        # z is [H, D_h]
        # W_O_layer is [H, D_h, D_m]
        # output is [H, D_m]
        head_result = torch.einsum("hd,hdm->hm", z, W_O_layer)
        
        attr_t = torch.matmul(head_result, dir_true) # [head]
        attr_f = torch.matmul(head_result, dir_false)
        
        head_attr_true[layer] = attr_t.float().cpu() # CAST TO FLOAT
        head_attr_false[layer] = attr_f.float().cpu()
        
    return head_attr_true, head_attr_false

def analyze_attention_pattern(model, prompt, head_l, head_h):
    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(tokens, names_filter=lambda x: x.endswith(f"blocks.{head_l}.attn.hook_pattern"))
    # [batch, head, query, key]
    pattern = cache[f"blocks.{head_l}.attn.hook_pattern"][0, head_h, :, :].detach().cpu().numpy()
    
    # Save plot
    plt.figure(figsize=(8, 8))
    sns.heatmap(pattern, cmap="Blues")
    plt.title(f"Attention Pattern: L{head_l}H{head_h}\nPrompt: {prompt[:20]}...")
    plt.savefig(f"{PLOTS_DIR}/03_attn_pattern_L{head_l}H{head_h}.png")
    plt.close()

def main():
    print("DEBUG: Entering main...")
    model_name = "meta-llama/Llama-3.1-8B"
    model = load_model(model_name)
    if not model: return

    # Load categorized data from Phase 2
    lens_csv = f"{RESULTS_DIR}/02_logit_lens_data.csv"
    if not os.path.exists(lens_csv):
        print("Run 02 first.")
        return
        
    df_lens = pd.read_csv(lens_csv)
    # Get unique prompt indices per category
    # Each prompt maps to one category
    df_meta = df_lens[['prompt_idx', 'category']].drop_duplicates()
    
    # We also need the original texts. Load baseline.
    baseline_csv = f"{RESULTS_DIR}/01_baseline_comparison.csv"
    df_base = pd.read_csv(baseline_csv)
    df_base = df_base[df_base['model'] == "Llama-3.1-8B"].reset_index(drop=True)
    
    # Sample Overwritten (for identification & benefit) and Correct (for harm)
    # n=20 each?
    idx_overwritten = df_meta[df_meta['category'] == "Overwritten"]['prompt_idx'].tolist()[:20]
    idx_correct = df_meta[df_meta['category'] == "Correct"]['prompt_idx'].tolist()[:20]
    
    if not idx_overwritten: 
        print("No Overwritten errors found!")
        return
        
    print(f"Identifying Global Suppressors using {len(idx_overwritten)} Overwritten examples...")
    
    # 1. Identify Global Suppressors
    agg_attr = None
    count = 0
    
    for idx in tqdm(idx_overwritten):
        row = df_base.iloc[idx]
        attr_t, attr_f = analyze_logit_attribution(model, row['prompt'], row['target_true'], row['target_false'])
        if attr_t is None: continue
        
        net = (attr_t - attr_f).detach().cpu().numpy()
        if agg_attr is None: agg_attr = np.zeros_like(net)
        agg_attr += net
        count += 1
        
    avg_attr = agg_attr / count
    # Find top-3 negative heads
    flat_indices = np.argsort(avg_attr.flatten())
    top_suppressors = []
    for i in range(3):
        flat_idx = flat_indices[i]
        l = flat_idx // model.cfg.n_heads
        h = flat_idx % model.cfg.n_heads
        # Check sign
        if avg_attr[l, h] < 0:
            top_suppressors.append((l, h))
            
    print(f"Top Suppressors: {top_suppressors}")
    np.save(f"{RESULTS_DIR}/mean_net_attr_overwritten.npy", avg_attr)
    
    # 2. Ablation Comparison
    # Define Ablation Hook
    def make_ablation_hook(heads_to_ablate):
        # List of (l, h)
        def hook_fn(value, hook):
            layer = hook.layer()
            for (l, h) in heads_to_ablate:
                if l == layer:
                    value[:, -1, h, :] = 0.0
            return value
        return hook_fn

    # Setup hooks for all relevant layers
    # We can attach one hook to each layer's z
    # Or cleaner: attach to specific locations
    
    def measure_ablation_effect(indices, desc):
        deltas = []
        for idx in tqdm(indices, desc=desc):
            row = df_base.iloc[idx]
            prompt = row['prompt']
            t_true = model.to_single_token(row['target_true'])
            t_false = model.to_single_token(row['target_false'])
            
            # Clean
            with torch.no_grad():
                logits_clean = model(prompt, return_type="logits")
                clean_diff = (logits_clean[0, -1, t_true] - logits_clean[0, -1, t_false]).item()
                
                # Ablated
                with model.hooks(fwd_hooks=[]):
                    # Add hooks for each suppressor layer
                    # Group by layer
                    heads_by_layer = {}
                    for (l, h) in top_suppressors:
                        if l not in heads_by_layer: heads_by_layer[l] = []
                        heads_by_layer[l].append((l, h))
                        
                    for l, heads in heads_by_layer.items():
                        model.add_hook(f"blocks.{l}.attn.hook_z", make_ablation_hook(heads))
                        
                    logits_abl = model(prompt, return_type="logits")
                    abl_diff = (logits_abl[0, -1, t_true] - logits_abl[0, -1, t_false]).item()
                    
                deltas.append(abl_diff - clean_diff)
        return deltas

    print("Measuring Ablation Impact...")
    gain_overwritten = measure_ablation_effect(idx_overwritten, "Ablation on Overwritten")
    gain_correct = measure_ablation_effect(idx_correct, "Ablation on Correct")
    
    print(f"Mean Gain (Overwritten): {np.mean(gain_overwritten):.4f}")
    print(f"Mean Gain (Correct): {np.mean(gain_correct):.4f}") # Expect negative or small positive
    
    # Save results
    with open(f"{RESULTS_DIR}/ablation_summary.txt", "w") as f:
        f.write(f"Top Suppressors: {top_suppressors}\n")
        f.write(f"Mean Gain (Overwritten): {np.mean(gain_overwritten):.4f}\n")
        f.write(f"Mean Gain (Correct): {np.mean(gain_correct):.4f}\n")
        
    # 3. Mechanism Check (Attention Pattern)
    if top_suppressors:
        l, h = top_suppressors[0]
        # Pick one overwritten example
        ex_idx = idx_overwritten[0]
        prompt = df_base.iloc[ex_idx]['prompt']
        analyze_attention_pattern(model, prompt, l, h)

if __name__ == "__main__":
    main()
