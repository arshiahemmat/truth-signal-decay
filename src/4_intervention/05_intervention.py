
print("DEBUG: Starting 05_intervention.py imports...")
import torch
import pandas as pd
import numpy as np
from transformer_lens import HookedTransformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from tqdm import tqdm
print("DEBUG: Imports complete.")

# Configuration
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")

def load_model(model_name_hf):
    print(f"DEBUG: Loading model: {model_name_hf}")
    try:
        model = HookedTransformer.from_pretrained(
            model_name_hf, 
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        return model
    except:
        return None

def early_exit_inference(model, prompt, exit_layer, target_true_id=None, target_false_id=None):
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        hook_name = f"blocks.{exit_layer}.hook_resid_post"
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x == hook_name)
        resid = cache[hook_name][0, -1, :] 
        ln_resid = model.ln_final(resid)
        logits = model.unembed(ln_resid)
        
        # Top-1
        pred_id = torch.argmax(logits).item()
        pred_str = model.to_string(pred_id)
        
        # Pairwise
        pairwise_correct = False
        if target_true_id is not None and target_false_id is not None:
             true_logit = logits[target_true_id].item()
             false_logit = logits[target_false_id].item()
             pairwise_correct = (true_logit > false_logit)
        
    return pred_str, pred_id, pairwise_correct

def run_intervention(model_name, label):
    print(f"\nRunning Intervention for {label}...")
    model = load_model(model_name)
    if not model: return
    
    target_layer = 20 if "Llama" in label else 25
    target_layer = 20 if "Llama" in label else 25
    probe_path = f"{RESULTS_DIR}/probe_{label}_L{target_layer}.pkl"
    
    if not os.path.exists(probe_path):
        print(f"Probe not found for {label}. Trying Base probe for transfer...")
        # Try base model name
        base_label = label.replace("-Instruct", "")
        probe_path = f"{RESULTS_DIR}/probe_{base_label}_L{target_layer}.pkl"
        
        if not os.path.exists(probe_path):
             print(f"Base probe also not found at {probe_path}. Aborting.")
             return
        else:
             print(f"Using Base probe: {probe_path}")
            
    clf = joblib.load(probe_path)
    
    baseline_csv = f"{RESULTS_DIR}/01_baseline_comparison.csv"
    if not os.path.exists(baseline_csv):
        return
        
    df = pd.read_csv(baseline_csv)
    df = df[df['model'] == label]
    
    prompts = df['prompt'].tolist()
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    tradeoff_results = []
    
    print("Extracting activations...")
    activations = []
    
    for prompt in tqdm(prompts):
        try:
            with torch.no_grad():
                hook_name = f"blocks.{target_layer}.hook_resid_post"
                _, cache = model.run_with_cache(prompt, names_filter=lambda x: x == hook_name)
                act = cache[hook_name][0, -1, :].float().cpu().numpy()
                activations.append(act)
        except:
            activations.append(None)
            
    X = np.array([a for a in activations if a is not None])
    probs = clf.predict_proba(X)[:, 1] 
    
    print("Computing Early Exit predictions...")
    early_outcomes = []
    
    for i, prompt in tqdm(enumerate(prompts), total=len(prompts)):
        t_true = df.iloc[i]['target_true']
        t_false = df.iloc[i]['target_false']
        t_true_id = model.to_single_token(t_true)
        t_false_id = model.to_single_token(t_false)
        
        if isinstance(t_true_id, list): t_true_id = t_true_id[0]
        if isinstance(t_false_id, list): t_false_id = t_false_id[0]
        
        pred_str, pred_id, pw_correct = early_exit_inference(model, prompt, target_layer, t_true_id, t_false_id)
        early_outcomes.append({
            "pred_str": pred_str, 
            "is_pairwise_correct": pw_correct
        })
        
    final_pairwise = np.array(df['is_correct_pairwise'].tolist()) # Baseline correctness
    
    # Add Baseline threshold (1.1 = Never Intervene coverage=0)
    thresholds = [1.1, 0.9, 0.8, 0.7, 0.6, 0.5]
    
    tradeoff_results = []
    
    for T in thresholds:
        n_intervened = 0
        
        # Determine Intervention Decisions
        # If T > 1.0, never intervene.
        if T > 1.0:
            intervention_mask = np.zeros(len(prompts), dtype=bool)
        else:
            intervention_mask = (probs > T)
            valid_mask = np.array([a is not None for a in activations])
            intervention_mask = intervention_mask & valid_mask
            
        n_intervened = np.sum(intervention_mask)
        coverage = n_intervened / len(prompts)
        
        # Monitor Outcomes
        early_correct = np.array([e['is_pairwise_correct'] for e in early_outcomes])
        current_correct = np.where(intervention_mask, early_correct, final_pairwise)
        
        acc = np.mean(current_correct)
        
        # Flip Accounting (Monitor)
        b_flips = (~final_pairwise & current_correct).sum()
        c_flips = (final_pairwise & ~current_correct).sum()
        net_gain = b_flips - c_flips
        
        # --- Killer Baseline: Random Gating ---
        # Select n_intervened indices randomly (averaged over 5 seeds)
        random_net_gains = []
        if n_intervened > 0 and n_intervened < len(prompts):
            for seed in range(5):
                rng = np.random.RandomState(seed + 42)
                random_mask = np.zeros(len(prompts), dtype=bool)
                random_indices = rng.choice(len(prompts), n_intervened, replace=False)
                random_mask[random_indices] = True
                
                rand_correct = np.where(random_mask, early_correct, final_pairwise)
                
                b_rand = (~final_pairwise & rand_correct).sum()
                c_rand = (final_pairwise & ~rand_correct).sum()
                random_net_gains.append(b_rand - c_rand)
            avg_random_net_gain = np.mean(random_net_gains)
        else:
            # If 0 or All, Random == Monitor
            avg_random_net_gain = net_gain
            
        # --- Baseline: Always Early ---
        # If we exited on EVERYTHING
        always_early_correct = early_correct
        b_always = (~final_pairwise & always_early_correct).sum()
        c_always = (final_pairwise & ~always_early_correct).sum()
        always_net_gain = b_always - c_always
        
        tradeoff_results.append({
            "model": label,
            "threshold": T, 
            "accuracy": acc, 
            "intervention_rate": coverage,
            "b_gain": b_flips,
            "c_loss": c_flips,
            "net_gain": net_gain,
            "net_gain_per_intervention": net_gain / n_intervened if n_intervened > 0 else 0.0,
            "random_net_gain": avg_random_net_gain,
            "always_early_net_gain": always_net_gain
        })
        
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(tradeoff_results)

def main():
    models = {
        "Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
        "Gemma-2-9B": "google/gemma-2-9b",
        "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct"
    }
    
    all_results = []
    for label, hf_name in models.items():
        df_res = run_intervention(hf_name, label)
        if df_res is not None:
            all_results.append(df_res)
            
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(f"{RESULTS_DIR}/05_intervention_tradeoff_comprehensive.csv", index=False)
        print(final_df[['model', 'threshold', 'intervention_rate', 'net_gain', 'random_net_gain']])
        
        plt.figure(figsize=(10, 6))
        
        # Plot Net Gain vs Coverage
        # Solid line = Monitor, Dashed = Random
        
        # We need to melt or iterate to plot both
        # Easiest: Iterate models and plot manually
        colors = {"Llama-3.1-8B": "blue", "Gemma-2-9B": "green", "Llama-3.1-8B-Instruct": "orange"}
        
        for model in final_df['model'].unique():
            subset = final_df[final_df['model'] == model]
            color = colors.get(model, "gray")
            
            # Monitor
            sns.lineplot(data=subset, x="intervention_rate", y="net_gain", 
                         color=color, label=f"{model} (Monitor)", marker="o")
            
            # Random
            sns.lineplot(data=subset, x="intervention_rate", y="random_net_gain", 
                         color=color, linestyle="--", alpha=0.5, label=f"{model} (Random)")
            
        plt.title("Intervention Efficacy: Monitor vs Random Baseline")
        plt.xlabel("Intervention Rate (Coverage)")
        plt.ylabel("Net Gain (Corrected Answers)")
        plt.axhline(0, color='black', linewidth=1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{PLOTS_DIR}/05_intervention_killer_baseline.png")
    
if __name__ == "__main__":
    main()
