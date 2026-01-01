import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from transformer_lens import HookedTransformer
import torch
from tqdm import tqdm

RESULTS_DIR = "/homes/55/arshia/project_1_hallucination/results"

def load_model(model_name_hf):
    try:
        model = HookedTransformer.from_pretrained(
            model_name_hf, 
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        return model
    except:
        return None

def get_fp_stats(model, df, probe_path, layer, label):
    if not os.path.exists(probe_path):
        print(f"Probe not found: {probe_path}")
        return []
        
    clf = joblib.load(probe_path)
    prompts = df['prompt'].tolist()
    baseline_correct = df['is_correct_pairwise'].values # Ground Truth
    
    # Extract Activations
    activations = []
    print(f"Extracting for {label}...")
    for prompt in tqdm(prompts):
        try:
            with torch.no_grad():
                hook_name = f"blocks.{layer}.hook_resid_post"
                _, cache = model.run_with_cache(prompt, names_filter=lambda x: x == hook_name)
                act = cache[hook_name][0, -1, :].float().cpu().numpy()
                activations.append(act)
        except:
            activations.append(None)
            
    X = np.array([a for a in activations if a is not None])
    probs = clf.predict_proba(X)[:, 1]
    
    valid_mask = np.array([a is not None for a in activations])
    y_true = baseline_correct[valid_mask]
    
    # Analyze Thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    stats = []
    
    for T in thresholds:
        # Flagged: Probe > T
        flagged_mask = (probs > T)
        k = np.sum(flagged_mask)
        
        # FP: Flagged AND Originally Correct
        # (Meaning the monitor thinks it's Overwritten (Wrong), but it was actually Right)
        # "Overwritten" means "Mid Correct -> Final Wrong".
        # If Ground Truth is "Correct" (Final Right), then it is NOT Overwritten.
        # So any Flag on a Correct Final is a False Positive.
        fp_mask = flagged_mask & y_true
        fp_count = np.sum(fp_mask)
        
        # 95% Upper Bound on FP Rate (Rule of Three / Clopper-Pearson approx)
        # If FP=0, p <= 1 - 0.05^(1/k)
        # General: Beta distribution or simple approx.
        # User explicitly asked for: p <= 1 - 0.05^(1/k) (for 0 FP case?)
        # Let's use the exact formula: 1 - alpha**(1/k) if FP=0.
        # If FP > 0, we report FP count.
        
        if k > 0:
            if fp_count == 0:
                # 95% Confidence that FP rate < ub
                ub = 1 - (0.05)**(1/k)
            else:
                # Naive rate + margin? User just said "report k flagged, FP=0, add bound"
                # If FP > 0, the bound is less relevant than the actual rate.
                ub = fp_count / k # Point estimate
        else:
            ub = 0.0
            
        stats.append({
            "model": label,
            "threshold": T,
            "k_flagged": k,
            "fp_count": fp_count,
            "fp_rate_bound": ub if fp_count == 0 else None,
            "observed_fp_rate": fp_count / k if k > 0 else 0.0
        })
        
    return stats

def main():
    models = [
        ("Llama-3.1-8B", "meta-llama/Llama-3.1-8B", 20),
        ("Gemma-2-9B", "google/gemma-2-9b", 25)
    ]
    
    full_stats = []
    
    df_all = pd.read_csv(f"{RESULTS_DIR}/01_baseline_comparison.csv")
    
    for label, hf_name, layer in models:
        df_model = df_all[df_all['model'] == label]
        probe_path = f"{RESULTS_DIR}/probe_{label}_L{layer}.pkl"
        
        # Load Model
        model = load_model(hf_name)
        if model:
            stats = get_fp_stats(model, df_model, probe_path, layer, label)
            full_stats.extend(stats)
            del model
            torch.cuda.empty_cache()
            
    res_df = pd.DataFrame(full_stats)
    res_df.to_csv(f"{RESULTS_DIR}/07_stats_rigor.csv", index=False)
    print("\nStatistical Rigor Results:")
    print(res_df)

if __name__ == "__main__":
    main()
