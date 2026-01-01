import pandas as pd
import numpy as np
import joblib
import os
from transformer_lens import HookedTransformer
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

RESULTS_DIR = "/homes/55/arshia/project_1_hallucination/results"
PLOTS_DIR = "/homes/55/arshia/project_1_hallucination/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

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

def get_logits_and_activations(model, prompt, target_layer, t_true_id, t_false_id):
    with torch.no_grad():
        hook_name = f"blocks.{target_layer}.hook_resid_post"
        logits, cache = model.run_with_cache(prompt, names_filter=lambda x: x == hook_name)
        
        # Final Logits
        final_logits = logits[0, -1, :]
        
        # Mid Activations
        resid = cache[hook_name][0, -1, :]
        
        # Mid Logits (Early Exit)
        ln_resid = model.ln_final(resid)
        mid_logits = model.unembed(ln_resid)
        
        # Extraction feature for probe
        act = resid.float().cpu().numpy()
        
    return final_logits, mid_logits, act

def run_mixing_experiment(model, df, probe_path, layer, label):
    if not os.path.exists(probe_path):
        print(f"Probe not found: {probe_path}")
        return None
        
    clf = joblib.load(probe_path)
    prompts = df['prompt'].tolist()
    baseline_correct = df['is_correct_pairwise'].values
    
    # Store data for fast mixing sweep
    data_points = []
    
    print(f"Processing {label}...")
    for i, prompt in tqdm(enumerate(prompts), total=len(prompts)):
        t_true = df.iloc[i]['target_true']
        t_false = df.iloc[i]['target_false']
        
        try:
            t_true_id = model.to_single_token(t_true)
            t_false_id = model.to_single_token(t_false)
            if isinstance(t_true_id, list): t_true_id = t_true_id[0]
            if isinstance(t_false_id, list): t_false_id = t_false_id[0]
            
            final_logits, mid_logits, act = get_logits_and_activations(model, prompt, layer, t_true_id, t_false_id)
            
            # Predict Probe
            prob = clf.predict_proba(act.reshape(1, -1))[0, 1]
            
            # Save relevant logits (only true/false needed for pairwise)
            f_true = final_logits[t_true_id].item()
            f_false = final_logits[t_false_id].item()
            m_true = mid_logits[t_true_id].item()
            m_false = mid_logits[t_false_id].item()
            
            data_points.append({
                "f_true": f_true, "f_false": f_false,
                "m_true": m_true, "m_false": m_false,
                "prob": prob,
                "base_correct": baseline_correct[i]
            })
        except Exception as e:
            print(f"Error {i}: {e}")
            continue
            
    # Sweep Alpha
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] 
    # 0.0 = Final (Baseline), 1.0 = Mid (Hard Early Exit)
    
    threshold = 0.9 # High precision operating point
    
    results = []
    for alpha in alphas:
        b_gain = 0
        c_loss = 0
        n_intervened = 0
        
        for dp in data_points:
            # Check Intervention
            if dp['prob'] > threshold:
                n_intervened += 1
                
                # Mixing
                # logits_mixed = alpha * mid + (1-alpha) * final
                # We only need specific logits
                mix_true = alpha * dp['m_true'] + (1-alpha) * dp['f_true']
                mix_false = alpha * dp['m_false'] + (1-alpha) * dp['f_false']
                
                is_correct = (mix_true > mix_false)
                
                # Flip Accounting
                was_correct = dp['base_correct']
                
                if not was_correct and is_correct:
                    b_gain += 1
                elif was_correct and not is_correct:
                    c_loss += 1
            
        net_gain = b_gain - c_loss
        results.append({
            "model": label,
            "alpha": alpha,
            "net_gain": net_gain,
            "b_gain": b_gain,
            "c_loss": c_loss,
            "n_intervened": n_intervened
        })
        
    return pd.DataFrame(results)

def main():
    models = [
        ("Llama-3.1-8B", "meta-llama/Llama-3.1-8B", 20),
        ("Gemma-2-9B", "google/gemma-2-9b", 25)
    ]
    
    all_res = []
    df_all = pd.read_csv(f"{RESULTS_DIR}/01_baseline_comparison.csv")
    
    for label, hf_name, layer in models:
        df_model = df_all[df_all['model'] == label]
        probe_path = f"{RESULTS_DIR}/probe_{label}_L{layer}.pkl"
        
        model = load_model(hf_name)
        if model:
            res = run_mixing_experiment(model, df_model, probe_path, layer, label)
            if res is not None:
                all_res.append(res)
            del model
            torch.cuda.empty_cache()
            
    if all_res:
        final_df = pd.concat(all_res)
        final_df.to_csv(f"{RESULTS_DIR}/08_logit_mixing.csv", index=False)
        print("\nLogit Mixing Results:")
        print(final_df)
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=final_df, x="alpha", y="net_gain", hue="model", marker="o")
        plt.axhline(0, color='black', linewidth=1)
        plt.title(f"Logit Mixing Efficacy (Threshold={0.9})")
        plt.xlabel("Mixing Alpha (0=Final, 1=Early Exit)")
        plt.ylabel("Net Gain")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{PLOTS_DIR}/08_logit_mixing.png")

if __name__ == "__main__":
    main()
