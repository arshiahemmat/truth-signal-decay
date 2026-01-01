
print("DEBUG: Starting 04_train_probe.py imports...")
import torch
import pandas as pd
import numpy as np
from transformer_lens import HookedTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
print("DEBUG: Imports complete.")

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
        return model
    except Exception as e:
        print(f"DEBUG: Failed to load {model_name_hf}: {e}")
        return None

def extract_activations(model, prompts, target_layer, n_samples=1000):
    print(f"DEBUG: Extracting activations from Layer {target_layer} for {len(prompts)} samples...")
    activations = []
    
    for prompt in prompts[:n_samples]:
        try:
            with torch.no_grad():
                hook_name = f"blocks.{target_layer}.hook_resid_post"
                _, cache = model.run_with_cache(prompt, names_filter=lambda x: x == hook_name)
                # CAST TO FLOAT BEFORE NUMPY
                act = cache[hook_name][0, -1, :].float().cpu().numpy()
                activations.append(act)
        except Exception as e:
            print(f"Error extraction: {e}")
            continue
            
    return np.array(activations)

def train_probe_and_eval(X, y, df):
    print("DEBUG: Training Logistic Regression Probe...")
    # Split
    # We want to keep X aligned with df to extract baselines
    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, indices, test_size=0.2, random_state=42)
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Predictions (Probe)
    y_pred_prob = clf.predict_proba(X_test)[:, 1] 
    
    test_df = df.iloc[idx_test]
    entropy_scores = test_df['entropy'].values
    
    return clf, y_test, y_pred_prob, entropy_scores

def plot_metrics(y_test, probe_scores, entropy_scores, model_name, layer):
    # ROC
    plt.figure(figsize=(10, 5))
    
    # Probe
    fpr, tpr, _ = roc_curve(y_test, probe_scores)
    roc_auc = auc(fpr, tpr)
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Probe (AUC = {roc_auc:.2f})')
    
    # Baseline: Entropy
    fpr_e, tpr_e, _ = roc_curve(y_test, entropy_scores)
    roc_auc_e = auc(fpr_e, tpr_e)
    plt.plot(fpr_e, tpr_e, color='navy', lw=2, linestyle='--', label=f'Entropy (AUC = {roc_auc_e:.2f})')
    
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC - Hallucination Detection')
    plt.legend(loc="lower right")
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(y_test, probe_scores)
    pr_auc = auc(recall, precision)
    
    precision_e, recall_e, _ = precision_recall_curve(y_test, entropy_scores)
    pr_auc_e = auc(recall_e, precision_e)
    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'Probe (AP = {pr_auc:.2f})')
    plt.plot(recall_e, precision_e, color='navy', lw=2, linestyle='--', label=f'Entropy (AP = {pr_auc_e:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve - Hallucination')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/metrics_{model_name.replace('/', '_')}_L{layer}.png")
    plt.close()
    
    # Recall @ Low FPR
    # Interpolate TPR at FPR=0.01 and 0.05
    tpr_at_1 = np.interp(0.01, fpr, tpr)
    tpr_at_5 = np.interp(0.05, fpr, tpr)
    print(f"Probe Recall @ 1% FPR: {tpr_at_1:.2%}")
    print(f"Probe Recall @ 5% FPR: {tpr_at_5:.2%}")
    
    tpr_e_at_1 = np.interp(0.01, fpr_e, tpr_e)
    print(f"Entropy Recall @ 1% FPR: {tpr_e_at_1:.2%}")
    
    return {
        "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
        "precision": precision, "recall": recall, "pr_auc": pr_auc,
        "fpr_e": fpr_e, "tpr_e": tpr_e, "roc_auc_e": roc_auc_e,
        "precision_e": precision_e, "recall_e": recall_e, "pr_auc_e": pr_auc_e
    }

def run_probe_experiment(model_name, label):
    print(f"\n{'='*20}\nProcessing {label}\n{'='*20}")
    model = load_model(model_name)
    if not model: return

    baseline_csv = f"{RESULTS_DIR}/01_baseline_comparison.csv"
    if not os.path.exists(baseline_csv):
        print(f"DEBUG: No baseline data found at {baseline_csv}")
        return
        
    df = pd.read_csv(baseline_csv)
    df = df[df['model'] == label] # Filter by label
    
    if len(df) == 0:
        print(f"No results for {label}")
        return

    # Load categorized data (Phase 2 output)
    # Check if we are running Instruct model logic or Base
    if "Instruct" in label:
        lens_csv = f"{RESULTS_DIR}/02_logit_lens_data_instruct.csv"
    else:
        lens_csv = f"{RESULTS_DIR}/02_logit_lens_data.csv"
        
    if not os.path.exists(lens_csv):
        print(f"No Phase 2 data found at {lens_csv}. Run 02 first.")
        return
        
    df_lens = pd.read_csv(lens_csv)

    # Determine n_samples from lens data
    n_samples_lens = df_lens['prompt_idx'].nunique()
    print(f"Phase 2 used n_samples={n_samples_lens}")
    
    # Reconstruct the EXACT subset used in Phase 2
    correct_df = df[df['is_correct_pairwise'] == True].head(n_samples_lens // 2)
    wrong_df = df[df['is_correct_pairwise'] == False].head(n_samples_lens // 2)
    
    if len(correct_df) + len(wrong_df) < n_samples_lens:
        print(f"Warning: Baseline has fewer samples for {label}. Alignment might fail.")
        
    subset_df = pd.concat([correct_df, wrong_df])
    
    # Validate alignment
    cat_map = df_lens[['prompt_idx', 'category']].drop_duplicates().set_index('prompt_idx')['category']
    
    prompts = subset_df['prompt'].tolist()
    y_labels = []
    
    for i in range(len(prompts)):
        if i not in cat_map.index:
            y_labels.append(0)
            continue
            
        cat = cat_map.loc[i]
        # Target: Overwritten (1) vs Others (0)
        y_labels.append(1 if cat == "Overwritten" else 0)
        
    y = np.array(y_labels)
    target_layer = 20 if "Llama" in label else 25 
    probe_name = f"probe_{label}_L{target_layer}.pkl"
    
    print(f"Extracting activations from Layer {target_layer}...")
    X = extract_activations(model, prompts, target_layer, n_samples=len(prompts))
    
    if len(X) == 0: return

    # Train and Eval
    # Train and Eval
    clf, y_test, y_scores, entropy_scores = train_probe_and_eval(X, y, subset_df)
    metrics_data = plot_metrics(y_test, y_scores, entropy_scores, label, target_layer)
    
    # Stats
    n_test = len(y_test)
    sorted_indices = np.argsort(y_scores)[::-1]
    y_test_sorted = y_test[sorted_indices]
    
    print("\nMonitor Utility (Precision @ Coverage):")
    for coverage in [0.1, 0.25, 0.5]:
        top_k = int(n_test * coverage)
        if top_k == 0: continue
        relevant = y_test_sorted[:top_k]
        precision = np.mean(relevant)
        recall = np.sum(relevant) / np.sum(y_test) if np.sum(y_test) > 0 else 0
        print(f"Coverage {coverage*100}%: Precision {precision:.2%}, Recall {recall:.2%}")

    joblib.dump(clf, f"{RESULTS_DIR}/{probe_name}")
    print(f"DEBUG: Probe {probe_name} saved.")
    
    del model
    torch.cuda.empty_cache()
    
    return metrics_data

def main():
    import pickle
    
    models = {
        "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct", # keep just in case
        "Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
        "Gemma-2-9B": "google/gemma-2-9b"
    }
    
    all_roc_data = {}
    
    # Run specifically for the requested Base models first
    targets = ["Llama-3.1-8B", "Gemma-2-9B"]
    
    for label in targets:
        hf_name = models[label]
        data = run_probe_experiment(hf_name, label)
        if data:
            all_roc_data[label] = data
            
    with open(f"{RESULTS_DIR}/04_roc_data.pkl", "wb") as f:
        pickle.dump(all_roc_data, f)
    print(f"Saved ROC Data to {RESULTS_DIR}/04_roc_data.pkl")
if __name__ == "__main__":
    main()
