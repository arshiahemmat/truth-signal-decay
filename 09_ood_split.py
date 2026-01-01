import pandas as pd
import numpy as np
import joblib
import os
from transformer_lens import HookedTransformer
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
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

def extract_activations(model, prompts, target_layer):
    activations = []
    print(f"Extracting activations for {len(prompts)} samples...")
    for prompt in tqdm(prompts):
        try:
            with torch.no_grad():
                hook_name = f"blocks.{target_layer}.hook_resid_post"
                _, cache = model.run_with_cache(prompt, names_filter=lambda x: x == hook_name)
                act = cache[hook_name][0, -1, :].float().cpu().numpy()
                activations.append(act)
        except:
            activations.append(None)
    return np.array(activations)

def run_ood_split(model, df, ds, label, layer):
    # 1. Map Index -> Relation ID
    rel_ids = []
    for idx in df['index']:
        try:
            # Assuming ds handles indexing
            item = ds[int(idx)]
            rel_ids.append(item['relation_id'])
        except:
            rel_ids.append("unknown")
            
    df['relation_id'] = rel_ids
    
    # 2. Load Labels (Overwritten) from 02
    lens_csv = f"{RESULTS_DIR}/02_logit_lens_data.csv"
    if not os.path.exists(lens_csv):
        print("02 lens data not found")
        return
    df_lens = pd.read_csv(lens_csv)
    
    # Map prompt_idx in lens to df index?
    # prompt_idx in lens likely corresponds to row index in filtered df?
    # 02 logic: n_samples=200 filtered from big df.
    # We must RE-RUN extraction on the subset used in 02, OR
    # Just run 09 on the SAME subset as 02.
    # 02 used "get_data_subset".
    # We can assume df passed in is the subset?
    # Actually, 04 probe training used `subset_df` derived from `01`.
    # Let's align carefully.
    
    # Reconstruct 02 subset logic
    n_samples_lens = df_lens['prompt_idx'].nunique()
    correct_df = df[df['is_correct_pairwise'] == True].head(n_samples_lens // 2)
    wrong_df = df[df['is_correct_pairwise'] == False].head(n_samples_lens // 2)
    subset_df = pd.concat([correct_df, wrong_df])
    
    # Validate alignment
    cat_map = df_lens[['prompt_idx', 'category']].drop_duplicates().set_index('prompt_idx')['category']
    prompts = subset_df['prompt'].tolist()
    y_labels = []
    
    for i in range(len(prompts)):
        if i not in cat_map.index:
            y_labels.append(0) 
        else:
            cat = cat_map.loc[i]
            y_labels.append(1 if cat == "Overwritten" else 0)
            
    y = np.array(y_labels)
    X = extract_activations(model, prompts, layer)
    
    # Filter valid X
    valid_mask = np.array([a is not None for a in X])
    X = np.array([x for x in X if x is not None])
    y = y[valid_mask]
    
    # 3. Split by Relation
    subset_rels = subset_df['relation_id'].values[valid_mask]
    unique_rels = np.unique(subset_rels)
    
    np.random.seed(42)
    train_rels = np.random.choice(unique_rels, size=int(len(unique_rels)*0.5), replace=False)
    
    train_mask = np.isin(subset_rels, train_rels)
    test_mask = ~train_mask
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"OOD Split: {len(X_train)} Train, {len(X_test)} Test")
    
    # 4. Train Probe
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    # 5. Eval
    probs = clf.predict_proba(X_test)[:, 1]
    
    # Metrics
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall, precision)
    
    # Precision @ Coverage
    sorted_indices = np.argsort(probs)[::-1]
    y_sorted = y_test[sorted_indices]
    
    print(f"\nOOD (Relation Split) Results for {label}:")
    print(f"AUC: {roc_auc:.3f}")
    print(f"PR-AUC: {pr_auc:.3f}")
    
    for cov in [0.1, 0.25, 0.5]:
        k = int(len(y_test) * cov)
        if k > 0:
            p = np.mean(y_sorted[:k])
            print(f"Precision @ {cov*100}%: {p:.2%}")
            
    # Plot
    plt.figure()
    plt.plot(fpr, tpr, label=f"OOD Probe (AUC={roc_auc:.2f})")
    plt.plot([0,1], [0,1], 'k--')
    plt.title(f"OOD Robustness (Relation Split) - {label}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.savefig(f"{PLOTS_DIR}/09_ood_roc_{label}.png")

def main():
    # Load Dataset
    print("Loading Dataset...")
    ds = load_dataset("NeelNanda/counterfact-tracing", split="train")
    
    # Load Baseline Data
    df_all = pd.read_csv(f"{RESULTS_DIR}/01_baseline_comparison.csv")
    
    # Run only for Base Llama (Highest Priority)
    label = "Llama-3.1-8B"
    hf_name = "meta-llama/Llama-3.1-8B"
    layer = 20
    
    model = load_model(hf_name)
    if model:
        df_model = df_all[df_all['model'] == label]
        run_ood_split(model, df_model, ds, label, layer)

if __name__ == "__main__":
    main()
