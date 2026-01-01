
import torch
import pandas as pd
import numpy as np
from transformer_lens import HookedTransformer
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import gc

# Configuration
MODELS = {
    "Llama-3.1-8B": "meta-llama/Meta-Llama-3.1-8B",
    "Gemma-2-9B": "google/gemma-2-9b" 
}
DATASET_NAME = "NeelNanda/counterfact-tracing"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_model(model_name_hf):
    print(f"Loading model: {model_name_hf}")
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

def get_counterfact_data():
    print("Loading dataset...")
    ds = load_dataset(DATASET_NAME, split="train")
    return ds

def evaluate_baseline(model, dataset, model_label, n_samples=1000):
    results = []
    
    print(f"Running evaluation for {model_label} on {n_samples} samples...")
    for i, item in tqdm(enumerate(dataset), total=min(len(dataset), n_samples)):
        if i >= n_samples:
            break
            
        prompt = item["prompt"]
        target_true = item["target_true"]
        target_false = item["target_false"]
        
        try:
            logits = model(prompt, return_type="logits")
            last_token_logits = logits[0, -1, :]
            
            target_true_id = model.to_single_token(target_true)
            target_false_id = model.to_single_token(target_false)
            
            if isinstance(target_true_id, list) or isinstance(target_false_id, list):
                 continue

            true_logit = last_token_logits[target_true_id].item()
            false_logit = last_token_logits[target_false_id].item()
            logit_diff = true_logit - false_logit
            
            # Pairwise Accuracy (Knowledge metric)
            is_pairwise_correct = logit_diff > 0
            
            probs = torch.softmax(last_token_logits, dim=-1)
            true_prob = probs[target_true_id].item()
            false_prob = probs[target_false_id].item()
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
            
            # Top-1 Exact Match (Strict metric)
            pred_id = torch.argmax(last_token_logits).item()
            pred_str = model.to_string(pred_id)
            is_top1_correct = (pred_id == target_true_id)
            
            results.append({
                "model": model_label,
                "index": i,
                "prompt": prompt,
                "target_true": target_true,
                "target_false": target_false,
                "true_prob": true_prob,
                "false_prob": false_prob,
                "prediction": pred_str,
                "is_correct_top1": is_top1_correct,
                "is_correct_pairwise": is_pairwise_correct,
                "entropy": entropy,
                "logit_diff": logit_diff
            })
            
        except Exception as e:
            # print(f"Error on item {i}: {e}")
            continue

    return pd.DataFrame(results)

def main():
    ds = get_counterfact_data()
    all_results = []
    
    # Check for existing results to append/merge
    output_path = f"{RESULTS_DIR}/01_baseline_comparison.csv"
    if os.path.exists(output_path):
        print(f"Found existing results at {output_path}. Loading...")
        try:
            existing_df = pd.read_csv(output_path)
            all_results.append(existing_df)
            print(f"Loaded {len(existing_df)} rows.")
        except:
            print("Failed to read existing results file.")

    # Select models to run
    # Only run Llama this time
    # Select models to run
    models_to_run = {
        "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct"
    }
    
    for label, hf_name in models_to_run.items():
        print(f"\n{'='*20}\nProcessing {label}\n{'='*20}")
        
        # Scout run size
        n_samples_run = 200
        
        # Load Model
        model = load_model(hf_name)
        if model is None:
            continue
            
        # Run Eval
        df_model = evaluate_baseline(model, ds, label, n_samples=n_samples_run)
        all_results.append(df_model)
        
        # Cleanup to save memory for next model
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        # Deduplicate just in case
        # final_df = final_df.drop_duplicates(subset=['model', 'index']) 
        final_df.to_csv(output_path, index=False)
        
        print(f"\nSaved combined results to {output_path}")
        print("\nSummary Results:")
        print(final_df.groupby("model")[["is_correct", "entropy"]].mean())
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
