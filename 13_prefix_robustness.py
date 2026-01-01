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
    # Load 11 CSV
    try:
        df = pd.read_csv(f"{RESULTS_DIR}/11_rank_mass_Llama-3.1-8B.csv")
    except:
        print("Could not load 11_rank_mass CSV. Please run 11 first.")
        return None
        
    # Load Dataset for Targets
    ds = load_dataset("NeelNanda/counterfact-tracing", split="train")
    
    # Init target cols
    df['target_true'] = str(np.nan)
    
    # Create dictionary mapping
    prompt_map = {}
    print("Building Prompt Map...")
    for item in ds:
        prompt_map[item['prompt']] = item['target_true']
        
    print("Mapping Targets...")
    t_trues = []
    for prompt in df['prompt']:
        t_trues.append(prompt_map.get(prompt, None))
            
    df['target_true'] = t_trues
    df = df.dropna(subset=['target_true'])
    return df

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
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Select categories to test
    # We want to see if "True-Competitive" and "Generic-Collapse" recover.
    categories = ["True-Competitive", "Generic-Collapse", "False-Selected"]
    
    results = []
    
    print("Running Generation Tests...")
    
    for cat in categories:
        subset = df[df['category'] == cat]
        if len(subset) == 0: continue
        
        # Sample N=50 to save time
        sample_size = min(50, len(subset))
        subset = subset.sample(sample_size, random_state=42)
        
        print(f"Testing {len(subset)} samples for category: {cat}")
        
        for i, row in tqdm(subset.iterrows(), total=len(subset)):
            prompt = row['prompt']
            target_true = row['target_true'] # e.g. " Paris"
            
            # Generate
            # max_new_tokens=5
            with torch.no_grad():
                # generate returns tensor of shape [batch, pos]
                output = model.generate(
                    prompt, 
                    max_new_tokens=5, 
                    temperature=0.0,
                    verbose=False
                )
                
            # Check output type
            if isinstance(output, str):
                 generated_text = output
            elif isinstance(output, list) and isinstance(output[0], str):
                 generated_text = output[0]
            else:
                 # Assume tensor
                 generated_text = model.to_string(output[0])
            # We want just the new text
            new_text = generated_text[len(prompt):]
            
            # Check Content Recovery
            # 1. Exact@1 check (should be false for these categories)
            # Actually, we can re-verify.
            # But let's check if target is in new_text.
            
            t_clean = target_true.strip()
            # Split new_text into words/tokens loosely
            # " the Paris" -> ["the", "Paris"]
            # " Paris" -> ["Paris"]
            
            # Simple check: is t_clean in new_text?
            # Issue: "Par" matches "Paris".
            # Better: Check token presence? 
            # Or token decoding.
            # Let's do string search for now, it's robust enough because targets are specific entities.
            
            recovered = t_clean in new_text
            
            results.append({
                "category": cat,
                "prompt": prompt,
                "target_true": t_clean,
                "generated": new_text,
                "recovered": recovered,
                "exact_at_1": False # By definition of the category
            })
            
    # Add stats to DF
    res_df = pd.DataFrame(results)
    
    # Calculate Recovery Rates
    stats = res_df.groupby('category')['recovered'].mean().reset_index()
    stats['exact_at_1'] = 0.0 # Baseline for these
    
    print("\nRecovery Rates (Content@5) by Category:")
    print(stats)
    
    # Plot
    # We want grouped bar: Exact@1 (0) vs Content@5 (Value)
    # Melt or adjust data structure
    plot_data = []
    for i, row in stats.iterrows():
        plot_data.append({"category": row['category'], "metric": "Exact@1", "score": 0.0})
        plot_data.append({"category": row['category'], "metric": "Content@5", "score": row['recovered']})
        
    plot_df = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(data=plot_df, x='category', y='score', hue='metric', palette="magma")
    plt.title("Prefix-Robustness: Do Errors Recover?")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.savefig(f"{PLOTS_DIR}/13_prefix_robustness.png")
    
    # Save examples of recovery
    recovered_examples = res_df[res_df['recovered'] == True].sample(min(5, len(res_df[res_df['recovered'] == True])))
    print("\nExamples of Recovery:")
    for i, row in recovered_examples.iterrows():
        print(f"Cat: {row['category']} | Prompt: ...{row['prompt'][-20:]}")
        print(f"Target: {row['target_true']} | Gen: {row['generated']}")
        print("-" * 30)

if __name__ == "__main__":
    main()
