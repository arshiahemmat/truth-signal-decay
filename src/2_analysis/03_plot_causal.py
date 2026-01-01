
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# Configuration
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def plot_causal_trace():
    # Load all attr_true_*.npy
    files = glob.glob(f"{RESULTS_DIR}/attr_true_*.npy")
    if not files:
        print("No causal trace files found.")
        return
        
    print(f"Aggregating {len(files)} traces...")
    
    avg_attr_true = None
    avg_attr_false = None
    
    count = 0
    for f in files:
        idx = f.split("_")[-1].replace(".npy", "")
        f_false = f"{RESULTS_DIR}/attr_false_{idx}.npy"
        
        if os.path.exists(f_false):
            at = np.load(f)
            af = np.load(f_false)
            
            # Shape: [n_layers, n_heads]
            if avg_attr_true is None:
                avg_attr_true = np.zeros_like(at)
                avg_attr_false = np.zeros_like(af)
            
            avg_attr_true += at
            avg_attr_false += af
            count += 1
            
    if count > 0:
        avg_attr_true /= count
        avg_attr_false /= count
        
        # Plot Heatmap
        # Difference: Attribution to True vs False? 
        # Or just "Net Contribution to Correctness" = Attr(True) - Attr(False)
        diff = avg_attr_true - avg_attr_false
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(diff, center=0, cmap="RdBu_r") # Red = Positive, Blue = Negative
        plt.title("Net Logit Attribution (True - False Token)\n(Red = Supports Correct Answer, Blue = Suppresses Correct Answer)")
        plt.xlabel("Head Index")
        plt.ylabel("Layer Index")
        plt.savefig(f"{PLOTS_DIR}/03_causal_heatmap.png")
        plt.close()
        
        print(f"Heatmap saved to {PLOTS_DIR}/03_causal_heatmap.png")

if __name__ == "__main__":
    plot_causal_trace()
