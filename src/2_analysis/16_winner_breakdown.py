import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "/homes/55/arshia/project_1_hallucination/results"
PLOTS_DIR = "/homes/55/arshia/project_1_hallucination/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def main():
    # Load data
    try:
        df = pd.read_csv(f"{RESULTS_DIR}/11_rank_mass_Llama-3.1-8B.csv")
    except:
        return

    # Filter for Errors relative to Top-1 (i.e. rank_true > 0)
    # The taxonomy "True-Competitive" and "Generic-Collapse" are the errors.
    errors = df[df['category'].isin(["True-Competitive", "Generic-Collapse", "False-Selected", "Other"])]
    
    # Analyze 'top1_final' strings
    # Clean up strings (strip spaces)
    errors['winner'] = errors['top1_final'].astype(str).str.strip()
    
    # Categorize Winners
    stopwords = ["the", "a", "of", "in", "to", "is", "was", "The", "A", "an"]
    
    def classify_winner(row):
        w = row['winner']
        if w in stopwords:
            return "Stopword (the/a/of)"
        # Check if it matches False Target?
        # We don't have target_false here easily unless we map it. 
        # But 'False-Selected' category implies it matched false target.
        if row['category'] == "False-Selected":
            return "False Target"
        return "Other Entity/Word"

    errors['winner_type'] = errors.apply(classify_winner, axis=1)
    
    # Count
    counts = errors['winner_type'].value_counts(normalize=True).reset_index()
    counts.columns = ['Winner Type', 'Proportion']
    
    print("Winner Token Breakdown:")
    print(counts)
    
    # Plot
    plt.figure(figsize=(6, 5))
    ax = sns.barplot(data=counts, x='Winner Type', y='Proportion', palette="rocket")
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
        
    plt.title("What Beats the Truth?\n(Distribution of Top-1 Tokens in Error Cases)")
    plt.ylabel("Proportion of Errors")
    plt.ylim(0, 1.0)
    plt.savefig(f"{PLOTS_DIR}/16_winner_breakdown.png")

if __name__ == "__main__":
    main()
