import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(filepath, output_dir):
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {filepath}")
        return

    # Check initial shape
    print(f"Dataset Shape: {df.shape}")

    # Create target variable (1 if rain > 0, else 0)
    # The column name in the dataset has a trailing space: 'rain '
    df['rain_class'] = (df['rain '] > 0.0).astype(int)

    # Clean column names by removing leading/trailing spaces for easier access
    df.columns = [col.strip() for col in df.columns]

    # Select only numeric features for correlation
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Exclude target-derived variables to avoid data leakage during feature analysis
    # 'precipitation', 'rain', 'snowfall' directly leak whether it's raining or not
    cols_to_exclude = ['rain', 'precipitation', 'snowfall']
    df_features = df_numeric.drop(columns=[col for col in cols_to_exclude if col in df_numeric.columns])

    # 1. Correlation Matrix Heatmap
    print("\n[1/3] Generating Correlation Heatmap...")
    plt.figure(figsize=(14, 10))
    corr = df_features.corr()
    
    # Mask the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Using seaborn heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Meteorological Features', fontsize=16)
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Saved: {heatmap_path}")

    # 2. Correlation with Target Variable
    print("\n[2/3] Generating Feature Correlation with Target Variable...")
    plt.figure(figsize=(10, 6))
    
    # Extract correlation of all features with 'rain_class' and sort them
    target_corr = corr['rain_class'].drop('rain_class').sort_values(ascending=False)
    
    sns.barplot(x=target_corr.values, y=target_corr.index, hue=target_corr.index, palette='viridis', legend=False)
    plt.title('Feature Correlation with Rain (rain_class)', fontsize=14)
    plt.xlabel('Spearman/Pearson Correlation Coefficient')
    plt.tight_layout()
    target_corr_path = os.path.join(output_dir, 'target_correlation.png')
    plt.savefig(target_corr_path)
    plt.close()
    print(f"Saved: {target_corr_path}")

    # 3. Distribution of Top Features by Rain Class
    print("\n[3/3] Generating Distributions for Top Features...")
    # Select the top 4 absolute most correlated features
    top_features = target_corr.abs().sort_values(ascending=False).head(4).index.tolist()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(top_features):
        sns.kdeplot(data=df, x=feature, hue='rain_class', fill=True, ax=axes[i], common_norm=False, palette='Set1', alpha=0.5)
        axes[i].set_title(f'Distribution of {feature} by Rain Class')
    
    plt.tight_layout()
    dists_path = os.path.join(output_dir, 'top_features_distribution.png')
    plt.savefig(dists_path)
    plt.close()
    print(f"Saved: {dists_path}")

    print("\n" + "="*50)
    print("EDA COMPLETED!")
    print("="*50)
    


if __name__ == "__main__":
    # Setup Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to Dataset
    # Level_1_Rain_Classification/eda/script.py -> up 2 levels
    dataset_path = os.path.abspath(os.path.join(script_dir, "..", "..", "data/meteorology_dataset.csv"))
    
    # Ensure EDA folder exists
    os.makedirs(script_dir, exist_ok=True)
    
    print(f"Starting Exploratory Data Analysis...")
    run_eda(dataset_path, script_dir)
