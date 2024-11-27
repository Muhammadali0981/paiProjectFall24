import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

def generate_graphs(csv_path="ENB2012_data.csv"):
    # Graphs dictionary to store different types of graphs
    graphs = {}

    # Load the dataset
    df = pd.read_csv(csv_path)

    # Convert columns to numeric, replacing invalid values
    cols = df.columns
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df.fillna(-1, inplace=True)
    df = df.replace(-1, np.nan)

    # Box Plots
    plt.figure(figsize=(16,12))
    for i, col in enumerate(['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'Y1', 'Y2'], 1):
        plt.subplot(5, 2, i)
        sns.boxplot(data=df, x=col)
    plt.subplots_adjust(hspace=1, wspace=1)
    
    # Save box plot
    box_plot_path = 'static/box_plot.png'
    plt.savefig(box_plot_path)
    graphs['box_plot'] = box_plot_path
    plt.close()

    # Distribution Plots
    distribution_plots = []
    for col in df.columns:
        plt.figure()
        sns.displot(df[col], kde=True, color='purple')
        
        
        # Save each distribution plot
        dist_plot_path = f'static/dist_plot_{col}.png'
        plt.savefig(dist_plot_path)
        distribution_plots.append(dist_plot_path)
        plt.close()
    graphs['distribution_plots'] = distribution_plots

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, annot_kws={"size": 8})
    sns.set_palette('pastel')
    
    # Save correlation heatmap
    correlation_plot_path = 'static/correlation_plot.png'
    plt.savefig(correlation_plot_path)
    graphs['correlation_plot'] = correlation_plot_path
    plt.close()

    # Regression Plots
    plt.figure(figsize=(16,12))
    plot_pairs = [
        ("X1", "X2"),
        ("X4", "X5"),
        ("X4", "X2"),
        ("Y1", "Y2"),
        ("Y1", "X5"),
        ("Y2", "X5")
    ]
    
    for i, (x, y) in enumerate(plot_pairs, 1):
        sns.set_theme(style='darkgrid')
        plt.subplot(3, 2, i)
        sns.regplot(x=df[x], y=df[y], 
                    line_kws={'color':'red'},
                    scatter_kws={'color':'darkblue', 'alpha':0.2, 'edgecolor':'black'})
        
    
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    
    # Save regression plot
    regression_plot_path = 'static/regression_plot.png'
    plt.savefig(regression_plot_path)
    graphs['regression_plot'] = regression_plot_path
    plt.close()

    return graphs

# If script is run directly, generate and show graphs
if __name__ == "__main__":
    graphs = generate_graphs()
    print("Graphs generated:", graphs)