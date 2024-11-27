import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 


def perform_statistical_analysis(csv_path="ENB2012_data.csv"):
    df = pd.read_csv("ENB2012_data.csv")

    x_labels = [
        ['X1', 'X7', 'X8'],  
        ['X2', 'X3', 'X4'],  
        ['X5', 'X6', 'Y1','Y2']   
    ]

    statistical_results = {
            'groups': []
        }

    data_groups = [
        {
            'Std Deviation': [df['X1'].std(), df['X7'].std(), df['X8'].std()], 
            'Mean': [df['X1'].mean(), df['X7'].mean(), df['X8'].mean()],          
            'Median': [df['X1'].median(), df['X7'].median(), df['X8'].median()],         
            'Mode': [df['X1'].mode().iloc[0] if not df['X1'].mode().empty else np.nan,
                    df['X7'].mode().iloc[0] if not df['X7'].mode().empty else np.nan,
                    df['X8'].mode().iloc[0] if not df['X8'].mode().empty else np.nan],
            'Var': [df['X1'].var()/10, df['X7'].var()/10, df['X8'].var()/10]
        },
        {
            'Std Deviation': [df['X2'].std(), df['X3'].std(), df['X4'].std()], 
            'Mean': [df['X2'].mean(), df['X3'].mean(), df['X4'].mean()],          
            'Median': [df['X2'].median(), df['X3'].median(), df['X4'].median()],         
            'Mode': [df['X2'].mode().iloc[0] if not df['X2'].mode().empty else np.nan,
                    df['X3'].mode().iloc[0] if not df['X3'].mode().empty else np.nan,
                    df['X4'].mode().iloc[0] if not df['X4'].mode().empty else np.nan],
            'Var': [df['X2'].var()/10, df['X3'].var()/10, df['X4'].var()/10]
        },
        {
            'Std Deviation': [df['X5'].std(), df['X6'].std(), df['Y1'].std(), df['Y2'].std()], 
            'Mean': [df['X5'].mean(), df['X6'].mean(), df['Y1'].mean(), df['Y2'].mean()],          
            'Median': [df['X5'].median(), df['X6'].median(), df['Y1'].median(), df['Y2'].median()],         
            'Mode': [df['X5'].mode().iloc[0] if not df['X5'].mode().empty else np.nan,
                    df['X6'].mode().iloc[0] if not df['X6'].mode().empty else np.nan,
                    df['Y1'].mode().iloc[0] if not df['Y1'].mode().empty else np.nan,
                    df['Y2'].mode().iloc[0] if not df['Y2'].mode().empty else np.nan],
            'Var': [df['X5'].var()/10, df['X6'].var()/10, df['Y1'].var()/10, df['Y2'].var()/10]
        }
    ]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=False) 


    width = 0.15  


    for i, (group_data, labels) in enumerate(zip(data_groups, x_labels)):
        ax = axs[i]  
        x = np.arange(len(labels))  
        
        # Plot each statistic as bars, slightly shifted to group them
        ax.bar(x - 2*width, group_data['Std Deviation'], width, label='Std Deviation', color='#8e44ad')  # Bar for standard deviation
        ax.bar(x - width, group_data['Mean'], width, label='Mean', color='#d35400')  # Bar for mean
        ax.bar(x, group_data['Median'], width, label='Median', color='#f1c40f')  # Bar for median
        ax.bar(x + width, group_data['Mode'], width, label='Mode', color='#16a085') # Bar for mode
        ax.bar(x + 2*width, group_data['Var'], width, label='Var', color='#e59866') # Bar for variation 
        # Add title and labels to the subplot

        ax.set_ylabel("Value")  # Y-axis label for the current group
        ax.set_xticks(x)  # Set the x-axis tick positions
        ax.set_xticklabels(labels)  # Set the unique labels for each category on the x-axis
        ax.legend(title="Statistic")  # Add a legend to identify bar colors


    for i, group in enumerate(data_groups, start=1):
        print(f"\nData Group {i}:")
        for key, values in group.items():
            print(f"{key}: {values}")


    # Add a shared x-axis label for all subplots
    plt.xlabel("Categories") 
    plt.tight_layout()  # Adjust subplot spacing to prevent overlap
    

    statistical_plot_path = 'static/statistical_analysis_plot.png'
    plt.savefig(statistical_plot_path)
    plt.close()

    # Add plot path to results
    statistical_results['plot'] = statistical_plot_path

    return statistical_results

if __name__ == "__main__":
    results = perform_statistical_analysis()
    print("Statistical Analysis Results:", results)