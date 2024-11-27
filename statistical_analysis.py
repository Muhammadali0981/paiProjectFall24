import matplotlib.pyplot as plt  # Import the matplotlib library for creating visualizations
import numpy as np  # Import numpy for numerical operations like creating arrays

# Define unique X-axis labels for each of the three subplots
x_labels = [
    ['x1', 'x2', 'x3'],  # X-axis labels for the first subplot
    ['x3', 'x4', 'x8'],  # X-axis labels for the second subplot
    ['y1', 'y2', 'y3']   # X-axis labels for the third subplot
]

# Define the data for three groups (one for each subplot)
data_groups = [
    {
        'Std Deviation': [5, 10, 15],  # Standard deviation values for each category
        'Mean': [20, 30, 40],          # Mean values for each category
        'Median': [25, 35, 45],         # Median values for each category
        'Mode': [25, 35, 45],
        'Var': [25, 35, 45]
    },
    {
        'Std Deviation': [7, 12, 18],  # Second group's standard deviation values
        'Mean': [25, 35, 50],          # Second group's mean values
        'Median': [28, 40, 55],         # Second group's median values
        'Mode': [25, 35, 45],
        'Var': [25, 35, 45]
    },
    {
        'Std Deviation': [4, 8, 13],   # Third group's standard deviation values
        'Mean': [18, 28, 38],          # Third group's mean values
        'Median': [22, 32, 42],         # Third group's median values
        'Mode': [25, 35, 45],
        'Var': [25, 35, 45]
    }
]

# Set up a figure with 3 vertical subplots (stacked on top of each other)
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=False)  # figsize specifies the overall size, sharex=False ensures independent x-axes

# Set bar width for grouped bars
width = 0.15  # Width of each bar in the grouped bar plot

# Loop through each group of data and labels to create subplots
for i, (group_data, labels) in enumerate(zip(data_groups, x_labels)):
    ax = axs[i]  # Access the current subplot (ax is the axis object for subplot i)
    x = np.arange(len(labels))  # Create an array of positions for bars (e.g., [0, 1, 2] for 3 categories)
    
    # Plot each statistic as bars, slightly shifted to group them
    ax.bar(x - 2*width, group_data['Std Deviation'], width, label='Std Deviation', color='skyblue')  # Bar for standard deviation
    ax.bar(x - width, group_data['Mean'], width, label='Mean', color='orange')  # Bar for mean
    ax.bar(x, group_data['Median'], width, label='Median', color='green')  # Bar for median
    ax.bar(x + width, group_data['Mode'], width, label='Mode', color='black')
    ax.bar(x + 2*width, group_data['Var'], width, label='Var', color='red')
    # Add title and labels to the subplot
# Title for the current group
    ax.set_ylabel("Value")  # Y-axis label for the current group
    ax.set_xticks(x)  # Set the x-axis tick positions
    ax.set_xticklabels(labels)  # Set the unique labels for each category on the x-axis
    ax.legend(title="Statistic")  # Add a legend to identify bar colors

# Add a shared x-axis label for all subplots
plt.xlabel("Categories")  # Common x-axis label for the entire figure
plt.tight_layout()  # Adjust subplot spacing to prevent overlap
plt.show()  # Display the final plot
