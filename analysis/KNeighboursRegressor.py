import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

def perform_knn_regression(csv_path="ENB2012_data.csv"):
    # Load the dataset
    df = pd.read_csv(csv_path)

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Results dictionary to store regression details
    regression_results = {
        'Y1': {},
        'Y2': {}
    }

    # Predicting Y1
    X = df.drop('Y1', axis=1)  # Features: All columns except 'Y1'
    y = df['Y1']  # Target: Y1

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Test different k values
    k_values = range(1, 251)
    mse_values = []
    r2_values = []

    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        mse_values.append(mean_squared_error(y_test, y_pred))
        r2_values.append(r2_score(y_test, y_pred))

    # Select best k for Y1
    best_k_y1 = mse_values.index(min(mse_values)) + 1
    
    # Store Y1 regression details
    regression_results['Y1']['best_k'] = best_k_y1
    regression_results['Y1']['mse'] = min(mse_values)
    regression_results['Y1']['r2'] = r2_values[best_k_y1 - 1]

    # Predicting Y2 (similar process)
    X = df.drop('Y2', axis=1)  # Features: All columns except 'Y2'
    y = df['Y2']  # Target: Y2

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reset lists for Y2
    mse_values = []
    r2_values = []

    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        mse_values.append(mean_squared_error(y_test, y_pred))
        r2_values.append(r2_score(y_test, y_pred))

    # Select best k for Y2
    best_k_y2 = mse_values.index(min(mse_values)) + 1
    
    # Store Y2 regression details
    regression_results['Y2']['best_k'] = best_k_y2
    regression_results['Y2']['mse'] = min(mse_values)
    regression_results['Y2']['r2'] = r2_values[best_k_y2 - 1]

    # Create regression comparison plot
    plt.figure(figsize=(16, 6))
    
    # Y1 Regression Plot
    plt.subplot(1, 2, 1)
    knn_best_y1 = KNeighborsRegressor(n_neighbors=best_k_y1)
    knn_best_y1.fit(X_train, y_train)
    y_pred_best_y1 = knn_best_y1.predict(X_test)
    
    sns.regplot(x=y_test, y=y_pred_best_y1, 
                line_kws={'color':'red'},
                    scatter_kws={'color':'darkblue', 'alpha':0.2, 'edgecolor':'black'})
    plt.xlabel('Actual Y1')
    plt.ylabel('Predicted Y1')
    plt.title(f'Y1 Regression (Best K: {best_k_y1})')

    # Y2 Regression Plot
    plt.subplot(1, 2, 2)
    knn_best_y2 = KNeighborsRegressor(n_neighbors=best_k_y2)
    knn_best_y2.fit(X_train, y_train)
    y_pred_best_y2 = knn_best_y2.predict(X_test)
    
    sns.regplot(x=y_test, y=y_pred_best_y2, 
                line_kws={'color':'red'},
                    scatter_kws={'color':'darkblue', 'alpha':0.2, 'edgecolor':'black'})
    plt.xlabel('Actual Y2')
    plt.ylabel('Predicted Y2')
    plt.title(f'Y2 Regression (Best K: {best_k_y2})')

    plt.tight_layout()
    
    # Save the plot
    regression_plot_path = 'static/regression_comparison_plot.png'
    plt.savefig(regression_plot_path)
    plt.close()

    # Add plot path to results
    regression_results['regression_plot'] = regression_plot_path

    return regression_results

# If script is run directly, perform regression and print results
if __name__ == "__main__":
    results = perform_knn_regression()
    print("Regression Results:", results)