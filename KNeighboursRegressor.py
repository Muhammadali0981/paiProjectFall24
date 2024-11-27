import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("ENB2012_data.csv")

# Drop rows with missing values
df.dropna(inplace=True)

#  Predicting Y1
X = df.drop('Y1', axis=1)  # Features: All columns except 'Y1' (the target)
y = df['Y1']  # Target: The column we want to predict

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Test different values of k (1 to 250)
k_values = range(1, 251)
mse_values = []
r2_values = []
# MSE stands for Mean Squared Error and is used in machine learning models to gauge the accuracy of its predictions
# r2 (R-Squared) is a measure that tells you how well your model's predictions match the true values. 
# It ranges from 0 to 1, where 1 means the model is perfect and 0 means it's no better than a random guess.

for k in k_values:
    # Initialize KNN regressor with the current k value
    knn = KNeighborsRegressor(n_neighbors=k)
    
    # Train the model on the training data
    knn.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = knn.predict(X_test)
    
    # Calculate MSE and R^2 for the current k value
    mse_values.append(mean_squared_error(y_test, y_pred))
    r2_values.append(r2_score(y_test, y_pred))

# Select best k based on MSE (lowest is best)
best_k_y1 = mse_values.index(min(mse_values)) + 1

# Train and predict using the best k for Y1
knn_best_y1 = KNeighborsRegressor(n_neighbors=best_k_y1)
knn_best_y1.fit(X_train, y_train)
y_pred_best_y1 = knn_best_y1.predict(X_test)

# Predicting Y2
X = df.drop('Y2', axis=1)  # Features: All columns except 'Y2' (the target)
y = df['Y2']  # Target: The column we want to predict

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Test different values of k (1 to 250)
mse_values = []
r2_values = []

for k in k_values:
    # Initialize KNN regressor with the current k value
    knn = KNeighborsRegressor(n_neighbors=k)
    
    # Train the model on the training data
    knn.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = knn.predict(X_test)
    
    # Calculate MSE and R^2 for the current k value
    mse_values.append(mean_squared_error(y_test, y_pred))
    r2_values.append(r2_score(y_test, y_pred))

# Select best k based on MSE (lowest is best)
best_k_y2 = mse_values.index(min(mse_values)) + 1

# Train and predict using the best k for Y2
knn_best_y2 = KNeighborsRegressor(n_neighbors=best_k_y2)
knn_best_y2.fit(X_train, y_train)
y_pred_best_y2 = knn_best_y2.predict(X_test)

# Create subplots for Y1 and Y2 comparisons
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Y1 comparison
sns.regplot(x=y_test, y=y_pred_best_y1, ax=axes[0], scatter_kws={'s': 50}, line_kws={'color': 'red'}, ci=None, color='cyan')
axes[0].set_xlabel('Actual Y1')
axes[0].set_ylabel('Predicted Y1')
axes[0].set_title(f'Regression Plot: Actual vs Predicted (Y1), Best K: {best_k_y1}')

# Y2 comparison
sns.regplot(x=y_test, y=y_pred_best_y2, ax=axes[1], scatter_kws={'s': 50}, line_kws={'color': 'red'}, ci=None, color='magenta')
axes[1].set_xlabel('Actual Y2')
axes[1].set_ylabel('Predicted Y2')
axes[1].set_title(f'Regression Plot: Actual vs Predicted (Y2), Best K: {best_k_y2}')

# Display the plots
plt.tight_layout()
plt.show()
