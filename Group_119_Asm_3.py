# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("po2_data.csv")

# Data Exploration
# Display basic statistics of the dataset
print(data.describe())

# Data Visualization
# Correlation Heatmap
correlation_matrix = data.corr()  # Calculate correlation matrix
plt.figure(figsize=(10, 8))  # Set figure size
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)  # Create a heatmap
plt.title("Correlation Heatmap")  # Set title
plt.show()  # Display the heatmap

# Pairplot for relationships between numerical variables
sns.pairplot(data, vars=["age", "motor_updrs", "total_updrs"])  # Create pairplots
plt.show()  # Display the pairplots

# Distribution of Age and Motor UPDRS Score
plt.figure(figsize=(12, 5))  # Set figure size
plt.subplot(1, 2, 1)  # Create subplot
sns.histplot(data["age"], bins=20, kde=True)  # Create a histogram for age
plt.xlabel("age")  # Set x-label
plt.ylabel("Frequency")  # Set y-label
plt.title("Distribution of Age")  # Set title

plt.subplot(1, 2, 2)  # Create subplot
sns.histplot(data["motor_updrs"], bins=20, kde=True)  # Create a histogram for Motor UPDRS scores
plt.xlabel("Motor UPDRS Score")  # Set x-label
plt.ylabel("Frequency")  # Set y-label
plt.title("Distribution of Motor UPDRS Scores")  # Set title
plt.show()  # Display the histograms

# Boxplot to visualize Motor UPDRS scores by gender
plt.figure(figsize=(8, 5))  # Set figure size
sns.boxplot(x="sex", y="motor_updrs", data=data)  # Create a boxplot
plt.xlabel("Gender")  # Set x-label
plt.ylabel("Motor UPDRS Score")  # Set y-label
plt.title("Motor UPDRS Scores by Gender")  # Set title
plt.xticks([0, 1], ["Male", "Female"])  # Set x-axis labels
plt.show()  # Display the boxplot

# Split the dataset into features (X) and target variables (motor and total UPDRS scores)
X = data.drop(columns=["motor_updrs", "total_updrs"])  # Features
y_motor = data["motor_updrs"]  # Target variable (Motor UPDRS)
y_total = data["total_updrs"]  # Target variable (Total UPDRS)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_motor_train, y_motor_test, y_total_train, y_total_test = train_test_split(
    X, y_motor, y_total, test_size=0.2, random_state=42
)

# Initialize and train linear regression models
motor_model = LinearRegression()  # Create a linear regression model for Motor UPDRS
motor_model.fit(X_train, y_motor_train)  # Train the model

total_model = LinearRegression()  # Create a linear regression model for Total UPDRS
total_model.fit(X_train, y_total_train)  # Train the model

# Make predictions
motor_predictions = motor_model.predict(X_test)  # Predict Motor UPDRS scores
total_predictions = total_model.predict(X_test)  # Predict Total UPDRS scores

# Evaluate model performance
motor_mae = mean_absolute_error(y_motor_test, motor_predictions)  # Calculate MAE for Motor UPDRS
motor_mse = mean_squared_error(y_motor_test, motor_predictions)  # Calculate MSE for Motor UPDRS
motor_r2 = r2_score(y_motor_test, motor_predictions)  # Calculate R-squared for Motor UPDRS

total_mae = mean_absolute_error(y_total_test, total_predictions)  # Calculate MAE for Total UPDRS
total_mse = mean_squared_error(y_total_test, total_predictions)  # Calculate MSE for Total UPDRS
total_r2 = r2_score(y_total_test, total_predictions)  # Calculate R-squared for Total UPDRS

# Print model performance metrics
print("Motor UPDRS Predictions:")
print(f"Mean Absolute Error (MAE): {motor_mae:.2f}")
print(f"Mean Squared Error (MSE): {motor_mse:.2f}")
print(f"R-squared (R2): {motor_r2:.2f}\n")

print("Total UPDRS Predictions:")
print(f"Mean Absolute Error (MAE): {total_mae:.2f}")
print(f"Mean Squared Error (MSE): {total_mse:.2f}")
print(f"R-squared (R2): {total_r2:.2f}")
