import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# File paths of datasets
file_paths = [
    "biflow_mqtt_bruteforce.csv",
    "biflow_normal.csv",
    "biflow_scan_A.csv",
    "biflow_scan_sU.csv",
    "biflow_sparta.csv"
]

# Load all datasets
dataframes = [pd.read_csv(file) for file in file_paths]

# Get unique features across all datasets
all_columns = set()
for df in dataframes:
    all_columns.update(df.columns)

total_unique_features = len(all_columns)
print(f"Total number of unique features across all datasets: {total_unique_features}")

# Merge datasets
df_combined = pd.concat(dataframes, ignore_index=True)

# Remove non-numeric columns (keeping only numerical features)
df_combined = df_combined.select_dtypes(include=['number'])

# Total number of features in the merged dataset
total_features = df_combined.shape[1] - 1  # Excluding target column
print(f"Total number of features in the merged dataset: {total_features}")

# Separate features and target variable
X = df_combined.drop(columns=['is_attack'])
y = df_combined['is_attack']

# Split dataset for feature selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model for feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)

# Select top 20 important features
N = 20
important_features = feature_importance[:N].index

# Create a new dataset with important features
df_important = df_combined[important_features]
df_important['is_attack'] = y  # Keep the target variable

# Save the new dataset
df_important.to_csv("important_features_20.csv", index=False)
print("New dataset with important features saved as 'important_features_20.csv'")

# Plot feature importance graph
plt.figure(figsize=(12, 6))
feature_importance[:20].plot(kind='bar', color='skyblue')
plt.title("Top 20 Important Features for IoT DDoS Detection")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.xticks(rotation=45)
plt.grid()
plt.show()
