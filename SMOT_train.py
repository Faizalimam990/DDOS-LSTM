import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_fscore_support

# Load dataset
df = pd.read_csv("optimal_features.csv")  # Ensure the file exists

# Feature Extraction
X = df.drop(columns=['is_attack']).values  # Extract features
y = df['is_attack'].values  # Target variable

# Feature Selection - Keep only the top 20 features
selector = SelectKBest(score_func=f_classif, k=20)  # Change 'k' as needed
X = selector.fit_transform(X, y)

print(f"Total number of selected features: {X.shape[1]}")

# One-hot encoding for classification
y = to_categorical(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape for CNN-LSTM
X_train = X_train.reshape((X_train.shape[0], X.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X.shape[1], 1))

# Handle Imbalanced Dataset using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train.reshape(X_train.shape[0], -1), np.argmax(y_train, axis=1))

y_train_resampled = to_categorical(y_train_resampled)
X_train_resampled = X_train_resampled.reshape((X_train_resampled.shape[0], X.shape[1], 1))

# Define CNN-LSTM Model
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    LSTM(32, return_sequences=True),
    Dropout(0.5),
    LSTM(32),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),  # L2 Regularization
    Dense(y.shape[1], activation='softmax')
])

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Early Stopping & Model Checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True)

# Train Model
history = model.fit(
    X_train_resampled, y_train_resampled,
    epochs=50, batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1, callbacks=[early_stopping, model_checkpoint]
)

# Predictions
y_pred = np.argmax(model.predict(X_test), axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Model Evaluation
accuracy = accuracy_score(y_test_labels, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test_labels, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")
print("\nClassification Report:\n", classification_report(y_test_labels, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_labels, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test_labels), yticklabels=np.unique(y_test_labels))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ROC Curve & AUC
fpr, tpr, _ = roc_curve(y_test_labels, y_pred, pos_label=1)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.show()

# Accuracy Graph
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Over Epochs")
plt.legend()
plt.show()

# Save Model & Training History
model.save("cnn_lstm_final_model.h5")
with open("training_history_final.pkl", "wb") as file:
    pickle.dump(history.history, file)

print("Model and training history saved successfully.")
