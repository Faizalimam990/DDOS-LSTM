import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_curve, auc, 
                             precision_recall_fscore_support, matthews_corrcoef, cohen_kappa_score)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, Flatten, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load dataset
df = pd.read_csv("optimal_features.csv")
X = df.drop(columns=['is_attack']).values
y = df['is_attack'].values

# Feature Selection
selector = SelectKBest(score_func=f_classif, k=20)
X = selector.fit_transform(X, y)

# Feature Importance Plot
feature_scores = selector.scores_
plt.figure(figsize=(12, 6))
sns.barplot(x=np.arange(len(feature_scores)), y=feature_scores, palette='viridis')
plt.xlabel("Feature Index")
plt.ylabel("Importance Score")
plt.title("Feature Importance Analysis")
plt.show()

# TSNE Visualization
X_embedded = TSNE(n_components=2, random_state=42).fit_transform(X)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y, palette='coolwarm')
plt.title("TSNE Visualization of Feature Space")
plt.show()

# One-hot Encoding
y = to_categorical(y)

# Train-test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape for CNN-LSTM
X_train = X_train.reshape((X_train.shape[0], X.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X.shape[1], 1))

# Define CNN-LSTM Model
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01), input_shape=(X.shape[1], 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    LSTM(32, kernel_regularizer=l2(0.01)),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(y.shape[1], activation='softmax')
])

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Model Summary
model.summary()

# Train Model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping, lr_scheduler], verbose=1)

# Predictions
y_pred = np.argmax(model.predict(X_test), axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Evaluation
accuracy = accuracy_score(y_test_labels, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test_labels, y_pred, average='weighted')
mcc = matthews_corrcoef(y_test_labels, y_pred)
kappa = cohen_kappa_score(y_test_labels, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print("\nClassification Report:\n", classification_report(y_test_labels, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_labels, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
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
plt.title("ROC Curve")
plt.legend()
plt.show()

# Accuracy & Loss Graphs
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()

# Save Model
model.save("cnn_lstm_enhanced_model.h5")
with open("training_history_enhanced.pkl", "wb") as file:
    pickle.dump(history.history, file)
print("Model and training history saved successfully.")
