# DDoS Detection using CNN-LSTM with SMOTE

## 📌 Overview
This repository contains an advanced **DDoS attack detection model** that leverages **CNN-LSTM (Convolutional Neural Network - Long Short-Term Memory)** architecture. To handle data imbalance, **Synthetic Minority Over-sampling Technique (SMOTE)** has been applied, ensuring better performance and fairness in classification.

## 🚀 Features
- **Hybrid CNN-LSTM Model** for sequential pattern recognition in network traffic.
- **SMOTE for Data Balancing** to improve classification on minority attack classes.
- **Feature Selection** for optimal feature extraction.
- **Model Performance Evaluation** using various metrics.
- **Visualization** of confusion matrix, ROC-AUC curve, and accuracy/loss trends.

## 📂 Dataset
The dataset used for training and evaluation is based on network traffic data with labeled attack patterns. The dataset was preprocessed to extract optimal features.

## 🛠️ Approach
1. **Data Preprocessing**
   - Load and clean dataset.
   - Apply **SMOTE** for oversampling imbalanced classes.
   - Perform **feature selection** to retain the most relevant features.
   - Normalize data using **StandardScaler**.
   
2. **Model Architecture**
   - **CNN Layer**: Extract spatial features from input data.
   - **LSTM Layers**: Capture temporal dependencies in network traffic.
   - **Dense Layers**: Fully connected layers for final classification.
   - **Dropout & L2 Regularization**: Prevent overfitting.
   
3. **Model Training & Evaluation**
   - Train-test split with **stratification**.
   - Use **Adam optimizer** with categorical cross-entropy loss.
   - Apply **Early Stopping** to avoid overfitting.
   - Evaluate using **Accuracy, Precision, Recall, F1-score, and AUC-ROC**.
   - Generate **confusion matrix and loss/accuracy plots**.

## 📊 Performance Metrics
| Metric       | Score  |
|-------------|--------|
| Accuracy    | 97.8%  |
| Precision   | 96.5%  |
| Recall      | 95.2%  |
| F1-Score    | 95.8%  |
| ROC-AUC     | 98.3%  |

## 📈 Visualizations
- **Confusion Matrix**: Displays correct vs incorrect classifications.
- **ROC-AUC Curve**: Evaluates the classifier's performance.
- **Training History Graphs**: Shows accuracy and loss over epochs.

## 🖥️ Installation & Usage
### 🔧 Prerequisites
Ensure you have the following installed:
```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

### 📌 Clone the Repository
```bash
git clone https://github.com/Faizalimam990/DDOS-LSTM.git
cd DDOS-LSTM
```

### 🚀 Train the Model
Run the following command to train the model:
```bash
python train.py
```

### 📊 Evaluate the Model
After training, you can evaluate the model using:
```bash
python evaluate.py
```

### 📁 Save & Load Model
Save the trained model:
```python
model.save("cnn_lstm_final_model.h5")
```
Load and use the model:
```python
from tensorflow.keras.models import load_model
model = load_model("cnn_lstm_final_model.h5")
```

## 🔮 Future Improvements
- Fine-tuning hyperparameters for better accuracy.
- Implementing real-time detection using streaming data.
- Expanding dataset with more attack types.
- Deploying as a web API for real-time classification.

## 🤝 Contributing
Feel free to contribute! Fork the repo, create a branch, make improvements, and submit a pull request.

## 📜 License
This project is licensed under the **MIT License**.

---
🚀 Developed by [Faizal Imam](https://github.com/Faizalimam990) | 💻 **DDoS A
