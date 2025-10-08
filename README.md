# Heart Disease Detection Using Multilayer Neural Networks  
### Comparative Analysis with Hopfield Network

---

## Abstract
This project presents the design and implementation of a **Heart Disease Detection System** using a **Multilayer Neural Network (MLNN)**.  
The proposed method is compared with two alternative approaches — **Hopfield Neural Network** and **Fuzzy Logic Model** — to evaluate their performance in terms of accuracy, convergence rate, and interpretability.

---

## 🔍 Introduction
Heart disease remains one of the leading causes of death worldwide.  
Accurate and early prediction can significantly improve treatment outcomes.  
Machine learning and deep learning techniques can identify hidden patterns in medical data, making them powerful tools for disease prediction.

---

## Problem Statement
The objective of this project is to develop an efficient and accurate model for detecting heart disease using key medical attributes.  
The focus is on achieving high prediction accuracy while preserving interpretability and computational efficiency.

---

## Related Works and Alternative Approaches
### Classical Machine Learning Models:
- **Support Vector Machine (SVM)**
- **Random Forest (RF)**

### Deep Learning Model:
- **Long Short-Term Memory (LSTM)** network as a performance benchmark.

### Proposed Method:
- **Multilayer Neural Network (MLNN)** integrated with 

---

## Dataset and Preprocessing
The dataset includes attributes such as:
- Age  
- Sex  
- Blood Pressure  
- Cholesterol  
- ECG Results  
- Fasting Blood Sugar  
- Exercise Angina, etc.

**Preprocessing Steps:**
- Handling missing data  
- Normalization and scaling  
- Outlier detection and removal  

---

## Model Architecture and Training
### Hybrid Architecture:
- Input Layer with Fuzzy Weighting  
- Multiple Hidden Layers with ReLU activation  
- Output Layer with Sigmoid activation  

### Training Process:
- Optimizer: Adam  
- Loss Function: Binary Cross-Entropy  
- Regularization: Dropout  
- Early Stopping for preventing overfitting  

---

## Optimization and Real-Time Implementation
- Hyperparameter tuning using **Grid Search** and **Genetic Algorithms**  
- Implementation optimized for **real-time predictions**  
- Model evaluation on both training and unseen test data  

---

## Results and Evaluation
### Training and Convergence Analysis:
- MLNN shows faster and smoother convergence compared to Hopfield

### Final Performance Metrics:
| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|-----------|---------|-----------|
| MLNN (Proposed) | 96.4% | 95.8% | 97.1% | 96.4% |
| Hopfield Network | 89.7% | 88.5% | 90.2% | 89.3% |

### Confusion Matrix and Classification Report:
Included in the results section with graphical plots and interpretation.

---

## Discussion and Conclusion
The proposed **MLNN with Fuzzy Feature Weighting** outperforms traditional models in terms of classification accuracy and convergence stability.  
The hybrid design enhances interpretability and demonstrates strong potential for integration in **medical decision support systems**.

---

## Technologies Used
- Python  
- TensorFlow / Keras  
- NumPy / Pandas  
- Scikit-learn  
- Matplotlib / Seaborn  


