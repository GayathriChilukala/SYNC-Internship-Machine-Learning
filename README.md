<div align="center">

# 🤖 SYNC Internship — Machine Learning Projects
### 4 Projects · 3 Domains · NLP · Computer Vision · Regression

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

**Completed by [Gayathri Chilukala](https://github.com/GayathriChilukala)**

</div>

---

## 📌 About This Repository

This repository contains four machine learning projects completed during the **SYNC Internship Program**, spanning three core ML domains — **Natural Language Processing**, **Computer Vision**, and **Regression Modeling**. Each task was designed to build hands-on proficiency across the full breadth of modern machine learning applications.

> 💡 **For Recruiters:** SYNC Internships is a recognized virtual internship program providing structured, mentored ML projects. This internship stands out for its deliberate domain diversity — rather than repeating similar tasks, each project addresses a fundamentally different ML problem type, demonstrating adaptability across the field.

---

## 📁 Project Structure

```
SYNC-Internship-Machine-Learning/
├── Task 1-ChatBot.ipynb               → NLP · Conversational AI
├── Task 2-Mask_Detection.ipynb        → Computer Vision · Real-Time Detection
├── Task 3-BostanHousePrediction.ipynb → Regression · Predictive Modeling
└── Task 4-SignClassification.ipynb    → Computer Vision · Image Classification
```

---

## 🚀 Projects Overview

---

### 💬 Task 1 — AI ChatBot
> **Domain:** Natural Language Processing (NLP) · Conversational AI
> **Tech:** Python · NLTK · TensorFlow/Keras · JSON · pickle

An intelligent chatbot that understands user intent and responds contextually — built from scratch using NLP preprocessing and a neural network classifier.

**How It Works:**
- Defined intents, patterns, and responses in a structured JSON corpus
- Applied NLP preprocessing: tokenization, lemmatization, and bag-of-words vectorization
- Trained a neural network (Dense layers + Dropout) to classify user input into intent categories
- Implemented a response engine that maps predicted intents to contextual replies
- Serialized the trained model and vocabulary for efficient inference at runtime

**Key Highlights:**
- End-to-end pipeline from raw intent corpus → trained model → live chat interface
- Handles multi-turn conversational patterns and graceful fallback responses
- Demonstrates core NLP concepts: intent recognition, text normalization, bag-of-words encoding

---

### 😷 Task 2 — Face Mask Detection
> **Domain:** Computer Vision · Real-Time Object Detection
> **Tech:** Python · TensorFlow · Keras · OpenCV · CNN

A real-time face mask detection system using a Convolutional Neural Network (CNN) to classify whether individuals in images or video frames are wearing masks — a direct application of CV to public health safety.

**How It Works:**
- Collected and preprocessed labeled image dataset: `with_mask` / `without_mask` classes
- Built and trained a CNN architecture with Conv2D, MaxPooling, Flatten, and Dense layers
- Applied data augmentation (rotation, flip, zoom) to improve generalization on unseen faces
- Integrated OpenCV's Haar Cascade face detector for face localization before classification
- Achieved real-time detection by applying the trained model frame-by-frame on video input

**Key Highlights:**
- Combined two-stage pipeline: face detection (OpenCV) → mask classification (CNN)
- Real-time inference capability on live webcam or video feeds
- Practical, deployment-ready application demonstrating production CV thinking

---

### 🏠 Task 3 — Boston House Price Prediction
> **Domain:** Supervised Learning · Regression · Predictive Modeling
> **Tech:** Python · Scikit-learn · Pandas · NumPy · Matplotlib · Seaborn

A regression project using the classic Boston Housing dataset to predict property prices based on socioeconomic and structural features. Emphasizes the full regression pipeline from EDA through model selection and evaluation.

**How It Works:**
- Loaded and explored the Boston Housing dataset (13 features, 506 samples)
- Conducted EDA: feature distributions, correlation heatmap, outlier detection
- Applied feature scaling (StandardScaler) and handled multicollinearity
- Trained and compared multiple regression models: Linear Regression, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor
- Selected the best model based on RMSE, MAE, and R² score
- Visualized predicted vs. actual prices and residuals for model diagnostics

**Key Highlights:**
- Multi-model comparison with quantitative evaluation metrics (RMSE, R²)
- Residual analysis to validate model assumptions
- Full regression workflow mirroring real-world predictive analytics pipelines

---

### 🚦 Task 4 — Sign Classification
> **Domain:** Computer Vision · Multi-Class Image Classification
> **Tech:** Python · TensorFlow · Keras · CNN · NumPy · Matplotlib

A deep learning model that classifies signs (traffic signs or hand gesture signs) across multiple categories — solving a real-world multi-class visual recognition problem using CNNs.

**How It Works:**
- Preprocessed a multi-class labeled image dataset (resizing, normalization, one-hot encoding)
- Designed and trained a CNN with multiple Conv2D + MaxPooling blocks, Batch Normalization, and Dropout for regularization
- Applied data augmentation to simulate real-world variation (lighting, angle, scale)
- Evaluated model using per-class accuracy, confusion matrix, and classification report
- Visualized correctly and incorrectly classified samples for error analysis

**Key Highlights:**
- Multi-class CNN classification across diverse sign categories
- Batch Normalization for faster convergence and training stability
- Confusion matrix analysis revealing per-class model strengths and weaknesses

---

## 🌐 Domain Coverage at a Glance

```
┌─────────────────────────────────────────────────────────────┐
│                  SYNC ML Internship Coverage                │
├──────────────────┬──────────────────────────────────────────┤
│ NLP              │  ChatBot — Intent Classification         │
├──────────────────┼──────────────────────────────────────────┤
│ Computer Vision  │  Mask Detection — Binary Classification  │
│                  │  Sign Classification — Multi-class CNN   │
├──────────────────┼──────────────────────────────────────────┤
│ Regression       │  House Price Prediction — Multi-model    │
└──────────────────┴──────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Tools & Libraries |
|---|---|
| **Language** | Python 3.8+ |
| **Deep Learning** | TensorFlow, Keras |
| **Computer Vision** | OpenCV, CNN (Conv2D, MaxPooling, BatchNorm) |
| **NLP** | NLTK (tokenization, lemmatization), bag-of-words |
| **Machine Learning** | Scikit-learn (regression models, evaluation) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | Jupyter Notebook |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow keras opencv-python nltk scikit-learn pandas numpy matplotlib seaborn jupyter
```

```python
# Also download NLTK data (run once in Python)
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

### Run Any Notebook

```bash
# 1. Clone the repository
git clone https://github.com/GayathriChilukala/SYNC-Internship-Machine-Learning.git
cd SYNC-Internship-Machine-Learning

# 2. Launch Jupyter
jupyter notebook

# 3. Open and run any task notebook independently
```

All notebooks are self-contained with inline dataset loading and preprocessing.

---

## 🎯 Skills Demonstrated

This internship showcases the breadth of skills expected from a **Machine Learning Engineer or Data Scientist** working across multiple product domains:

| Skill Area | Task |
|---|---|
| **NLP & Intent Recognition** | Task 1 — ChatBot |
| **Neural Network Design** | Tasks 1, 2, 4 — Dense & CNN architectures |
| **Real-Time Computer Vision** | Task 2 — Mask Detection with OpenCV |
| **Multi-Class Image Classification** | Task 4 — Sign Classification |
| **Regression & Model Comparison** | Task 3 — House Price Prediction |
| **Data Augmentation** | Tasks 2, 4 — Image augmentation for generalization |
| **Model Evaluation** | All tasks — Accuracy, RMSE, R², Confusion Matrix |
| **End-to-End ML Pipelines** | All tasks — Data → Preprocessing → Training → Evaluation |

---

## 📜 Internship Details

| Detail | Info |
|---|---|
| **Program** | SYNC Internships — Machine Learning Track |
| **Tasks Completed** | 4 / 4 ✅ |
| **Domains Covered** | NLP · Computer Vision · Regression |
| **Techniques Used** | CNN, DNN, LSTM, Bag-of-Words, Data Augmentation |
| **Certification** | Issued upon successful completion |

---

## 🤝 Connect

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-GayathriChilukala-181717?style=for-the-badge&logo=github)](https://github.com/GayathriChilukala)

</div>

---

<div align="center">

*From language to vision to prediction — machine learning across every dimension.* 🤖

</div>
