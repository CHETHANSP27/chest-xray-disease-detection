# 🫁 AI-Based Chest X-Ray Disease Detection

An explainable AI system for detecting multiple thoracic diseases from chest X-ray images using Deep Learning.

## 🎯 Features

- **Multi-Label Classification**: Detects 14 different thoracic diseases simultaneously
- **Explainable AI**: Grad-CAM visualizations show which regions influenced predictions
- **High Accuracy**: DenseNet121 architecture achieves ~0.85 AUC
- **Web Interface**: User-friendly Streamlit application
- **Production-Ready**: Complete training and deployment pipeline

## 🩺 Detectable Diseases

1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural Thickening
14. Hernia

## 📊 Dataset

**NIH Chest X-ray14 Dataset**
- 112,120 frontal-view X-ray images
- 30,805 unique patients
- 14 disease labels
- Source: [Kaggle NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data)

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 50GB free disk space for dataset

### Setup Steps

1. **Clone/Create Project Directory**
