# Human Activity Recognition using Deep Learning Model Comparison

A deep learning-based **Human Activity Recognition (HAR)** project focused on evaluating and comparing transfer learning architectures for image-based activity classification. This project analyzes the performance of **EfficientNetB3**, **ResNet50**, and **MobileNetV2** using validation trends, fine-tuning behavior, and class-level evaluation metrics.

The objective was not only to build a classifier, but also to perform a detailed **model performance comparison** across architectures.

---

## Overview

Human Activity Recognition (HAR) identifies human actions from visual inputs. This implementation uses transfer learning with pretrained CNN models to classify **15 activity categories**.

### Activities

- Calling
- Clapping
- Cycling
- Dancing
- Drinking
- Eating
- Fighting
- Hugging
- Laughing
- Listening to Music
- Running
- Sitting
- Sleeping
- Texting
- Using Laptop

---

## Features

- Transfer Learning with multiple architectures
- Fine-tuning strategy for performance optimization
- Validation Accuracy comparison
- Validation Loss comparison
- Precision, Recall and F1-score evaluation
- Training convergence analysis
- Class-wise performance comparison
- Comparative deep learning study

---

## Models Used

### EfficientNetB3

- Validation Accuracy: **81%**
- Validation Loss: **0.67**
- Best performing architecture
- Stable convergence after fine-tuning

### ResNet50

- Validation Accuracy: **~76%**
- Moderate performance
- Strong feature extraction capability

### MobileNetV2

- Validation Accuracy: **74%**
- Lightweight architecture
- Lower computational requirements

---

## Training Strategy

The project follows a two-stage transfer learning pipeline.

### Stage 1

- Freeze pretrained backbone
- Train classification head

### Stage 2

- Unfreeze upper layers
- Apply fine-tuning
- Use reduced learning rate

Fine-tuning begins after initial feature adaptation to improve model specialization.

---

## Model Performance Comparison

| Model          | Accuracy | Validation Loss |
| -------------- | -------: | --------------: |
| EfficientNetB3 |      81% |            0.67 |
| ResNet50       |      76% |            0.88 |
| MobileNetV2    |      74% |            0.96 |

### Key Findings

- **EfficientNetB3 achieved highest overall accuracy**
- Lower validation loss after fine-tuning
- Strong convergence behavior
- Better class-wise prediction quality
- MobileNetV2 offered lightweight deployment benefits

---

## Classification Highlights

### EfficientNetB3

| Activity     |        Metric |
| ------------ | ------------: |
| Cycling      | 98% Precision |
| Eating       |  90% F1-score |
| Running      |  85% F1-score |
| Using Laptop |  82% F1-score |

### MobileNetV2 Observations

- Faster inference
- Lower performance in difficult activities
- Better suited for resource-constrained environments

---

## Training Visualizations

Included visualizations:

- Validation Accuracy Comparison
- Validation Loss Comparison
- Fine-tuning Start Marker
- Convergence Analysis

These graphs show how different architectures respond during training and fine-tuning stages.

---

## Project Structure

```
HAR-Model-Comparison/
│
├── dataset/
├── notebooks/
│   └── training.ipynb
│
├── models/
│   ├── EfficientNetB3
│   ├── ResNet50
│   └── MobileNetV2
│
├── outputs/
│   ├── accuracy_plots
│   ├── loss_plots
│   ├── confusion_matrices
│   └── classification_reports
│
├── requirements.txt
└── README.md
```

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Scikit-learn
- Matplotlib
- Pandas

---

## Installation

```bash
git clone https://github.com/adityasatuluri/Human-Activity-Recognition.git
cd HAR-Model-Comparison
pip install -r requirements.txt
jupyter notebook
```

---

## Future Improvements

- Vision Transformers (ViT)
- Ensemble learning
- Real-time webcam recognition
- Streamlit deployment
- Hyperparameter optimization

---

## Results Summary

Among all tested architectures, **EfficientNetB3 consistently achieved the best balance of accuracy, convergence stability, and class-wise prediction performance**, demonstrating the importance of comparative model evaluation rather than selecting architectures based solely on popularity.

---

## Author

**Aditya Satuluri**
