# Flowers Recognition Project

This project implements a deep learning model to classify images of flowers into five categories: daisy, dandelion, rose, sunflower, and tulip.

## 1. Dataset
- **Source:** Kaggle `alxmamaev/flowers-recognition`.
- **Content:** 4,317 images.
- **Classes:** 5 (daisy, dandelion, rose, sunflower, tulip).
- **Split:** 80% training (3,454 images), 20% validation (863 images).

## 2. Preprocessing
- **Image Size:** All images resized to 224x224 pixels.
- **Batch Size:** 32.
- **Format:** Images loaded as a TensorFlow dataset with integer labels.

## 3. Model Architecture
- **Base Model:** ResNet50 (Pre-trained on ImageNet) with `include_top=False`.
- **Fine-tuning:** The base model was set to trainable, with specific top layers frozen to preserve feature extraction.
- **Custom Top Layers:**
    - `GlobalAveragePooling2D` to reduce spatial dimensions.
    - `Dropout(0.3)` for regularization.
    - `Dense(128, activation='relu')`.
    - `Dense(5, activation='softmax')` for multi-class classification.

## 4. Training
- **Optimizer:** Adam (learning rate = 0.0001).
- **Loss Function:** `sparse_categorical_crossentropy`.
- **Epochs:** 5.
- **Result:** Reached ~94% training accuracy and ~87% validation accuracy.

## 5. Inference
- The model has been tested on individual images (sunflower, dandelion, tulip, rose) with high confidence scores for the correct labels.
