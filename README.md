# 🌿 Plant Disease Classification with InceptionV3

This project uses a **Convolutional Neural Network (CNN)** based on **InceptionV3** pre-trained on ImageNet to classify plant diseases. It leverages transfer learning to fine-tune the model for a specific plant disease dataset.

---

## 🎯 Features
- **Pre-trained InceptionV3 Model**: Uses the InceptionV3 model with pre-trained weights from ImageNet for faster training.
- **GPU Acceleration**: Configured to use GPU memory efficiently for faster model training.
- **Data Augmentation**: Applied during training to improve model generalization.
- **Image Preprocessing**: Standardized preprocessing for plant disease images.

---

## 🛠️ Technologies Used
- **TensorFlow/Keras**: For deep learning model creation and training.
- **InceptionV3**: Pre-trained model used for transfer learning.
- **ImageDataGenerator**: Used for data augmentation and preprocessing.
- **Python**: The core programming language for building the model.

---

## 🚀 How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/plant-disease-classification.git
   pip install tensorflow numpy
   PlantDiseasesDataset
   python plant_disease_classification.py



🤔 Notes
Ensure you have an active GPU for efficient training (if available).
You can increase the number of epochs for better model performance.
The dataset should be organized in subdirectories where each subdirectory contains images of a particular class.
