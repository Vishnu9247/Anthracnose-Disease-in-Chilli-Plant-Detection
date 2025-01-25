# **Anthracnose Disease in Chilli Plant Detection 🌶️ (Using Deep Learning to Identify Affected Plants)**

## **Overview 🧠**

The **Anthracnose Disease Detection in Chilli Plants** project leverages deep learning to provide a powerful solution for identifying chili plants affected by anthracnose disease. This fungal disease can devastate crops, reducing yields and affecting agricultural productivity. By utilizing advanced image classification techniques, this project helps detect early signs of infection, allowing farmers to take timely action and mitigate the disease’s spread. 🌱

## **Dataset 📊**

The dataset used for this project was sourced from Kaggle, containing images of chili plants categorized as healthy or affected by anthracnose disease.

- **Dataset Link:** [Kaggle - Anthracnose Disease in Chilli Plant](https://www.kaggle.com/datasets/prudhvi143413s/anthracnose-disease-in-chilli-mobile-captured)

## **Model Architecture 🏗️**

For this project, we utilize the **ResNet-50** architecture, known for its impressive performance in image classification tasks. ResNet-50 is a deep convolutional neural network (CNN) with 50 layers, enabling it to learn intricate patterns and features from input images. The model is fine-tuned to effectively classify images into two categories: **Healthy** ✅ and **Anthracnose** 🍂.

## **Training Process 🏋️**

### **Data Preprocessing 🔄:**
- **Resizing:** All images were resized to a consistent dimension for uniformity in the input.
- **Normalization:** Pixel values were normalized to enhance model convergence and training stability.

### **Model Initialization ⚙️:**
- Imported the **pre-trained ResNet-50 model** with weights trained on **ImageNet**.
- Removed the top classification layer to adapt the model for our specific problem.

### **Custom Classification Layer 🧩:**
- Added a **fully connected layer** at the end with **two output classes**: **“Healthy”** ✅ and **“Anthracnose”** 🍂.

### **Loss Function 📉:**
- Used **Cross Entropy Loss** to quantify the difference between the predicted and actual labels during training.

### **Optimizer ⚡:**
- The **Adam optimizer** was employed for efficient gradient-based optimization.

## **Training Strategy 💡**

- **Data Split:** The dataset was divided into **training**, **validation**, and **test** sets to evaluate model performance effectively.
- **Fine-tuning:** The ResNet-50 model was fine-tuned on our chili plant images, adjusting its weights for the specific task.
- **Monitoring:** We tracked **training loss** and **validation accuracy** throughout the process to ensure the model was improving and to prevent overfitting.
- **Evaluation:** After training, the model was evaluated on the **test set** to assess its final performance.

## **Data Augmentation 🎨**

To enhance the robustness of the model and improve its ability to generalize to unseen data, we applied **data augmentation** techniques. These included:
- Random **rotation** 🔄 and **flipping** ↩️ of images.
- Adjustments to **brightness** 🌞 and **contrast** 🌚.

Augmentation helps the model learn invariant features and prevents overfitting, ensuring better performance on new, unseen images.

## **Model Evaluation 📊**

To measure the effectiveness of the trained model, we used the following evaluation metrics:
- **Accuracy:** The Model predicted the disease with 96.12% acuuracy ✅.

## **Conclusion 🎯**

This deep learning model for detecting anthracnose disease in chili plants offers an automated, scalable solution for farmers. By utilizing cutting-edge image classification techniques, this system provides early disease detection, helping safeguard crops and optimize agricultural practices. 🌾🌟

## **Technologies Used 🔧**
- **Deep Learning:** ResNet-50, Convolutional Neural Networks (CNNs)
- **Frameworks:** PyTorch
- **Optimization:** Adam Optimizer
- **Metrics:** Accuracy, Cross Entropy Loss, Confusion Matrix

---
