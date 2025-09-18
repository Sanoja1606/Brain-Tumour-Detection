**Brain Tumor Classification**

This repository contains a deep learning project for classifying brain tumors from MRI images. The model is trained to distinguish between four categories: glioma, meningioma, pituitary tumor, and no tumor.

The project is built using a Convolutional Neural Network (CNN) and includes a training notebook and a Streamlit web application for real-time predictions.

**Solution Overview**

The solution consists of a two-stage process: training and deployment.

Training (training.ipynb): This Jupyter Notebook handles the entire machine learning pipeline. It starts with loading and preprocessing the MRI image dataset. The images are resized and normalized to be suitable for a neural network. A Convolutional Neural Network (CNN) architecture is defined using TensorFlow/Keras, which is particularly effective for image classification tasks. The model is then trained on the dataset, and its performance is evaluated on a separate test set. The trained model is saved as a .h5 file for later use.

Deployment (app.py): This is a user-friendly web application built with Streamlit. It loads the pre-trained model and provides an interface for users to upload their own brain MRI images. The uploaded image is preprocessed in the same way as the training data, passed to the model for prediction, and the predicted tumor type (or "no tumor") is displayed to the user.

**Files**

training.ipynb: A Jupyter Notebook containing the complete workflow for data loading, model training, and evaluation. It includes the code for generating the confusion matrix and calculating the final accuracy.

app.py: A Streamlit application that allows users to upload an MRI image and receive an instant prediction from the trained model.

brain_tumor_detection_model.h5: The trained Keras model file.

**Model Performance**

<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/bdc06c71-13fd-431f-b437-358add65f6b4" />


The deep learning model was evaluated on a test dataset to assess its performance. The overall accuracy of the model is 91.25%.

**Confusion Matrix Analysis**

<img width="649" height="548" alt="image" src="https://github.com/user-attachments/assets/356e5ab8-bc62-4856-bed4-9c65e0c337b5" />


The confusion matrix provides a detailed breakdown of the model's performance, showing the number of correct and incorrect predictions for each class.

By analyzing the confusion matrix, we can gain deeper insights into the model's strengths and weaknesses:

True Positives (TP): These are the number of images correctly classified for each class (e.g., a glioma image is correctly identified as glioma). The diagonal values of the matrix represent these. The higher these numbers are, the better the model is at correct identification.

False Positives (FP): These are cases where the model incorrectly predicts a class. For example, a no tumor image might be misclassified as a pituitary tumor. This is also known as a Type I error and can be particularly critical in medical applications.

False Negatives (FN): These are cases where the model fails to detect the correct class. For instance, a meningioma tumor image might be classified as no tumor. This is also known as a Type II error and can be a significant concern as it represents a missed detection.

The confusion matrix analysis indicates that while the model has a high overall accuracy, it's essential to examine the misclassifications. The model generally performs well in distinguishing between tumor and non-tumor cases, but there may be some confusion between the different types of tumors, which is a common challenge in medical image classification.

*How to Run the App*

Clone the repository:

git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

*Run the Streamlit application:*

streamlit run app.py

The application will open in your default web browser, and you can start uploading images for prediction.

**Sample working**
<img width="988" height="788" alt="image" src="https://github.com/user-attachments/assets/60bf28a1-7010-49ba-9e2c-466d448e3f3b" /><img width="1118" height="731" alt="image" src="https://github.com/user-attachments/assets/889b3e87-afd8-4bd0-942d-b47e1a222703" />

<img width="982" height="789" alt="image" src="https://github.com/user-attachments/assets/58c27736-d0e9-419a-86a6-335b4cb687b5" />



Add an option to display the confidence score for each prediction.<img width="982" height="789" alt="image" src="https://github.com/user-attachments/assets/497f80e8-a2d8-403d-970d-76ab87e3d43d" />
