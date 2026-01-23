# Celebrity-Face-Classification-using-Wavelet-Transform-and-Machine-Learning
A machine learning–based web application that classifies celebrity faces using image processing, wavelet feature extraction, and a trained classifier, deployed with Streamlit.

According to the authors, this study investigates the use of wavelet transform and machine learning to classification of celebrity faces.

It is a machine learning end-to-end project, which involves classification of celebrity faces based on an uploaded photo. It uses classical image processing methods and machine learning with an interactive web interface through Streamlit.

This system operates on the Open CV to process the images, Wavelet Transform to extract the features and a trained Scikit-learn model to classify the features.

---

## 🚀 Features

- Load a face image and become familiar with known classes of celebrity.
- Extraction of high-frequency facial features using **Wavelet Transform (Haar)].
- Binomial pixel features coupled with wavelet features are combined to enhance accuracy.
- Shows the likelihood of predictions of the individual celebrity.
- Interactive and clean Streamlit UI.

---

## 🧠 Celebrities Supported

- Lionel Messi  
- Cristiano Ronaldo  
- Roger Federer  
- Serena Williams  
- Maria Sharapova  

List of classes may be extended without much difficulty.

---

## 🛠️ Tech Stack

- **Python**
- **OpenCV**
- **NumPy**
- **PyWavelets**
- **Scikit-learn**
- **Pandas**
- **Streamlit**

---

## ⚙️ How It Works

1. **Image Upload**
   - User posts an image through Streamlit UI.

2. **Preprocessing**
   - Image is resized to 32×32
   - Converted to grayscale
   - Wavelet Transform Haar is utilized.

3. **Feature Engineering**
   - Pixels of raw images + wavelet features are appended.

4. **Prediction**
   - A trained Scikit-learn model predicts:
     - Celebrity class
     - The likelihood scores of each of the classes.


