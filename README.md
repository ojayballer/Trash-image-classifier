
# Project Title

A brief description of what this project does and who it's for

# 🗑️ Trash Classifier — Classical Machine Learning Approach
```bash
A machine learning web app that classifies trash into **plastic, paper, glass, or metal** using classical ML techniques (no deep learning).  
Deployed with Flask on Render.  
```
## 🚀 Installation and Live Demo
nstallation / Run Locally
## 💻 Run Locally

Clone the repo:
```bash
git clone https://github.com/ojayballer/Trash-image-classifier.git
cd Trash-image-classifier
```
Set up a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the Flask app:
```bash
python app.py
```

Open in browser:
```bash
👉 http://127.0.0.1:5000/
```
Live Demo
```bash
👉 Try it online: [Trash Classifier on Render](https://trash-image-classifier.onrender.com)  

Upload a **.jpg image** of trash (e.g., bottle, newspaper, soda can) and the model will predict its category in real time
```

## Features
```bash
✅ 76% accuracy (benchmark was 67%)  
✅ Can detect multiple items in one image (e.g., paper + metal together)  
✅ Fully deployed as a Flask web app on Render  
✅ Robust preprocessing and optimized feature extraction  
```
## ⚙️ Tech Stack  

Language: Python
Libraries: scikit-learn, scikit-image, NumPy, Pillow, Flask
Dataset: TrashNet & similar Kaggle datasets
Deployment: Flask + Render
##  🖼️ Sample UI  


### Web App Interface
![Web App Interface](examples/Screenshot (1678).png)

### Example Prediction
![Prediction Example](examples/Screenshot (1678).png)

## 🧠 Methodology


```bash
Data Preprocessing

Converted all images to grayscale

Resized to 64×64

Normalized pixel values

Feature Engineering

HOG: captured edges and orientations

LBP: captured textures

GLCM: captured spatial patterns (contrast, homogeneity, correlation)

Final feature vector = HOG + LBP + GLCM

Model Training

Tried SVM, Random Forest, Logistic Regression

Selected the best performing model

Saved with joblib

Evaluation

Achieved 76% accuracy (benchmark was 67%)

Works with images containing multiple objects
```
## 🌍 Real-World Impact
```bash
Waste management is a global challenge.
This project shows how AI — even classical ML without deep learning — can help automate recycling systems by detecting and sorting waste more efficiently.
```
## 🙌 Acknowledgments
```bash
Inspired by 20+ research papers on trash classification

Dataset: Kaggle TrashNet

Thanks to the open-source community 💙
```
## 🏷️ License
```bash
This project is licensed under the MIT License — free to use and modify
```
