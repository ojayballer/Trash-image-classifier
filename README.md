🗑️ Trash Classifier — Classical Machine Learning Approach

Can classical machine learning compete with deep learning in image classification?
I set out to prove that it can. This project is a trash classification system (plastic, paper, glass, metal) built entirely with classical ML techniques — no deep learning.

🚀 Live Demo

You can try the project in two ways:

Option 1: Run in your browser (no setup required)
👉 Trash Classifier on Render

Option 2: Run it locally

git clone https://github.com/ojayballer/Trash-image-classifier.git
cd Trash-image-classifier
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt
python app.py


Then open http://127.0.0.1:5000/
 in your browser.

📖 Project Overview

Most research uses deep learning (CNNs) for image classification.
I wanted to see how far I could go with classical machine learning techniques.

With careful data preprocessing, feature engineering, and insights from 20+ research papers, I achieved:

✅ 76% accuracy (previous benchmark was 67%)
✅ Ability to recognize multiple items in one image (e.g., paper + metal)
✅ A fully working web app deployed with Flask on Render

📊 Results

Benchmark exceeded: 76% vs 67%
Generalization: Handles mixed waste images effectively
Robustness: Fixed data leakage issues and optimized preprocessing

Example prediction (mixed objects):

Metal → 71%

Paper → 62%

Plastic → 5%

Glass → 2%

✔ Correctly identifies both Metal + Paper

⚙️ Tech Stack

Language: Python
Libraries: scikit-learn, scikit-image, NumPy, Pillow, Flask
Dataset: Kaggle TrashNet and similar datasets
Deployment: Flask app hosted on Render

📂 Repository Structure
Trash-image-classifier/
│── app.py                   # Flask web app
│── Recycling_project.joblib # Saved ML model
│── requirements.txt         # Dependencies
│── Procfile                 # Render deploy config
│── runtime.txt              # Python version
│── README.md                # Project documentation
│── examples/                # Example images
│── training/                # Training scripts
│── LICENSE

🧠 Methodology

Data Preprocessing
Images were converted to grayscale, resized to 64×64, and normalized.

Feature Engineering

HOG (edges and orientations)

LBP (texture features)

GLCM (spatial properties: contrast, homogeneity, correlation)

Combined into a single feature vector

Model Training
Tried SVM, Random Forest, Logistic Regression.
Selected the best performing model and saved it with joblib.

Evaluation
Achieved 76% accuracy (compared to 67% benchmark).
Validated on images with multiple waste items.

🌍 Real-World Impact

Waste management is a global challenge.
This project shows how AI (even without deep learning) can power automated recycling systems — detecting and sorting waste more efficiently.

🙌 Acknowledgments

Inspired by 20+ research papers on trash classification.
Dataset: Kaggle TrashNet.
Thanks to the open-source community 💙

🏷️ License

This project is licensed under the MIT License — free to use and modify.
