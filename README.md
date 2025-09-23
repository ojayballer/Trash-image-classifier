# 🗑️ Trash Classifier — Classical Machine Learning Approach

Can classical machine learning compete with deep learning in image classification?  
I set out to prove that it can. This project is a trash classification system (plastic, paper, glass, metal) built entirely with classical ML techniques — no deep learning.

---

## 🚀 Live Demo

You can try the project in two ways:

**Option 1: Run in your browser (no setup required).**  
👉 [Click here to use the Trash Classifier on Render](https://trash-image-classifier.onrender.com)  

**Option 2: Run it locally.**  

```bash
git clone https://github.com/ojayballer/Trash-image-classifier.git
cd Trash-image-classifier
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt
python app.py
Then open http://127.0.0.1:5000/ in your browser.

📖 Project Overview
Most research uses deep learning (CNNs) for image classification.
I wanted to see how far I could go with classical machine learning techniques.

With careful data preprocessing, feature engineering, and insights from over 20 research papers, I achieved:

✅ 76% accuracy (previous benchmark was 67%)
✅ Ability to recognize multiple items in one image (for example paper and metal together)
✅ A fully working web app deployed with Flask on Render

📊 Results
Benchmark exceeded: 76% compared to 67%
Generalization: Works on mixed waste images
Robustness: Fixed data leakage issues and optimized preprocessing

Example prediction (mixed objects):
Metal → 71%
Paper → 62%
Plastic → 5%
Glass → 2%
✔ Correctly identifies both Metal and Paper

⚙️ Tech Stack
Language used: Python
Libraries: scikit-learn, scikit-image, NumPy, Pillow, Flask
Dataset: TrashNet and similar Kaggle datasets
Deployment: Flask web app hosted on Render

📂 Repository Structure
Trash-image-classifier/
│── app.py (Flask web app)
│── Recycling_project.joblib (Saved ML model)
│── requirements.txt (Dependencies)
│── Procfile (Render deploy config)
│── runtime.txt (Python version)
│── README.md (Project documentation)
│── examples/ (Example images)
│── training/ (Training scripts)
│── LICENSE

🧠 Methodology
Data Preprocessing
Images were converted to grayscale, resized to 64×64, and normalized.

Feature Engineering
Histogram of Oriented Gradients captured edges and orientations.
Local Binary Patterns captured textures.
Gray-Level Co-occurrence Matrix captured spatial properties like contrast, homogeneity, and correlation.
The final feature vector was a combination of HOG, LBP, and GLCM.

Model Training
Several classical algorithms such as Support Vector Machines, Random Forest, and Logistic Regression were tested.
The best performing model was selected and saved with joblib.

Evaluation
The model achieved 76% accuracy compared to the 67% benchmark reported in earlier work.
It was also validated on images containing multiple waste items.

🌍 Real-World Impact
Waste management is a global problem.
This project shows that even without deep learning, AI can support automated recycling systems and make waste sorting more efficient.

🙌 Acknowledgments
Inspired by over 20 research papers on trash classification.
Dataset: Kaggle TrashNet.
Thanks to the open-source community for tools and support.

🏷️ License
This project is licensed under the MIT License, free to use and modify.

yaml
Copy code

---
