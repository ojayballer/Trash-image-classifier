title: "🗑️ Trash Classifier — Classical Machine Learning Approach"
description: |
  Can classical machine learning compete with deep learning in image classification?
  I set out to prove that it can. This project is a trash classification system 
  (plastic, paper, glass, metal) built entirely with classical ML techniques — no deep learning.

live_demo: |
  You can try the project in two ways:

  Option 1: Run in your browser (no setup required)
  👉 https://trash-image-classifier.onrender.com

  Option 2: Run it locally
    git clone https://github.com/ojayballer/Trash-image-classifier.git
    cd Trash-image-classifier
    python -m venv venv
    source venv/bin/activate   # Linux/Mac
    venv\Scripts\activate      # Windows
    pip install -r requirements.txt
    python app.py

  Then open http://127.0.0.1:5000/ in your browser.

overview: |
  Most research uses deep learning (CNNs) for image classification.
  I wanted to see how far I could go with classical machine learning techniques.

  With careful data preprocessing, feature engineering, and insights from over 20 research papers, I achieved:
  ✅ 76% accuracy (previous benchmark was 67%)
  ✅ Ability to recognize multiple items in one image (e.g. paper + metal)
  ✅ A fully working web app deployed with Flask on Render

results: |
  Benchmark exceeded: 76% > 67%
  Generalization: Works on mixed waste images
  Robustness: Fixed data leakage issues and optimized preprocessing

  Example prediction (mixed objects):
    Metal → 71%
    Paper → 62%
    Plastic → 5%
    Glass → 2%
    ✔ Correctly identifies both Metal + Paper

tech_stack: |
  Language: Python
  Libraries: scikit-learn, scikit-image, NumPy, Pillow, Flask
  Dataset: TrashNet and similar Kaggle datasets
  Deployment: Flask web app hosted on Render

repository_structure: |
  Trash-image-classifier/
    ├── app.py                   # Flask web app
    ├── Recycling_project.joblib # Saved ML model
    ├── requirements.txt         # Dependencies
    ├── Procfile                 # Render deploy config
    ├── runtime.txt              # Python version
    ├── README.md                # Project documentation
    ├── examples/                # Example images + screenshots
    ├── training/                # Training scripts
    └── LICENSE

screenshots: |
  Web App Interface:
    ![Web App Interface](examples/interface.png)

  Example Prediction:
    ![Prediction Example](examples/prediction.png)

  (Save your screenshots in the examples/ folder with these filenames so they render properly.)

methodology: |
  Data Preprocessing:
    Images converted to grayscale, resized to 64×64, and normalized.

  Feature Engineering:
    HOG (edges/orientation), LBP (textures), and GLCM (spatial properties like contrast, homogeneity, correlation).
    Final feature vector is a combination of HOG + LBP + GLCM.

  Model Training:
    Tried classical algorithms such as SVM, Random Forest, and Logistic Regression.
    Selected the best model and saved it with joblib.

  Evaluation:
    Achieved 76% accuracy compared to the 67% benchmark.
    Validated on images containing multiple waste items.

impact: |
  Waste management is a global problem.
  This project shows that even without deep learning, AI can support automated recycling systems — making sorting of waste more efficient.

acknowledgments: |
  Inspired by over 20 research papers on trash classification.
  Dataset: Kaggle TrashNet.
  Thanks to the open-source community for tools and support.

license: |
  This project is licensed under the MIT License — free to use and modify.
