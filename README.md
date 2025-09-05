# 🗑️ Trash Classifier — Classical Machine Learning Approach

> **Can classical machine learning compete with deep learning in image classification?**  
I set out to prove that it can. This project is a **trash classification system** (plastic, paper, glass, metal) built entirely with **classical ML techniques** — no deep learning.

---

## 🚀 Project Overview
Most research uses **deep learning** for image classification, but I wanted to challenge myself by applying **classical machine learning**.  

With careful **data preprocessing**, **feature engineering**, and insights from **20+ research papers**, I achieved:  
- ✅ **76% accuracy** (previous reported benchmark: 67%)  
- ✅ Model can recognize images containing **multiple items** (e.g., a paper + metal object in one photo)  
- ✅ Fully deployed as a **Flask web app**  

---

## 📊 Results
- **Benchmark exceeded**: 76% > 67%  
- **Generalization**: Handles mixed waste images effectively  
- **Robustness**: Debugged data leakage issues and optimized feature extraction  

Example prediction (mixed objects):  
```
Metal → 71%  
Paper → 62%  
Plastic → 5%  
Glass → 2%  
```
✔ Correctly identifies both **Metal + Paper**  

---

## ⚙️ Tech Stack
- **Language:** Python  
- **Libraries:** scikit-learn, OpenCV, NumPy, Pandas, Matplotlib, Flask  
- **Dataset:** Kaggle (TrashNet & similar datasets)  
- **Deployment:** Flask web application  

---

## 📂 Repository Structure
```
📦 trash-classifier
 ┣ 📂 data/                # Training & testing images (not included in repo)
 ┣ 📂 examples/            # Example input images + predictions
 ┣ 📂 notebooks/           # Jupyter notebooks for training & experiments
 ┣ 📂 app/                 # Flask web app
 ┃ ┣ app.py
 ┃ ┣ static/
 ┃ ┗ templates/
 ┣ requirements.txt        # Dependencies
 ┣ README.md               # Project documentation
 ┗ LICENSE
```

---

## 🖼️ Screenshots

### Web App Interface
![Web App Screenshot](examples/Screenshot%20%281678%29.png)

### Prediction Example
![Prediction Example](examples/Screenshot%20%281679%29.png)

---

## 💡 Key Learnings
- Classical ML is still powerful with the **right data + preprocessing**  
- Handling **shapes, patterns, edges** was critical for higher accuracy  
- Fixed **data leakage** and learned to debug systematically  
- Persistence > doubt: *If I can do it, you can do it too.*  

---

## 🔗 Try It Yourself
1. Clone this repo:  
   ```bash
   git clone https://github.com/<your-username>/trash-classifier.git
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask app:  
   ```bash
   python app/app.py
   ```
4. Open in browser:  
   ```
   http://127.0.0.1:5000/
   ```

---

## 🌍 Real-World Impact
Waste management is a global problem.  
This project shows how **AI (even classical ML)** can help in **automated recycling systems** — detecting and sorting waste more efficiently.  

---

## 🙌 Acknowledgments
- Inspired by **20+ research papers** on trash classification  
- Dataset: [Kaggle TrashNet](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)  
- Support from the open-source community 💙  

---

## 🏷️ License
This project is licensed under the MIT License — free to use and modify.  
