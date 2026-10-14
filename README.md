# 📱 Activity Recognition Using Cell Phones

A machine learning mini project that predicts human physical activities (Walking, Sitting, Standing, etc.) from smartphone accelerometer and gyroscope data.

---

## 🎯 Objective

Build and compare two ML classifiers — **Logistic Regression** and **Random Forest** — on the UCI Human Activity Recognition dataset, and serve predictions through a beginner-friendly Streamlit web dashboard.

---

## 📦 Dataset

**UCI Human Activity Recognition Using Smartphones**
- Link: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- 30 volunteers, Samsung Galaxy S II, waist-mounted
- 561 pre-extracted statistical features per 2.56-second sensor window
- 7,352 training samples / 2,947 test samples
- 6 activity classes

---

## 📂 Project Structure

```
ml/
├── app.py                  ← Streamlit web dashboard (5 pages)
├── train_model.py          ← Model training + evaluation pipeline
├── download_dataset.py     ← Auto-downloads UCI HAR dataset
├── model.pkl               ← Saved best model (created after training)
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
├── data/
│   └── UCI HAR Dataset/    ← Dataset files (created by downloader)
│       ├── train/
│       │   ├── X_train.txt
│       │   └── y_train.txt
│       ├── test/
│       │   ├── X_test.txt
│       │   └── y_test.txt
│       ├── features.txt
│       └── activity_labels.txt
└── outputs/                ← Generated plots and metrics (after training)
    ├── metrics.csv
    ├── model_comparison.png
    ├── confusion_matrix.png
    ├── activity_distribution.png
    └── classification_report.txt
```

---

## 🚀 Setup & Execution

### Step 1 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Download the dataset

```bash
python download_dataset.py
```

This automatically downloads and extracts the UCI HAR dataset into `data/UCI HAR Dataset/`.

> **Manual alternative:** Download the ZIP from the UCI page, extract it, and place the `UCI HAR Dataset` folder inside the `data/` directory.

### Step 3 — Train the models

```bash
python train_model.py
```

This will:
- Load and explore the dataset
- Train Logistic Regression and Random Forest
- Print evaluation metrics for both models
- Save plots to `outputs/`
- Save the best model to `model.pkl`

### Step 4 — Launch the web dashboard

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📊 Expected Results

| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | ~96% | ~96% |
| Random Forest | ~92-93% | ~92-93% |

> Actual results are computed on the test set and displayed in the dashboard.

---

## 🧠 Algorithm Summary

### Logistic Regression
- Generalisation of binary logistic regression to multi-class problems.
- Uses the **softmax function** to output a probability distribution over 6 classes.
- Solver: `lbfgs` (Limited-memory BFGS — efficient for multi-class problems).
- Simple, fast, interpretable — strong baseline on high-quality features.

### Random Forest
- An **ensemble of decision trees** trained on random feature subsets (bagging).
- Each tree votes; the majority class wins.
- More robust to overfitting than a single tree, but slower to train.
- 100 trees used in this project.

---

## 🌐 Dashboard Pages

| Page | Description |
|---|---|
| 🏠 Home | Project overview, workflow, dataset info |
| 🔮 Activity Prediction | Upload CSV or try a test-set sample for live prediction |
| 📊 Model Performance | Accuracy scores, comparison chart, confusion matrix, classification report |
| 🗃️ Dataset Explorer | Activity distribution, feature stats, sample rows |
| ℹ️ About | Technologies, methodology, limitations, references |

---

## 🎓 Likely Viva Questions

1. What is the difference between Logistic Regression and Random Forest?
2. Why did you use the UCI HAR dataset?
3. What are the 561 features? How were they extracted?
4. What is a confusion matrix and what does it tell you?
5. Why is accuracy alone insufficient? What are precision and recall?
6. What is overfitting? How does Random Forest reduce it?
7. Why did you use a separate test set instead of training accuracy?
8. What is the softmax function? Why is it used in multi-class classification?
9. What are the limitations of your model?
10. How would you improve this project further?

---

## 📚 References

- Anguita et al. (2013). *A Public Domain Dataset for Human Activity Recognition Using Smartphones.* ESANN.
- UCI ML Repository: https://archive.ics.uci.edu/dataset/240
- Scikit-learn documentation: https://scikit-learn.org
- Streamlit documentation: https://docs.streamlit.io
