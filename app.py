"""
app.py
------
Streamlit dashboard for the Activity Recognition project.

Run with:
    streamlit run app.py
"""

import os
import io
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Activity Recognition | Cell Phones",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
ACTIVITY_LABELS = [
    "Walking", "Walking Upstairs", "Walking Downstairs",
    "Sitting", "Standing", "Laying",
]
DATA_DIR = os.path.join("data", "UCI HAR Dataset")
OUTPUT_DIR = "outputs"
MODEL_PATH = "model.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: LOAD CACHED RESOURCES
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the saved model and feature names from disk."""
    if not os.path.exists(MODEL_PATH):
        return None, None
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["feature_names"]


@st.cache_data(show_spinner=False)
def load_metrics():
    """Load model evaluation metrics saved during training."""
    path = os.path.join(OUTPUT_DIR, "metrics.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_train_data():
    """Load training data for visualisations."""
    x_path = os.path.join(DATA_DIR, "train", "X_train.txt")
    y_path = os.path.join(DATA_DIR, "train", "y_train.txt")
    features_path = os.path.join(DATA_DIR, "features.txt")

    if not all(os.path.exists(p) for p in [x_path, y_path, features_path]):
        return None, None

    features_df = pd.read_csv(features_path, sep=r"\s+", header=None)
    names = features_df[1].tolist()
    seen = {}
    unique_names = []
    for name in names:
        if name in seen:
            seen[name] += 1
            unique_names.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            unique_names.append(name)

    X = pd.read_csv(x_path, sep=r"\s+", header=None)
    X.columns = unique_names
    y_raw = pd.read_csv(y_path, sep=r"\s+", header=None).squeeze()
    activity_map = {1: "Walking", 2: "Walking Upstairs", 3: "Walking Downstairs",
                    4: "Sitting", 5: "Standing", 6: "Laying"}
    y = y_raw.map(activity_map)
    return X, y


@st.cache_data(show_spinner=False)
def load_test_data():
    """Load test data for live prediction comparison."""
    x_path = os.path.join(DATA_DIR, "test", "X_test.txt")
    y_path = os.path.join(DATA_DIR, "test", "y_test.txt")
    features_path = os.path.join(DATA_DIR, "features.txt")

    if not all(os.path.exists(p) for p in [x_path, y_path, features_path]):
        return None, None

    features_df = pd.read_csv(features_path, sep=r"\s+", header=None)
    names = features_df[1].tolist()
    seen = {}
    unique_names = []
    for name in names:
        if name in seen:
            seen[name] += 1
            unique_names.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            unique_names.append(name)

    X = pd.read_csv(x_path, sep=r"\s+", header=None)
    X.columns = unique_names
    y_raw = pd.read_csv(y_path, sep=r"\s+", header=None).squeeze()
    activity_map = {1: "Walking", 2: "Walking Upstairs", 3: "Walking Downstairs",
                    4: "Sitting", 5: "Standing", 6: "Laying"}
    y = y_raw.map(activity_map)
    return X, y


def load_report():
    path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return None


def fig_to_bytes(fig):
    """Convert a matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/smartphone-tablet.png", width=72)
    st.title("📱 Activity Recognition")
    st.markdown("**Using Cell Phone Sensors**")
    st.divider()
    page = st.radio(
        "Navigate to",
        ["🏠 Home", "🔮 Activity Prediction", "📊 Model Performance",
         "🗃️ Dataset Explorer", "ℹ️ About"],
    )
    st.divider()
    model, feature_names = load_model()
    if model:
        st.success("✅ Model loaded")
    else:
        st.error("❌ Model not found\nRun `python train_model.py` first")

    st.caption("Mini Project — BTech CSE")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: HOME
# ─────────────────────────────────────────────────────────────────────────────

if page == "🏠 Home":
    st.title("📱 Activity Recognition Using Cell Phones")
    st.markdown("### Predicting human activities from smartphone motion sensor data")
    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ## 🎯 Project Objective
        This project builds a machine learning system that automatically identifies
        what physical activity a person is performing — just from the raw motion
        sensor data collected by their smartphone.

        Modern smartphones carry **accelerometers** and **gyroscopes** that
        continuously measure movement in three axes (X, Y, Z). By analyzing
        patterns in this data, machine learning models can classify activities
        like walking, sitting, or laying down with high accuracy.

        ---

        ## ⚙️ How It Works

        1. **Data Collection** — The UCI HAR dataset provides readings from 30
           volunteers wearing a Samsung Galaxy S II on their waist.
        2. **Feature Extraction** — 561 statistical features (mean, std, energy,
           correlation, etc.) are pre-extracted from raw sensor windows.
        3. **Model Training** — Two classifiers (Logistic Regression and
           Random Forest) are trained on the labelled dataset.
        4. **Prediction** — The better model predicts the activity for new inputs.
        """)

    with col2:
        st.markdown("### 🏃 Activities Detected")
        for emoji, label in zip(["🚶", "⬆️", "⬇️", "🪑", "🧍", "🛏️"], ACTIVITY_LABELS):
            st.markdown(f"- {emoji} **{label}**")

        st.divider()
        st.markdown("### 📦 Dataset Info")
        st.markdown("""
        - **Source**: UCI Machine Learning Repository
        - **Participants**: 30 volunteers
        - **Sensor**: Accelerometer + Gyroscope
        - **Train samples**: 7,352
        - **Test samples**: 2,947
        - **Features**: 561 per sample
        """)

    st.divider()
    st.markdown("## 🔁 System Workflow")
    st.markdown("""
    ```
    Smartphone Sensors
         │
         ▼
    Raw Accelerometer & Gyroscope Signals
         │
         ▼
    Feature Extraction (561 statistical features per 2.56-second window)
         │
         ▼
    Machine Learning Model (Logistic Regression / Random Forest)
         │
         ▼
    Predicted Activity Label (Walking, Sitting, etc.)
    ```
    """)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: ACTIVITY PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🔮 Activity Prediction":
    st.title("🔮 Activity Prediction")
    st.markdown("Predict the physical activity from sensor feature data.")

    if model is None:
        st.error("Model not loaded. Please run `python train_model.py` first.")
        st.stop()

    st.divider()
    tab1, tab2 = st.tabs(["📤 Upload CSV", "🔬 Try a Sample from Test Set"])

    # ── Tab 1: Upload CSV ──────────────────────────────────────────────────
    with tab1:
        st.markdown("""
        Upload a CSV file where each row is one sensor window containing
        exactly **561 feature columns** (same order as the UCI HAR dataset).
        """)

        uploaded = st.file_uploader("Choose a CSV file", type=["csv", "txt"])

        if uploaded:
            try:
                df_upload = pd.read_csv(uploaded, header=None)
                st.write(f"**Uploaded shape:** {df_upload.shape}")

                # Validate column count
                if df_upload.shape[1] != len(feature_names):
                    st.error(
                        f"❌ Expected {len(feature_names)} columns, "
                        f"got {df_upload.shape[1]}. "
                        "Please use a valid UCI HAR-compatible CSV."
                    )
                else:
                    df_upload.columns = feature_names
                    predictions = model.predict(df_upload)

                    results_df = pd.DataFrame({
                        "Row": range(1, len(predictions) + 1),
                        "Predicted Activity": predictions,
                    })
                    st.success(f"✅ Predicted {len(predictions)} sample(s)")
                    st.dataframe(results_df, use_container_width=True)

                    # Activity count summary
                    count_df = pd.Series(predictions).value_counts().reset_index()
                    count_df.columns = ["Activity", "Count"]
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar(count_df["Activity"], count_df["Count"],
                           color="steelblue", edgecolor="black")
                    ax.set_title("Prediction Summary")
                    ax.set_ylabel("Count")
                    plt.xticks(rotation=30, ha="right")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

            except Exception as e:
                st.error(f"Error reading file: {e}")

    # ── Tab 2: Sample from Test Set ────────────────────────────────────────
    with tab2:
        st.markdown("""
        Pick a random sample directly from the test set to see how the model performs.
        """)

        X_test, y_test = load_test_data()

        if X_test is None:
            st.warning("Test dataset not found. Please download the dataset first.")
        else:
            if st.button("🎲 Pick a Random Sample"):
                idx = np.random.randint(0, len(X_test))
                sample = X_test.iloc[[idx]]
                true_label = y_test.iloc[idx]
                pred_label = model.predict(sample)[0]

                col1, col2, col3 = st.columns(3)
                col1.metric("Sample Index", idx)
                col2.metric("True Activity", true_label)

                if pred_label == true_label:
                    col3.metric("Predicted Activity", pred_label, "✅ Correct")
                else:
                    col3.metric("Predicted Activity", pred_label, "❌ Incorrect", delta_color="inverse")

                st.divider()
                with st.expander("🔍 Show top 20 feature values for this sample"):
                    top_features = sample.iloc[0].abs().nlargest(20)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    top_features.plot(kind="barh", ax=ax, color="teal", edgecolor="black")
                    ax.set_title("Top 20 Features by Absolute Value")
                    ax.set_xlabel("Feature Value")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📊 Model Performance":
    st.title("📊 Model Performance")

    metrics_df = load_metrics()

    if metrics_df is None:
        st.error("Metrics not found. Please run `python train_model.py` first.")
        st.stop()

    st.divider()

    # ── Score Cards ────────────────────────────────────────────────────────
    st.markdown("### 🏆 Accuracy Scores")
    cols = st.columns(len(metrics_df))
    for col, (_, row) in zip(cols, metrics_df.iterrows()):
        badge = "⭐ Best Model" if row["Best"] else ""
        col.metric(
            label=f"{row['Model']} {badge}",
            value=f"{row['Accuracy']*100:.2f}%",
        )

    st.divider()

    # ── Comparison Table ───────────────────────────────────────────────────
    st.markdown("### 📋 Detailed Metrics")
    display_df = metrics_df[["Model", "Accuracy", "Precision", "Recall", "F1 Score"]].copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        display_df[col] = display_df[col].map(lambda v: f"{v*100:.2f}%")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Comparison Chart ───────────────────────────────────────────────────
    st.markdown("### 📈 Model Comparison Chart")
    img_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)
    else:
        st.info("Run `python train_model.py` to generate this chart.")

    st.divider()

    # ── Confusion Matrix ───────────────────────────────────────────────────
    st.markdown("### 🔷 Confusion Matrix")
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    if os.path.exists(cm_path):
        st.image(cm_path, use_container_width=True)
        st.caption(
            "Rows = True labels | Columns = Predicted labels | "
            "Diagonal = correctly classified samples"
        )
    else:
        st.info("Run `python train_model.py` to generate the confusion matrix.")

    st.divider()

    # ── Classification Report ──────────────────────────────────────────────
    st.markdown("### 📄 Classification Report")
    report = load_report()
    if report:
        st.code(report, language="text")
    else:
        st.info("Run `python train_model.py` to generate the classification report.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: DATASET EXPLORER
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🗃️ Dataset Explorer":
    st.title("🗃️ Dataset Explorer")
    st.markdown("Explore the UCI HAR dataset used for training.")

    X_train, y_train = load_train_data()

    if X_train is None:
        st.error("Dataset not found at `data/UCI HAR Dataset/`. "
                 "Please run `python download_dataset.py` first.")
        st.stop()

    st.divider()

    # ── Summary Stats ──────────────────────────────────────────────────────
    st.markdown("### 📌 Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Training Samples", f"{len(X_train):,}")
    col2.metric("Features per Sample", f"{X_train.shape[1]}")
    col3.metric("Activity Classes", "6")
    col4.metric("Missing Values", f"{X_train.isnull().sum().sum()}")

    st.divider()

    # ── Activity Distribution ─────────────────────────────────────────────
    st.markdown("### 📊 Activity Distribution (Training Set)")
    dist = y_train.value_counts().reset_index()
    dist.columns = ["Activity", "Count"]
    dist["Percentage"] = (dist["Count"] / dist["Count"].sum() * 100).round(1)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.dataframe(dist, use_container_width=True, hide_index=True)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = plt.cm.Set2(np.linspace(0, 1, len(dist)))
        ax.barh(dist["Activity"], dist["Count"], color=colors, edgecolor="black")
        ax.set_xlabel("Number of Samples")
        ax.set_title("Activity Distribution")
        for i, (val, pct) in enumerate(zip(dist["Count"], dist["Percentage"])):
            ax.text(val + 10, i, f"{pct}%", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()

    # ── Saved Distribution Plot ────────────────────────────────────────────
    dist_img = os.path.join(OUTPUT_DIR, "activity_distribution.png")
    if os.path.exists(dist_img):
        st.markdown("### 📊 Train vs Test Distribution")
        st.image(dist_img, use_container_width=True)

    st.divider()

    # ── Feature Statistics ────────────────────────────────────────────────
    st.markdown("### 🔢 Feature Statistics (first 10 features)")
    with st.expander("Show feature statistics"):
        st.dataframe(
            X_train.iloc[:, :10].describe().round(4),
            use_container_width=True,
        )

    st.divider()

    # ── Sample rows ───────────────────────────────────────────────────────
    st.markdown("### 🔎 Sample Data Rows")
    n_rows = st.slider("Number of rows to show", 5, 50, 10)
    display_cols = X_train.columns[:10].tolist()
    st.dataframe(
        X_train[display_cols].head(n_rows),
        use_container_width=True,
    )
    st.caption(f"Showing first 10 of 561 features. Labels: {y_train.head(n_rows).tolist()}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5: ABOUT
# ─────────────────────────────────────────────────────────────────────────────

elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ## 📖 Project Overview

        **Title:** Activity Recognition Using Cell Phones

        **Course:** Machine Learning Mini Project

        **Dataset:** UCI Human Activity Recognition Using Smartphones
        (Anguita et al., 2013)

        **Objective:** Train ML models to classify six physical activities
        using 561 pre-extracted features from smartphone accelerometer and
        gyroscope signals.

        ---

        ## 🛠️ Technologies Used

        | Technology | Purpose |
        |---|---|
        | Python 3.x | Core language |
        | Pandas / NumPy | Data manipulation |
        | Scikit-learn | ML models & evaluation |
        | Matplotlib / Seaborn | Visualisations |
        | Streamlit | Web dashboard |
        | Joblib | Model serialisation |

        ---

        ## 🔬 Methodology

        1. **Data Loading** — Read X_train, y_train, X_test, y_test from the
           UCI HAR text files.
        2. **Preprocessing** — Rename 561 features, map integer labels to
           activity names, verify no missing values.
        3. **Model Training** — Train Logistic Regression (multi-class,
           `lbfgs` solver) and Random Forest (100 trees) on training data.
        4. **Evaluation** — Compute accuracy, precision, recall, F1 on the
           held-out test set. Never use test data during training.
        5. **Selection** — The model with higher accuracy is chosen as the
           best model and saved to `model.pkl`.
        6. **Deployment** — Streamlit app loads the model for live predictions.
        """)

    with col2:
        st.markdown("""
        ## ⚠️ Limitations

        - The 561 features are **pre-extracted** by the dataset authors;
          we don't process raw sensor signals.
        - The model is trained on data from only **30 participants** — it
          may not generalise to all populations, device placements, or
          phone models.
        - Only **six fixed activities** are recognised; any other activity
          would be misclassified.
        - No real-time sensor streaming is implemented; predictions rely
          on pre-processed CSV input.
        - Deep learning models (LSTM, CNN) would likely achieve higher
          accuracy on raw signals.

        ---

        ## 📚 References

        1. Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz,
           J.L. (2013). *A Public Domain Dataset for Human Activity
           Recognition Using Smartphones.* ESANN.
        2. UCI ML Repository: Human Activity Recognition Using Smartphones.
           https://archive.ics.uci.edu/dataset/240
        3. Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1).
        4. Bishop, C.M. (2006). *Pattern Recognition and Machine Learning.*
           Springer.

        ---

        ## 🗂️ Project Files

        ```
        ml/
        ├── app.py                  ← Streamlit dashboard
        ├── train_model.py          ← Training pipeline
        ├── download_dataset.py     ← Dataset downloader
        ├── model.pkl               ← Saved best model
        ├── requirements.txt        ← Dependencies
        ├── README.md               ← Setup instructions
        ├── data/
        │   └── UCI HAR Dataset/    ← Dataset files
        └── outputs/
            ├── metrics.csv
            ├── model_comparison.png
            ├── confusion_matrix.png
            ├── activity_distribution.png
            └── classification_report.txt
        ```
        """)
