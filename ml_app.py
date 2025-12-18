# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from io import BytesIO

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

st.set_page_config(layout="wide", page_title="End-to-End ML with Streamlit")

st.title("End-to-End ML Pipeline — (Demo: Titanic)")

# ------------------------
# 1. DATA INPUT
# ------------------------
st.sidebar.header("1) Dataset selection")
dataset_source = st.sidebar.selectbox(
    "Choose dataset source",
    ("Seaborn built-in", "Upload CSV")
)

if dataset_source == "Seaborn built-in":
    sns_datasets = ["titanic", "penguins", "iris", "tips"]  # common choices
    sns_choice = st.sidebar.selectbox("Seaborn dataset", sns_datasets)
    @st.cache_data
    def load_sns(name):
        return sns.load_dataset(name)
    df = load_sns(sns_choice)
    st.sidebar.write(f"Loaded seaborn dataset: **{sns_choice}**")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    df = None
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.sidebar.write("Uploaded CSV loaded.")

if df is None:
    st.info("Upload a CSV or select a seaborn dataset to continue.")
    st.stop()

st.write("### Dataset preview")
st.dataframe(df.head().style.format(na_rep="NA"))

# ------------------------
# 2. TARGET & FEATURES
# ------------------------
st.sidebar.header("2) Target & features")
all_columns = df.columns.tolist()
target_col = st.sidebar.selectbox("Select target column (the label)", all_columns)

# Auto-suggest features (all except target)
default_features = [c for c in all_columns if c != target_col]
features = st.sidebar.multiselect("Select features (leave blank = auto pick numeric+categorical)", default_features, default_features)

if not features:
    st.warning("No features selected — using all columns except target.")
    features = default_features

# If target is non-numeric, try encode later
st.sidebar.markdown("**Quick checks:**")
st.sidebar.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
st.sidebar.write(f"Target dtype: {df[target_col].dtype}")

# ------------------------
# 3. Preprocessing options
# ------------------------
st.sidebar.header("3) Preprocessing & model options")
test_size = st.sidebar.slider("Test set fraction", 0.1, 0.5, 0.25, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)
apply_feature_selection = st.sidebar.checkbox("Apply feature selection (SelectFromModel)", value=True)
do_hyperparam_tuning = st.sidebar.checkbox("Hyperparameter tuning (GridSearchCV)", value=True)
n_jobs = st.sidebar.number_input("n_jobs for training", min_value=1, max_value=16, value=4)

# ------------------------
# Helpers
# ------------------------
def get_column_types(df, features):
    num_cols = df[features].select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df[features].select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    # Some numeric-coded columns may be int but actually categorical; keep it simple for now
    return num_cols, cat_cols

# ------------------------
# 4. Prepare data
# ------------------------
X = df[features].copy()
y = df[target_col].copy()

# If target is categorical, encode to numeric
if y.dtype == "object" or y.dtype.name == "category" or y.dtype == "bool":
    y_encoded, y_uniques = pd.factorize(y)
    y = y_encoded
    label_mapping = {i: val for i, val in enumerate(y_uniques)}
else:
    # If numeric but not binary classification, still handle general classification
    label_mapping = None

num_cols, cat_cols = get_column_types(df, features)
st.write(f"Detected numeric columns: {num_cols}")
st.write(f"Detected categorical columns: {cat_cols}")

# ------------------------
# 5. BUILD PIPELINE
# ------------------------
# Numeric pipeline
num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical pipeline
cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
], remainder="drop")

base_clf = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)

# Optionally insert feature selection after model fitting (we'll wrap with SelectFromModel)
# Build outer pipeline (preprocessing -> (feature selection) -> classifier)
steps = [("preprocessor", preprocessor)]
if apply_feature_selection:
    # SelectFromModel requires an estimator; we'll use a RandomForest within feature selection
    selector = ("selector", SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=random_state), threshold="median"))
    steps.append(selector)
steps.append(("clf", base_clf))
pipeline = Pipeline(steps=steps)

# ------------------------
# 6. Train / Tune
# ------------------------
if st.button("Train model"):
    with st.spinner("Splitting data and training..."):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y))>1 else None)

        if do_hyperparam_tuning:
            st.info("Running GridSearchCV — this may take time depending on compute.")
            param_grid = {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 6, 12],
                "clf__min_samples_split": [2, 5]
            }
            # If feature selection is used, grid still applies to classifier named 'clf'
            grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=n_jobs, scoring="f1", verbose=0)
            grid.fit(X_train, y_train)
            best = grid.best_estimator_
            st.success(f"GridSearch done. Best params: {grid.best_params_}")
            model = best
        else:
            pipeline.fit(X_train, y_train)
            model = pipeline

        # Predictions
        y_pred = model.predict(X_test)
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="binary" if len(np.unique(y))==2 else "macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="binary" if len(np.unique(y))==2 else "macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="binary" if len(np.unique(y))==2 else "macro", zero_division=0)

        st.subheader("Evaluation metrics")
        st.write(f"Accuracy: **{acc:.4f}** — Precision: **{prec:.4f}** — Recall: **{rec:.4f}** — F1: **{f1:.4f}**")
        st.write("Classification report:")
        st.text(classification_report(y_test, y_pred, zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

        # ROC AUC (only for binary)
        if y_proba is not None and len(np.unique(y)) == 2:
            auc = roc_auc_score(y_test, y_proba)
            st.write(f"ROC AUC: **{auc:.4f}**")
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
            ax_roc.plot([0, 1], [0, 1], linestyle="--")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.legend()
            st.pyplot(fig_roc)

        # Feature importances (if classifier supports it)
        st.subheader("Feature importance / selected features")
        try:
            # To get feature names after preprocessing, we need to build them
            ohe_cols = []
            if len(cat_cols) > 0:
                # fit OneHotEncoder to get actual categories
                tmp_ohe = OneHotEncoder(handle_unknown="ignore").fit(X[cat_cols].astype(str))
                for i, col in enumerate(cat_cols):
                    cats = tmp_ohe.categories_[i]
                    ohe_cols += [f"{col}__{c}" for c in cats]
            feature_names = num_cols + ohe_cols

            # If feature selection used, model.named_steps may contain selector
            clf_step = model.named_steps.get("clf", model) if isinstance(model, Pipeline) else model
            # Try to get importances from final estimator
            if hasattr(clf_step, "feature_importances_"):
                importances = clf_step.feature_importances_
                # If selector present, map selected features
                if apply_feature_selection and "selector" in model.named_steps:
                    mask = model.named_steps["selector"].get_support()
                    selected_feature_names = [f for f, m in zip(feature_names, mask) if m]
                    selected_importances = importances
                    df_imp = pd.DataFrame({"feature": selected_feature_names, "importance": selected_importances})
                else:
                    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
                df_imp = df_imp.sort_values("importance", ascending=False).head(30)
                st.dataframe(df_imp)
                fig_imp, ax_imp = plt.subplots(figsize=(6, min(6, len(df_imp)/2)))
                ax_imp.barh(df_imp["feature"][::-1], df_imp["importance"][::-1])
                st.pyplot(fig_imp)
            else:
                st.info("Feature importances not available for this final estimator.")
        except Exception as e:
            st.write("Could not compute feature importances:", e)

        # Save model for download
        buf = BytesIO()
        joblib.dump(model, buf)
        buf.seek(0)
        st.download_button("Download trained model (.joblib)", buf.read(), file_name="model.joblib", mime="application/octet-stream")

        st.success("Training & evaluation complete.")

# ------------------------
# 7. Single-record prediction UI
# ------------------------
st.sidebar.header("Prediction")
with st.sidebar.expander("Make a single prediction (manual input)"):
    st.write("Enter values for features (leave blank to use mean/mode).")
    input_vals = {}
    for col in features:
        if col in num_cols:
            val = st.number_input(f"{col} (numeric)", value=float(df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else 0.0))
            input_vals[col] = val
        else:
            unique_vals = df[col].dropna().unique().tolist()
            if len(unique_vals) <= 10:
                val = st.selectbox(f"{col} (categorical)", options=[str(x) for x in unique_vals])
            else:
                val = st.text_input(f"{col} (categorical free text)")
            input_vals[col] = val

    if st.button("Predict single record"):
        # Need a trained model available: we try to use last trained pipeline if possible
        try:
            model  # check existence
        except NameError:
            st.warning("Train a model first using the 'Train model' button.")
        else:
            single_df = pd.DataFrame([input_vals])
            # Align dtypes to original df where possible
            for c in single_df.columns:
                if c in df.columns:
                    try:
                        single_df[c] = single_df[c].astype(df[c].dtype)
                    except Exception:
                        pass
            pred = model.predict(single_df)[0]
            if label_mapping:
                pred_label = label_mapping.get(pred, pred)
            else:
                pred_label = pred
            st.write(f"Predicted label: **{pred_label}**")

# ------------------------
# 8. Notes & tips
# ------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("**Notes:**")
st.sidebar.markdown("""
- This app uses `ColumnTransformer` + pipelines to handle numeric & categorical preprocessing.  
- Feature selection uses `SelectFromModel` (RandomForest) when enabled.  
- Hyperparameter tuning uses `GridSearchCV` with a small grid — expand if you have time/compute.  
- For other datasets: ensure target is classification-friendly (not high-cardinality regression).  
""")
