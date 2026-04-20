from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent

sns.set_theme(style="whitegrid")


TRAINING_PROFILES = {
    "Quick demo": {
        "sample_size": 50_000,
        "cv_folds": 3,
        "tune_random_forest": False,
        "description": "Fastest option for classroom demos and UI checks.",
    },
    "Balanced demo": {
        "sample_size": 100_000,
        "cv_folds": 3,
        "tune_random_forest": False,
        "description": "Good balance between speed and reasonably stable metrics.",
    },
    "Proper training": {
        "sample_size": None,
        "cv_folds": 5,
        "tune_random_forest": True,
        "description": "Uses the full dataset, stronger cross-validation, and tuned Random Forest.",
    },
}


def find_dataset_path() -> Path | None:
    candidates = [
        BASE_DIR / "creditcard.csv",
        BASE_DIR / "creditcard .csv",
        Path.cwd() / "creditcard.csv",
        Path.cwd() / "creditcard .csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def load_dataset(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


def take_stratified_sample(
    df: pd.DataFrame, sample_size: int | None, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(df):
        return df.copy()

    sampled_index, _ = train_test_split(
        df.index,
        train_size=sample_size,
        stratify=df["Class"],
        random_state=random_state,
    )
    return df.loc[sampled_index].copy().reset_index(drop=True)


def build_pipelines(smote_strategy: float) -> dict[str, ImbPipeline]:
    return {
        "Logistic Regression": ImbPipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
                (
                    "smote",
                    SMOTE(
                        random_state=RANDOM_STATE,
                        sampling_strategy=smote_strategy,
                    ),
                ),
                (
                    "model",
                    LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Random Forest": ImbPipeline(
            [
                (
                    "smote",
                    SMOTE(
                        random_state=RANDOM_STATE,
                        sampling_strategy=smote_strategy,
                    ),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=220,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        class_weight="balanced_subsample",
                    ),
                ),
            ]
        ),
        "Gradient Boosting": ImbPipeline(
            [
                (
                    "smote",
                    SMOTE(
                        random_state=RANDOM_STATE,
                        sampling_strategy=smote_strategy,
                    ),
                ),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=100,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Support Vector Machine": ImbPipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
                (
                    "smote",
                    SMOTE(
                        random_state=RANDOM_STATE,
                        sampling_strategy=smote_strategy,
                    ),
                ),
                ("model", LinearSVC(random_state=RANDOM_STATE, max_iter=5000)),
            ]
        ),
    }


def get_model_scores(pipeline: ImbPipeline, features: pd.DataFrame) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(features)[:, 1]
    if hasattr(pipeline, "decision_function"):
        return pipeline.decision_function(features)
    raise AttributeError("The selected model does not expose scores for ROC-AUC.")


def evaluate_model(
    model_name: str,
    pipeline: ImbPipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, float | str], np.ndarray, np.ndarray]:
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    scores = get_model_scores(pipeline, x_test)

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, scores),
    }
    return metrics, predictions, scores


@st.cache_resource(show_spinner=False)
def run_experiment(
    path_str: str,
    sample_size: int | None,
    smote_strategy: float,
    cv_folds: int,
    tune_random_forest: bool,
) -> dict:
    df = load_dataset(path_str)
    working_df = take_stratified_sample(df, sample_size)

    x = working_df.drop(columns="Class")
    y = working_df["Class"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    pipelines = build_pipelines(smote_strategy)

    cv_rows = []
    for model_name, pipeline in pipelines.items():
        scores = cross_validate(
            pipeline,
            x_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        cv_rows.append(
            {
                "Model": model_name,
                "CV Accuracy": scores["test_accuracy"].mean(),
                "CV Precision": scores["test_precision"].mean(),
                "CV Recall": scores["test_recall"].mean(),
                "CV F1": scores["test_f1"].mean(),
                "CV ROC-AUC": scores["test_roc_auc"].mean(),
            }
        )

    tuning_summary = None
    if tune_random_forest:
        param_grid = {
            "model__n_estimators": [220, 320],
            "model__max_depth": [None, 16, 24],
            "model__min_samples_leaf": [1, 2],
            "model__min_samples_split": [2, 5],
        }
        search = GridSearchCV(
            estimator=pipelines["Random Forest"],
            param_grid=param_grid,
            scoring="recall",
            cv=cv,
            n_jobs=-1,
        )
        search.fit(x_train, y_train)
        pipelines["Random Forest"] = search.best_estimator_
        tuning_summary = {
            "best_params": search.best_params_,
            "best_cv_recall": search.best_score_,
        }

    comparison_rows = []
    model_outputs = {}
    for model_name, pipeline in pipelines.items():
        metrics, predictions, scores = evaluate_model(
            model_name,
            pipeline,
            x_train,
            x_test,
            y_train,
            y_test,
        )
        comparison_rows.append(metrics)
        model_outputs[model_name] = {
            "pipeline": pipeline,
            "predictions": predictions,
            "scores": scores,
            "confusion_matrix": confusion_matrix(y_test, predictions),
        }

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        by=["Recall", "ROC-AUC"], ascending=False
    )
    cv_df = pd.DataFrame(cv_rows).sort_values(by="CV Recall", ascending=False)

    rf_model = model_outputs["Random Forest"]["pipeline"].named_steps["model"]
    feature_importance_df = pd.DataFrame(
        {
            "Feature": x_train.columns,
            "Importance": rf_model.feature_importances_,
        }
    ).sort_values(by="Importance", ascending=False)

    test_samples = x_test.copy()
    test_samples["Actual Class"] = y_test.values
    test_samples = test_samples.reset_index(drop=True)

    return {
        "working_df": working_df,
        "comparison_df": comparison_df.reset_index(drop=True),
        "cv_df": cv_df.reset_index(drop=True),
        "model_outputs": model_outputs,
        "feature_importance_df": feature_importance_df.reset_index(drop=True),
        "test_samples": test_samples,
        "y_test": y_test.reset_index(drop=True),
        "tuning_summary": tuning_summary,
        "dataset_summary": {
            "rows_used": len(working_df),
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "fraud_cases_used": int(working_df["Class"].sum()),
            "cv_folds": cv_folds,
        },
    }


def plot_class_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    class_percent = df["Class"].value_counts(normalize=True).sort_index() * 100

    sns.countplot(x="Class", data=df, ax=axes[0], palette="Set2")
    axes[0].set_title("Class Distribution")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Count")

    class_percent.plot(kind="bar", ax=axes[1], color=["#4C72B0", "#DD8452"])
    axes[1].set_title("Class Percentage")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].tick_params(axis="x", rotation=0)

    fig.tight_layout()
    return fig


def plot_confusion_matrices(results: dict) -> plt.Figure:
    num_models = len(results["model_outputs"])
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))
    for axis, (model_name, output) in zip(axes, results["model_outputs"].items()):
        sns.heatmap(
            output["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axis,
        )
        axis.set_title(f"{model_name}")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")

    fig.tight_layout()
    return fig


def plot_roc_curves(results: dict) -> plt.Figure:
    fig, axis = plt.subplots(figsize=(8, 6))
    y_test = results["y_test"]

    for model_name, output in results["model_outputs"].items():
        fpr, tpr, _ = roc_curve(y_test, output["scores"])
        auc_score = roc_auc_score(y_test, output["scores"])
        axis.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {auc_score:.4f})")

    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random Guess")
    axis.set_title("ROC Curve Comparison")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend()
    fig.tight_layout()
    return fig


def plot_feature_importance(feature_importance_df: pd.DataFrame) -> plt.Figure:
    top_features = feature_importance_df.head(10)
    fig, axis = plt.subplots(figsize=(9, 5))
    sns.barplot(data=top_features, x="Importance", y="Feature", palette="viridis", ax=axis)
    axis.set_title("Top 10 Feature Importances - Random Forest")
    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")
    fig.tight_layout()
    return fig


def render_prediction_lab(results: dict) -> None:
    st.subheader("Transaction Prediction Demo")
    st.write(
        "Choose a transaction from the test set and score it with the trained Random Forest model."
    )

    test_samples = results["test_samples"]
    sample_index = st.slider(
        "Select a test transaction",
        min_value=0,
        max_value=len(test_samples) - 1,
        value=0,
    )

    selected_row = test_samples.iloc[sample_index]
    actual_class = int(selected_row["Actual Class"])
    feature_row = selected_row.drop(labels=["Actual Class"]).to_frame().T

    st.dataframe(feature_row, use_container_width=True)

    if st.button("Predict Fraud Risk", use_container_width=True):
        rf_pipeline = results["model_outputs"]["Random Forest"]["pipeline"]
        prediction = int(rf_pipeline.predict(feature_row)[0])
        probability = float(rf_pipeline.predict_proba(feature_row)[0, 1])

        left, right = st.columns(2)
        left.metric("Predicted Class", "Fraud" if prediction == 1 else "Legitimate")
        right.metric("Fraud Probability", f"{probability:.2%}")

        st.caption(f"Actual class for this test example: {'Fraud' if actual_class == 1 else 'Legitimate'}")

        if prediction == 1:
            st.error("The model flagged this transaction as potentially fraudulent.")
        else:
            st.success("The model classified this transaction as legitimate.")


def main() -> None:
    st.set_page_config(
        page_title="Credit Card Fraud Detection",
        page_icon="💳",
        layout="wide",
    )

    st.title("💳 Credit Card Fraud Detection Dashboard")
    st.write(
        "This Streamlit app converts the fraud detection notebook into an interactive demo with "
        "dataset analysis, model comparison, evaluation metrics, and a live prediction screen."
    )

    dataset_path = find_dataset_path()
    if dataset_path is None:
        st.error(
            "Dataset not found. Add `creditcard.csv` or `creditcard .csv` to the project folder and rerun the app."
        )
        st.stop()

    dataset = load_dataset(str(dataset_path))

    st.sidebar.header("Run Settings")
    st.sidebar.caption(f"Dataset: `{dataset_path.name}`")

    training_profile = st.sidebar.selectbox(
        "Training profile",
        list(TRAINING_PROFILES.keys()),
        index=2,
    )
    profile_config = TRAINING_PROFILES[training_profile]
    st.sidebar.caption(profile_config["description"])

    smote_strategy = st.sidebar.slider(
        "SMOTE sampling strategy",
        min_value=0.10,
        max_value=1.00,
        value=0.25,
        step=0.05,
        help="0.25 means minority samples are generated until they reach 25% of the majority class.",
    )
    cv_folds = st.sidebar.slider(
        "Cross-validation folds",
        min_value=3,
        max_value=5,
        value=profile_config["cv_folds"],
    )
    tune_random_forest = st.sidebar.checkbox(
        "Tune Random Forest with GridSearchCV",
        value=profile_config["tune_random_forest"],
    )

    sample_size = profile_config["sample_size"]
    rows_label = f"{sample_size:,}" if sample_size is not None else "All rows"
    st.sidebar.write(f"Rows used for training: **{rows_label}**")

    st.subheader("Project Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Transactions", f"{len(dataset):,}")
    col2.metric("Fraud Cases", f"{int(dataset['Class'].sum()):,}")
    col3.metric("Fraud Rate", f"{dataset['Class'].mean() * 100:.3f}%")

    st.info(
        "Recall matters most in fraud detection because false negatives mean real fraud transactions were missed."
    )
    st.pyplot(plot_class_distribution(dataset))

    if st.button("Train Models and Build Dashboard", type="primary", use_container_width=True):
        with st.spinner("Training models, applying SMOTE, and computing evaluation metrics..."):
            results = run_experiment(
                str(dataset_path),
                sample_size,
                smote_strategy,
                cv_folds,
                tune_random_forest,
            )
        st.session_state["results"] = results
        st.session_state["settings"] = {
            "training_profile": training_profile,
            "smote_strategy": smote_strategy,
            "cv_folds": cv_folds,
            "tune_random_forest": tune_random_forest,
        }

    results = st.session_state.get("results")
    if results is None:
        st.warning("Click the training button to generate model results and enable the prediction demo.")
        return

    st.subheader("Run Summary")
    st.write(
        f"Profile: **{st.session_state['settings']['training_profile']}** | "
        f"SMOTE strategy: **{st.session_state['settings']['smote_strategy']:.2f}** | "
        f"CV folds: **{st.session_state['settings']['cv_folds']}** | "
        f"Random Forest tuning: **{'On' if st.session_state['settings']['tune_random_forest'] else 'Off'}**"
    )

    summary = results["dataset_summary"]
    sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
    sum_col1.metric("Rows Used", f"{summary['rows_used']:,}")
    sum_col2.metric("Train Rows", f"{summary['train_rows']:,}")
    sum_col3.metric("Test Rows", f"{summary['test_rows']:,}")
    sum_col4.metric("Fraud Cases Used", f"{summary['fraud_cases_used']:,}")

    if results["tuning_summary"] is not None:
        st.success(
            f"Best tuned Random Forest recall: {results['tuning_summary']['best_cv_recall']:.4f}"
        )
        st.json(results["tuning_summary"]["best_params"])

    best_row = results["comparison_df"].iloc[0]
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Best Model", str(best_row["Model"]))
    metric2.metric("Best Recall", f"{best_row['Recall']:.4f}")
    metric3.metric("Best ROC-AUC", f"{best_row['ROC-AUC']:.4f}")

    st.info(
        "For final submission, use the `Proper training` profile so the dashboard trains on the full dataset with 5-fold cross-validation and tuned Random Forest."
    )

    st.subheader("Model Comparison Table")
    st.dataframe(
        results["comparison_df"].style.format(
            {
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1-score": "{:.4f}",
                "ROC-AUC": "{:.4f}",
            }
        ),
        use_container_width=True,
    )

    st.subheader("Cross-Validation Summary")
    st.dataframe(
        results["cv_df"].style.format(
            {
                "CV Accuracy": "{:.4f}",
                "CV Precision": "{:.4f}",
                "CV Recall": "{:.4f}",
                "CV F1": "{:.4f}",
                "CV ROC-AUC": "{:.4f}",
            }
        ),
        use_container_width=True,
    )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Confusion Matrices")
        st.pyplot(plot_confusion_matrices(results))
    with chart_col2:
        st.subheader("ROC Curve")
        st.pyplot(plot_roc_curves(results))

    st.subheader("Random Forest Feature Importance")
    st.dataframe(results["feature_importance_df"].head(10), use_container_width=True)
    st.pyplot(plot_feature_importance(results["feature_importance_df"]))

    render_prediction_lab(results)


if __name__ == "__main__":
    main()
