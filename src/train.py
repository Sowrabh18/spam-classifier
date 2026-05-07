# ─────────────────────────────────────────────────────
# src/train.py
#
# Trains a spam classifier with hyperparameter tuning.
# Every experiment is tracked with MLflow.
# ─────────────────────────────────────────────────────

import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from src.preprocess import preprocess

# ── Constants ──────────────────────────────────────────
PROCESSED_DATA_PATH = 'data/processed/spam_cleaned.csv'
MODEL_PATH = 'models/spam_classifier.pkl'
MLFLOW_EXPERIMENT_NAME = 'spam-classifier'

# Create models folder if it doesn't exist
os.makedirs('models', exist_ok=True)



# ── Load data ───────────────────────────────────────────
def load_data():
    """
    Loads processed data.
    Always runs preprocessing to ensure
    data is fresh and clean.
    """

    print("Running preprocessing to ensure clean data...")
    preprocess()

    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Extra safety check — convert text column to string
    # and drop any remaining NaN values
    df['text'] = df['text'].astype(str)
    df = df[df['text'].str.strip() != '']
    df = df[df['text'] != 'nan']

    print(f"Loaded {len(df)} rows of processed data")
    return df

# ── Prepare features ────────────────────────────────────
def prepare_features(df):
    """
    Splits data into train and test sets.

    We have two types of features from preprocessing:
    - text column → goes through TF-IDF inside the pipeline
    - length column → numeric feature, used directly

    For simplicity in this project we use text only.
    Length will be added as an enhancement in future iterations.
    """

    X = df['text']
    y = df['label']

    # Split into 80% training, 20% testing
    # random_state=42 means the split is always the same
    # so results are reproducible
    # stratify=y means the split maintains the same
    # spam/ham ratio in both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Training set: {len(X_train)} emails")
    print(f"Test set:     {len(X_test)} emails")

    return X_train, X_test, y_train, y_test


# ── Train and evaluate function ─────────────────────────
def train_model(X_train, X_test, y_train, y_test, model_name, model, params):
    """
    Trains one model with hyperparameter tuning.
    Logs everything to MLflow.

    Parameters:
    - model_name: string name for MLflow logging
    - model: the sklearn model object
    - params: dictionary of hyperparameters to try
    """

    print(f"\n{'='*50}")
    print(f"Training: {model_name}")
    print(f"{'='*50}")

    # ── Start MLflow run ───────────────────────────────
    # Every block of code inside 'with mlflow.start_run()'
    # gets recorded as one experiment run in MLflow
    # run_name gives it a readable label in the MLflow UI
    with mlflow.start_run(run_name=model_name):

        # ── Build pipeline ─────────────────────────────
        # A Pipeline chains steps together in order
        # Step 1: TfidfVectorizer converts text to numbers
        # Step 2: the model learns from those numbers
        # Treating them as one object means we can tune
        # both TF-IDF settings and model settings together
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', model)
        ])

        # ── Hyperparameter tuning with GridSearchCV ────
        # GridSearchCV tries every combination of parameters
        # and finds which combination gives the best F1 score
        # cv=5 means 5-fold cross validation
        # meaning each combination is tested 5 times on
        # different splits of the training data
        # scoring='f1' means we optimise for F1 not accuracy
        # because our data is imbalanced
        grid_search = GridSearchCV(
            pipeline,
            params,
            cv=5,
            scoring='f1',
            n_jobs=-1,        # use all CPU cores to speed up
            verbose=1
        )

        # ── Fit the model ──────────────────────────────
        print("Running hyperparameter search...")
        grid_search.fit(X_train, y_train)

        # ── Get best model ─────────────────────────────
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        print(f"Best parameters: {best_params}")

        # ── Evaluate on test set ───────────────────────
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"\nResults:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"\nDetailed report:")
        print(classification_report(y_test, y_pred,
              target_names=['ham', 'spam']))

        # ── Log everything to MLflow ───────────────────
        # Log parameters — what settings did we use?
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("best_params", str(best_params))

        # Log metrics — how well did it perform?
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Log the actual model — save it inside MLflow
        mlflow.sklearn.log_model(best_model, "model")

        # Return results for comparison
        return {
            'model_name': model_name,
            'model': best_model,
            'f1_score': f1,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'params': best_params
        }
    

# ── Main function ────────────────────────────────────────
def main():
    """
    Runs the full training pipeline:
    1. Load data
    2. Prepare features
    3. Train multiple models with hyperparameter tuning
    4. Compare results
    5. Save the best model
    """

    # ── Set up MLflow experiment ───────────────────────
    # An experiment is a named collection of runs
    # All our training runs go under 'spam-classifier'
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # ── Load and prepare data ──────────────────────────
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_features(df)

    # ── Define models and hyperparameters to try ───────
    # Each entry has:
    # - a name for display and MLflow logging
    # - the model object
    # - the hyperparameters to search through
    #
    # Note the parameter names use double underscore __
    # because they're inside a Pipeline
    # 'tfidf__max_features' means the max_features
    # parameter of the tfidf step inside the pipeline
    # 'classifier__C' means the C parameter of
    # the classifier step inside the pipeline
    experiments = [
        {
            'name': 'Logistic Regression',
            'model': LogisticRegression(
                class_weight='balanced',  # handles class imbalance
                max_iter=1000,
                random_state=42
            ),
            'params': {
                'tfidf__max_features': [3000, 5000],
                'tfidf__ngram_range': [(1,1), (1,2)],
                'classifier__C': [0.1, 1, 10]
            }
        },
        {
            'name': 'Naive Bayes',
            'model': MultinomialNB(),
            'params': {
                'tfidf__max_features': [3000, 5000],
                'tfidf__ngram_range': [(1,1), (1,2)],
                'classifier__alpha': [0.1, 0.5, 1.0]
            }
        },
        {
            'name': 'Random Forest',
            'model': RandomForestClassifier(
                class_weight='balanced',  # handles class imbalance
                random_state=42,
                n_jobs=-1
            ),
            'params': {
                'tfidf__max_features': [3000, 5000],
                'tfidf__ngram_range': [(1,1), (1,2)],
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 10]
            }
        }
    ]

    # ── Train all models ───────────────────────────────
    results = []
    for exp in experiments:
        result = train_model(
            X_train, X_test, y_train, y_test,
            exp['name'], exp['model'], exp['params']
        )
        results.append(result)

    # ── Compare all results ────────────────────────────
    print("\n" + "="*50)
    print("RESULTS COMPARISON")
    print("="*50)
    print(f"{'Model':<25} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Accuracy':>10}")
    print("-"*65)
    for r in results:
        print(
            f"{r['model_name']:<25}"
            f"{r['f1_score']:>8.4f}"
            f"{r['precision']:>10.4f}"
            f"{r['recall']:>8.4f}"
            f"{r['accuracy']:>10.4f}"
        )

    # ── Pick the best model ────────────────────────────
    # Best = highest F1 score
    best = max(results, key=lambda x: x['f1_score'])

    print(f"\nBest model: {best['model_name']}")
    print(f"Best F1:    {best['f1_score']:.4f}")

    # ── Save the best model ────────────────────────────
    joblib.dump(best['model'], MODEL_PATH)
    print(f"\nBest model saved to: {MODEL_PATH}")

    return best


# ── Entry point ──────────────────────────────────────────
if __name__ == '__main__':
    main()