import torch
import numpy as np
import pandas as pd
import pickle
import time
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 1. Load Data
data_path = 'data/processed_taxi_data.pt'
data = torch.load(data_path)

X_train = data['X_train'].numpy()
y_train = data['y_train'].numpy()
X_test = data['X_test'].numpy()
y_test = data['y_test'].numpy()

print(f"Data loaded successfully!")
print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Classes: {np.unique(y_train)} (0=low, 1=middle, 2=high)")
print(f"Class distribution: {np.bincount(y_train.astype(int))}")

# 2. Define Models
models = {
    # LOGISTIC REGRESSION: Increased max_iter to ensure convergence
    'Logistic Regression': LogisticRegression(
        class_weight='balanced',
        max_iter=5000,  # Increased from 1000
        random_state=42,
        n_jobs=-1
    ),

    # RANDOM FOREST
    'Random Forest': RandomForestClassifier(
        class_weight='balanced',
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    ),

    # HIST GRADIENT BOOSTING: Faster & Better for large tabular data
    'Hist Gradient Boosting': HistGradientBoostingClassifier(
        class_weight='balanced',
        max_iter=100,
        max_depth=10,
        random_state=42,
        early_stopping=True
    ),

    # DECISION TREE
    'Decision Tree': DecisionTreeClassifier(
        class_weight='balanced',
        max_depth=20,
        random_state=42
    ),

    # KNN: Kept as requested. WARNING: This will be very slow to predict.
    'K-Nearest Neighbors': KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1
    ),

    # NAIVE BAYES
    'Naive Bayes': GaussianNB(),

    # SVM (SGD): Replaces SVC. SVC will crash on 1.4M rows.
    # This acts as a linear SVM optimized for speed.
    'SVM (Linear SGD)': SGDClassifier(
        loss='hinge',  # 'hinge' loss makes it an SVM
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        max_iter=5000
    )
}


def train_and_evaluate_model(model_name, model, X_train, y_train, X_test, y_test):
    """Train a single model and evaluate it"""
    print(f"\n{'=' * 60}")
    print(f"Training {model_name}...")
    print(f"{'=' * 60}")

    # Train
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Predict
    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Low', 'Middle', 'High'], zero_division=0)

    print(f"\nTraining time: {train_time:.2f} seconds")
    print(f"Prediction time: {predict_time:.2f} seconds")
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"\nClassification Report:")
    print(report)
    print(f"\nConfusion Matrix:")
    print(cm)

    return {
        'model_name': model_name,
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'train_time': train_time,
        'predict_time': predict_time,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': y_pred
    }


def display_summary(results):
    """Display summary of all model results"""
    print(f"\n{'=' * 80}")
    print("MODEL PERFORMANCE SUMMARY")
    print(f"{'=' * 80}")

    summary_data = []
    for model_name, result in results.items():
        summary_data.append({
            'Model': model_name,
            'Accuracy': f"{result['accuracy']:.4f}",
            'Precision': f"{result['precision']:.4f}",
            'Recall': f"{result['recall']:.4f}",
            'F1-Score': f"{result['f1_score']:.4f}",
            'Train Time (s)': f"{result['train_time']:.2f}",
            'Predict Time (s)': f"{result['predict_time']:.2f}"
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Accuracy', ascending=False)
    print(summary_df.to_string(index=False))

    # Best model
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_accuracy = results[best_model_name]['accuracy']
    print(f"\n{'=' * 80}")
    print(f"BEST MODEL: {best_model_name} with accuracy {best_accuracy:.4f}")
    print(f"{'=' * 80}")

    return best_model_name


def save_model(model_name, model, results, filename='best_tip_classifier.pkl'):
    """Save a model to a file"""
    with open(filename, 'wb') as f:
        pickle.dump({
            'model_name': model_name,
            'model': model,
            'results': results
        }, f)
    print(f"\nModel ({model_name}) saved to {filename}")


# --- MAIN EXECUTION ---
results = {}
for model_name, model in models.items():
    try:
        result = train_and_evaluate_model(model_name, model, X_train, y_train, X_test, y_test)
        results[model_name] = result
    except Exception as e:
        print(f"\nError training {model_name}: {str(e)}")
        continue

if results:
    best_model_name = display_summary(results)

    # Save all models
    for model_name in results:
        # Create a clean filename
        clean_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()

        if model_name == best_model_name:
            save_model(model_name, results[model_name]['model'], results[model_name],
                       filename=f'./models/{clean_name}_best_tip_classifier.pkl')
        else:
            save_model(model_name, results[model_name]['model'], results[model_name],
                       filename=f"./models/{clean_name}_tip_classifier.pkl")