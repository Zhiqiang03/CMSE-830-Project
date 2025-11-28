"""
Models for Taxi Tip Classification
This module contains various machine learning models to predict tip_class (low, middle, high)
"""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
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
import pandas as pd
from typing import Dict, Tuple, Any
import time


class TipClassificationModels:
    """
    A class to train and evaluate multiple classification models for tip prediction
    """

    def __init__(self, data_path: str = 'data/processed_taxi_data.pt'):
        """
        Initialize the model trainer with preprocessed data

        Args:
            data_path: Path to the processed data file
        """
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.load_data()

    def load_data(self):
        """Load the preprocessed data"""
        print(f"Loading data from {self.data_path}...")
        data = torch.load(self.data_path)

        # Convert to numpy arrays for sklearn
        self.X_train = data['X_train'].numpy()
        self.y_train = data['y_train'].numpy()
        self.X_test = data['X_test'].numpy()
        self.y_test = data['y_test'].numpy()

        print(f"Data loaded successfully!")
        print(f"Training set: {self.X_train.shape[0]} samples, {self.X_train.shape[1]} features")
        print(f"Test set: {self.X_test.shape[0]} samples")
        print(f"Classes: {np.unique(self.y_train)} (0=low, 1=middle, 2=high)")
        print(f"Class distribution: {np.bincount(self.y_train.astype(int))}")

    def initialize_models(self):
        """Initialize all classification models"""
        print("\nInitializing models...")

        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=20,
                random_state=42
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            ),
            'Naive Bayes': GaussianNB(),
            'SVM': SVC(
                kernel='rbf',
                random_state=42,
                max_iter=1000
            )
        }

        print(f"Initialized {len(self.models)} models")

    def train_model(self, model_name: str, model: Any) -> Dict:
        """
        Train a single model and evaluate it

        Args:
            model_name: Name of the model
            model: The model instance

        Returns:
            Dictionary containing model performance metrics
        """
        print(f"\n{'='*60}")
        print(f"Training {model_name}...")
        print(f"{'='*60}")

        # Train
        start_time = time.time()
        model.fit(self.X_train, self.y_train)
        train_time = time.time() - start_time

        # Predict
        start_time = time.time()
        y_pred = model.predict(self.X_test)
        predict_time = time.time() - start_time

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        # Get confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)

        # Get classification report
        report = classification_report(
            self.y_test,
            y_pred,
            target_names=['Low', 'Middle', 'High']
        )

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

    def train_all_models(self):
        """Train all models and store results"""
        self.initialize_models()

        for model_name, model in self.models.items():
            try:
                result = self.train_model(model_name, model)
                self.results[model_name] = result
            except Exception as e:
                print(f"\nError training {model_name}: {str(e)}")
                continue

        self.display_summary()

    def display_summary(self):
        """Display summary of all model results"""
        print(f"\n{'='*80}")
        print("MODEL PERFORMANCE SUMMARY")
        print(f"{'='*80}")

        # Create summary dataframe
        summary_data = []
        for model_name, result in self.results.items():
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
        best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n{'='*80}")
        print(f"BEST MODEL: {best_model[0]} with accuracy {best_model[1]['accuracy']:.4f}")
        print(f"{'='*80}")

    def get_best_model(self) -> Tuple[str, Any, Dict]:
        """
        Get the best performing model

        Returns:
            Tuple of (model_name, model_instance, results_dict)
        """
        if not self.results:
            raise ValueError("No models have been trained yet. Call train_all_models() first.")

        best_model_name = max(self.results.items(), key=lambda x: x[1]['accuracy'])[0]
        best_result = self.results[best_model_name]

        return best_model_name, best_result['model'], best_result

    def save_best_model(self, filename: str = 'best_model.pkl'):
        """
        Save the best model to a file

        Args:
            filename: Name of the file to save the model
        """
        import pickle

        model_name, model, results = self.get_best_model()

        with open(filename, 'wb') as f:
            pickle.dump({
                'model_name': model_name,
                'model': model,
                'results': results
            }, f)

        print(f"\nBest model ({model_name}) saved to {filename}")

    def predict(self, X_new, model_name: str = None):
        """
        Make predictions with a trained model

        Args:
            X_new: New data to predict
            model_name: Name of the model to use (uses best model if None)

        Returns:
            Predictions
        """
        if model_name is None:
            model_name, model, _ = self.get_best_model()
        else:
            if model_name not in self.results:
                raise ValueError(f"Model {model_name} not found. Available models: {list(self.results.keys())}")
            model = self.results[model_name]['model']

        predictions = model.predict(X_new)
        return predictions


def main():
    """Main function to run the model training pipeline"""
    print("Starting Taxi Tip Classification Model Training")
    print("="*80)

    # Initialize and train models
    trainer = TipClassificationModels(data_path='data/processed_taxi_data.pt')
    trainer.train_all_models()

    # Save best model
    trainer.save_best_model('best_tip_classifier.pkl')

    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)


if __name__ == "__main__":
    main()

