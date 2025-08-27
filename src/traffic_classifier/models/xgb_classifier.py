"""XGBoost-based Multi-UE Traffic Classifier."""

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib
from typing import Dict, List, Any, Optional, Tuple
import time

class MultiUETrafficClassifier:
    """XGBoost classifier for multi-UE traffic classification."""

    def __init__(self, **xgb_params):
        """Initialize classifier with XGBoost parameters.

        Args:
            **xgb_params: XGBoost parameters
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss'
        }

        # Update with provided parameters
        default_params.update(xgb_params)

        self.model = xgb.XGBClassifier(**default_params)
        self.label_encoder = LabelEncoder()
        self.classes = [
            'gaming', 'audio_calls', 'video_calls', 'video_streaming',
            'browsing', 'video_uploads', 'texting'
        ]
        self.feature_names = []
        self.is_trained = False
        self.calibrator = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
              class_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Train the XGBoost model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            class_weights: Class weights for imbalanced data

        Returns:
            Training results dictionary
        """
        print(f"Training on {len(X_train)} samples with {X_train.shape[1]} features")

        # Store feature names
        self.feature_names = list(X_train.columns)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_train)

        # Apply class weights if provided
        sample_weights = None
        if class_weights:
            sample_weights = np.array([class_weights.get(cls, 1.0) for cls in y_train])
            print(f"Applied class weights: {class_weights}")

        # Setup evaluation set
        eval_set = None
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            eval_set = [(X_val, y_val_encoded)]

        # Train model
        start_time = time.time()

        self.model.fit(X_train, y_encoded, sample_weight=sample_weights, eval_set=eval_set, verbose=False)

        training_time = time.time() - start_time

        # Generate training results
        results = {
            'training_time_seconds': training_time,
            'n_features': X_train.shape[1],
            'n_samples': len(X_train),
            'feature_importance': self._get_feature_importance()
        }

        # Cross-validation if no validation set provided
        if X_val is None:
            cv_scores = self._cross_validate(X_train, y_encoded)
            results['cv_scores'] = cv_scores

        self.is_trained = True
        print(f"Training completed in {training_time:.2f} seconds")

        return results

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict traffic classes.

        Args:
            X: Feature DataFrame

        Returns:
            Predicted class labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        predictions_encoded = self.model.predict(X)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Predict class probabilities.

        Args:
            X: Feature DataFrame

        Returns:
            Tuple of (probabilities array, class names)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        if self.calibrator:
            probabilities = self.calibrator.predict_proba(X)
        else:
            probabilities = self.model.predict_proba(X)

        class_names = self.label_encoder.classes_.tolist()

        return probabilities, class_names

    def predict_with_confidence(self, X: pd.DataFrame, 
                              confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """Predict with confidence scores and fallback logic.

        Args:
            X: Feature DataFrame
            confidence_threshold: Minimum confidence for primary prediction

        Returns:
            Dictionary with predictions and confidence information
        """
        probabilities, class_names = self.predict_proba(X)

        results = {
            'predictions': [],
            'confidences': [],
            'top2_predictions': [],
            'abstain_flags': []
        }

        for prob_row in probabilities:
            # Get top 2 predictions
            top2_indices = np.argsort(prob_row)[-2:][::-1]
            top_pred = class_names[top2_indices[0]]
            second_pred = class_names[top2_indices[1]]
            top_conf = prob_row[top2_indices[0]]

            # Determine final prediction
            if top_conf >= confidence_threshold:
                final_pred = top_pred
                abstain = False
            else:
                final_pred = 'unknown'
                abstain = True

            results['predictions'].append(final_pred)
            results['confidences'].append(float(top_conf))
            results['top2_predictions'].append([top_pred, second_pred])
            results['abstain_flags'].append(abstain)

        return results

    def calibrate_probabilities(self, X_val: pd.DataFrame, y_val: pd.Series, 
                              method: str = 'platt') -> 'MultiUETrafficClassifier':
        """Calibrate probability predictions.

        Args:
            X_val: Validation features
            y_val: Validation labels
            method: Calibration method ('platt' or 'isotonic')

        Returns:
            Self for method chaining
        """
        from sklearn.calibration import CalibratedClassifierCV

        print(f"Calibrating probabilities using {method} method...")

        # Create calibrated classifier
        self.calibrator = CalibratedClassifierCV(
            self.model, 
            method=method, 
            cv='prefit'  # Use prefit since model is already trained
        )

        # Fit calibrator
        y_val_encoded = self.label_encoder.transform(y_val)
        self.calibrator.fit(X_val, y_val_encoded)

        print("Probability calibration complete")
        return self

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Evaluation metrics dictionary
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        # Predictions
        y_pred = self.predict(X_test)
        probabilities, class_names = self.predict_proba(X_test)

        # Basic metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None, labels=self.classes, zero_division=0
        )

        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        micro_f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)

        # Per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(self.classes):
            if i < len(precision):
                per_class_metrics[class_name] = {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1_score': float(f1[i]),
                    'support': int(support[i]) if i < len(support) else 0
                }

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=self.classes)

        # Top-2 accuracy
        top2_accuracy = self._calculate_top2_accuracy(y_test, probabilities, class_names)

        evaluation_results = {
            'accuracy': float(accuracy),
            'macro_f1': float(macro_f1),
            'micro_f1': float(micro_f1),
            'top2_accuracy': float(top2_accuracy),
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm.tolist(),
            'class_names': self.classes
        }

        return evaluation_results

    def _cross_validate(self, X: pd.DataFrame, y: np.ndarray, cv: int = 5) -> Dict[str, Any]:
        """Perform cross-validation.

        Args:
            X: Features
            y: Encoded labels
            cv: Number of CV folds

        Returns:
            Cross-validation results
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        cv_scores = cross_val_score(self.model, X, y, cv=skf, scoring='f1_macro')

        return {
            'mean_f1': float(np.mean(cv_scores)),
            'std_f1': float(np.std(cv_scores)),
            'scores': cv_scores.tolist()
        }

    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        importance_scores = self.model.feature_importances_

        feature_importance = {}
        for i, feature_name in enumerate(self.feature_names):
            if i < len(importance_scores):
                feature_importance[feature_name] = float(importance_scores[i])

        # Sort by importance
        sorted_importance = dict(sorted(feature_importance.items(), 
                                      key=lambda x: x[1], reverse=True))

        return sorted_importance

    def _calculate_top2_accuracy(self, y_true: pd.Series, probabilities: np.ndarray, 
                                class_names: List[str]) -> float:
        """Calculate top-2 accuracy.

        Args:
            y_true: True labels
            probabilities: Prediction probabilities
            class_names: Class names

        Returns:
            Top-2 accuracy score
        """
        correct_top2 = 0

        for i, true_label in enumerate(y_true):
            if i < len(probabilities):
                top2_indices = np.argsort(probabilities[i])[-2:]
                top2_predictions = [class_names[idx] for idx in top2_indices]

                if true_label in top2_predictions:
                    correct_top2 += 1

        return correct_top2 / len(y_true) if len(y_true) > 0 else 0.0

    def save(self, filepath: str) -> None:
        """Save trained model to file.

        Args:
            filepath: Output file path
        """
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'classes': self.classes,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'calibrator': self.calibrator
        }

        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'MultiUETrafficClassifier':
        """Load trained model from file.

        Args:
            filepath: Model file path

        Returns:
            Loaded classifier instance
        """
        model_data = joblib.load(filepath)

        classifier = cls()
        classifier.model = model_data['model']
        classifier.label_encoder = model_data['label_encoder']
        classifier.classes = model_data['classes']
        classifier.feature_names = model_data['feature_names']
        classifier.is_trained = model_data['is_trained']
        classifier.calibrator = model_data.get('calibrator')

        return classifier
