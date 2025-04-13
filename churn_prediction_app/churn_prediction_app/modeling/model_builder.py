#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module handles building and evaluating churn prediction models.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# XGBoost support removed
XGBOOST_AVAILABLE = False

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

from config.config import Config


class ChurnModelBuilder:
    """Class to build and evaluate churn prediction models"""
    
    def __init__(self, X, y):
        """
        Initialize the model builder.
        
        Args:
            X (DataFrame): Feature DataFrame
            y (Series): Target variable
        """
        self.X = X
        self.y = y
        self.best_model = None
        self.best_score = 0
        self.feature_importances = None
        
    def create_train_test_split(self, test_size=None, time_based=True):
        """
        Create train/test split, with option for time-based split.
        
        Args:
            test_size (float, optional): Proportion of data for testing
            time_based (bool): Whether to use time-based split
            
        Returns:
            self: For method chaining
        """
        test_size = test_size or Config.TEST_SIZE
        
        if time_based and 'tenure_days' in self.X.columns:
            # Sort by tenure for time-based split
            indices = np.argsort(self.X['tenure_days'].values)
            train_idx = indices[:int(len(indices) * (1 - test_size))]
            test_idx = indices[int(len(indices) * (1 - test_size)):]
            
            self.X_train = self.X.iloc[train_idx]
            self.X_test = self.X.iloc[test_idx]
            self.y_train = self.y.iloc[train_idx]
            self.y_test = self.y.iloc[test_idx]
            
            print(f"Time-based split created. Training on {len(self.X_train)} samples, testing on {len(self.X_test)} samples.")
        else:
            # Regular random split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=Config.RANDOM_SEED
            )
            print(f"Random split created. Training on {len(self.X_train)} samples, testing on {len(self.X_test)} samples.")
        
        return self
    
    def create_preprocessing_pipeline(self):
        """
        Create preprocessing pipeline with appropriate transformers.
        
        Returns:
            self: For method chaining
        """
        # Identify column types
        numeric_features = self.X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = self.X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create preprocessing steps for different column types
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Fit the preprocessor
        self.preprocessor.fit(self.X_train)
        
        # Transform the data
        self.X_train_processed = self.preprocessor.transform(self.X_train)
        self.X_test_processed = self.preprocessor.transform(self.X_test)
        
        print(f"Preprocessing pipeline created and applied.")
        print(f"Processed training data shape: {self.X_train_processed.shape}")
        
        return self
    
    def train_models(self, model_type='random_forest'):
        """
        Train a model of the specified type.
        
        Args:
            model_type (str): Type of model to train ('random_forest', 'logistic_regression')
            
        Returns:
            self: For method chaining
        """
        # If xgboost was requested but is unavailable, switch to random_forest
        if model_type == 'xgboost':
            print("XGBoost is not available in this installation. Using Random Forest instead.")
            model_type = 'random_forest'
            
        print(f"Training {model_type} model...")
        
        # Initialize the appropriate model type
        if model_type == 'logistic_regression':
            self.best_model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight='balanced',
                random_state=Config.RANDOM_SEED
            )
        else:  # default to random_forest
            self.best_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=Config.RANDOM_SEED
            )
        
        # Train the model
        self.best_model.fit(self.X_train_processed, self.y_train)
        
        # Make predictions
        y_pred = self.best_model.predict(self.X_test_processed)
        
        # Check if we have multiple classes
        unique_classes = len(np.unique(self.y_train))
        print(f"Number of unique classes in training data: {unique_classes}")
        
        if unique_classes > 1:
            # Normal case - multiple classes
            if hasattr(self.best_model, 'predict_proba'):
                y_pred_proba = self.best_model.predict_proba(self.X_test_processed)[:, 1]
            else:
                # For models that don't have predict_proba
                y_pred_proba = y_pred.astype(float)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            # Print metrics
            print(f"Model evaluation:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
        else:
            # Only one class - can't calculate most metrics
            print("Warning: Only one class present in the training data")
            print("This usually means all samples are predicted as not churned")
            accuracy = accuracy_score(self.y_test, y_pred)
            print(f"Model evaluation:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Other metrics not available with only one class")
            
            # Create dummy probabilities (all 0 since that's the only class)
            y_pred_proba = np.zeros(len(y_pred))
        
        # Store best score (accuracy if only one class)
        self.best_score = accuracy if unique_classes <= 1 else f1_score(self.y_test, y_pred)
        
        return self
    
    def analyze_feature_importance(self):
        """
        Analyze feature importance from the model.
        
        Returns:
            self: For method chaining
        """
        if not self.best_model:
            print("No model trained yet. Run train_models first.")
            return self
        
        # Get feature names
        feature_names = []
        
        # Get numeric feature names directly
        numeric_features = self.X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        feature_names.extend(numeric_features)
        
        # Get one-hot encoded categorical feature names
        categorical_features = self.X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_features and hasattr(self.preprocessor, 'named_transformers_') and 'cat' in self.preprocessor.named_transformers_:
            cat_transformer = self.preprocessor.named_transformers_['cat']
            if hasattr(cat_transformer, 'named_steps') and 'onehot' in cat_transformer.named_steps:
                encoder = cat_transformer.named_steps['onehot']
                if hasattr(encoder, 'categories_'):
                    for i, cat_feature in enumerate(categorical_features):
                        for category in encoder.categories_[i]:
                            feature_names.append(f"{cat_feature}_{category}")
        
        # For different model types, get importances
        if hasattr(self.best_model, 'feature_importances_'):
            # For tree-based models (Random Forest, XGBoost)
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            # For linear models (Logistic Regression)
            importances = np.abs(self.best_model.coef_[0])
        else:
            print(f"Feature importance not available for this model type")
            return self
        
        # Adjust feature names if needed
        if len(feature_names) != len(importances):
            print(f"Warning: Feature name count ({len(feature_names)}) doesn't match importance count ({len(importances)})")
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
        # Create feature importance dataframe
        self.feature_importances = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Display top features
        print("\nTop 5 most important features:")
        print(self.feature_importances.head(5))
        
        return self
    
    def save_model(self, filepath='models/churn_model_latest.joblib'):
        """
        Save the model and preprocessor to disk.
        
        Args:
            filepath (str): Path to save the model
            
        Returns:
            self: For method chaining
        """
        if not self.best_model:
            print("No model trained yet. Run train_models first.")
            return self
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Create model package with both preprocessor and model
        model_type = type(self.best_model).__name__
        model_package = {
            'preprocessor': self.preprocessor,
            'model': self.best_model,
            'feature_names': self.X.columns.tolist(),
            'model_type': model_type,
            'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'best_score': self.best_score
        }
        
        # Save to disk
        joblib.dump(model_package, filepath)
        print(f"Model saved to {filepath}")
        
        return self
    
    def build(self, model_type='random_forest'):
        """
        Run full model building pipeline with specified model type.
        
        Args:
            model_type (str): Type of model to train
            
        Returns:
            self: For method chaining
        """
        return (self
                .create_train_test_split(time_based=True)
                .create_preprocessing_pipeline()
                .train_models(model_type=model_type)
                .analyze_feature_importance()
                .save_model())