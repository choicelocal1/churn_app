#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module handles loading models and making predictions with explanations.
"""

import os
import joblib
import pandas as pd
import numpy as np


class ChurnPredictor:
    """Class to load the model and make predictions with explanations"""
    
    def __init__(self, model_path):
        """
        Initialize the predictor.
        
        Args:
            model_path (str): Path to the saved model package
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.model_package = joblib.load(model_path)
        self.preprocessor = self.model_package['preprocessor']
        self.model = self.model_package['model']
        self.feature_names = self.model_package['feature_names']
        
        print(f"Loaded {self.model_package.get('model_type', 'Unknown')} model created on {self.model_package.get('created_date', 'Unknown')}")
    
    def predict(self, deal_data):
        """
        Make predictions for new deal data.
        
        Args:
            deal_data (DataFrame): DataFrame containing deal data
            
        Returns:
            dict: Dictionary with prediction results
        """
        # Ensure deal_data has the expected features
        missing_features = set(self.feature_names) - set(deal_data.columns)
        if missing_features:
            print(f"Warning: Missing features in input data: {missing_features}")
            # For each missing feature, add it with NaN values
            for feature in missing_features:
                deal_data[feature] = np.nan
        
        # Select only the features used by the model
        X = deal_data[self.feature_names].copy()
        
        # Apply preprocessing
        X_processed = self.preprocessor.transform(X)
        
        # Make predictions
        y_pred = self.model.predict(X_processed)
        y_pred_proba = self.model.predict_proba(X_processed) if hasattr(self.model, 'predict_proba') else np.array([y_pred, 1-y_pred]).T
        
        # Get probability of churning (class 1)
        churn_probs = y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba
        
        # Calculate feature contributions for churn predictions
        explanations = self._explain_predictions(X, y_pred)
        
        # Create result DataFrame
        result = pd.DataFrame({
            'deal_id': deal_data['id'] if 'id' in deal_data.columns else range(len(deal_data)),
            'deal_name': deal_data['dealname'] if 'dealname' in deal_data.columns else None,
            'churn_probability': churn_probs,
            'churn_prediction': y_pred,
            'explanation': explanations
        })
        
        # Add other useful columns if they exist
        for col in ['dealstage', 'amount', 'createdate', 'closedate']:
            if col in deal_data.columns:
                result[col] = deal_data[col]
        
        return result
    
    def _explain_predictions(self, X, predictions):
        """
        Generate explanations for each prediction.
        
        Args:
            X (DataFrame): Input features
            predictions (array): Model predictions
            
        Returns:
            list: List of explanation strings
        """
        explanations = []
        
        # Get feature importances from the model
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For models without feature importances (e.g. logistic regression)
            importances = np.abs(self.model.coef_[0])
        else:
            # Default equal importances if model doesn't provide them
            importances = np.ones(len(self.feature_names))
        
        # Create dictionary of feature names to importances
        importance_dict = dict(zip(self.feature_names, importances))
        
        # For each prediction, generate an explanation
        for i, pred in enumerate(predictions):
            if pred == 1:  # Only explain churn predictions
                row = X.iloc[i]
                
                # Get the top 3 most important features for this prediction
                row_values = {feat: row[feat] for feat in self.feature_names}
                
                # Sort features by importance and select top 3
                top_features = sorted(
                    [(feat, importance_dict[feat]) for feat in self.feature_names],
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                
                # Create explanation text
                explanation_parts = []
                for feat, imp in top_features:
                    if pd.notna(row[feat]):
                        feat_value = row[feat]
                        # Format the value based on type
                        if isinstance(feat_value, (int, float)):
                            formatted_value = f"{feat_value:,.2f}" if feat_value >= 10 else f"{feat_value:.2f}"
                        else:
                            formatted_value = str(feat_value)
                            
                        explanation_parts.append(f"{feat}={formatted_value}")
                
                explanation = "Top factors: " + ", ".join(explanation_parts)
            else:
                explanation = "No churn predicted"
                
            explanations.append(explanation)
            
        return explanations