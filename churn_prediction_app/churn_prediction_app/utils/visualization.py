#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization utilities for the churn prediction application.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Set style
sns.set(style="whitegrid")


def create_feature_importance_plot(feature_importance_df, output_path=None, top_n=15):
    """
    Create feature importance plot.
    
    Args:
        feature_importance_df (DataFrame): DataFrame with 'Feature' and 'Importance' columns
        output_path (str, optional): Path to save the plot
        top_n (int, optional): Number of top features to display
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Sort and get top N features
    sorted_df = feature_importance_df.sort_values('Importance', ascending=False).head(top_n)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create horizontal bar plot
    ax = sns.barplot(x='Importance', y='Feature', data=sorted_df)
    
    # Add labels and title
    plt.title(f'Top {top_n} Feature Importance', fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {output_path}")
    
    # Return figure
    return plt.gcf()


def create_evaluation_plots(y_true, y_pred, y_pred_proba, output_dir):
    """
    Create model evaluation plots.
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        y_pred_proba (array): Predicted probabilities
        output_dir (str): Directory to save plots
        
    Returns:
        dict: Dictionary of figure objects
    """
    figures = {}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()
    
    # Save confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {cm_path}")
    figures['confusion_matrix'] = plt.gcf()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic', fontsize=16)
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    # Save ROC curve
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {roc_path}")
    figures['roc_curve'] = plt.gcf()
    
    return figures