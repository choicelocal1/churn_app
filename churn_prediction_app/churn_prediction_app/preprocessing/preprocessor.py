#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module handles data preprocessing and feature engineering.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from config.config import Config


class ChurnDataPreprocessor:
    """Class to preprocess and engineer features for churn prediction"""
    
    def __init__(self, companies_df, deals_df=None, tickets_df=None, churn_definition=None):
        """
        Initialize the preprocessor.
        
        Args:
            companies_df (DataFrame): DataFrame containing company data
            deals_df (DataFrame, optional): DataFrame containing deal data
            tickets_df (DataFrame, optional): DataFrame containing ticket data
            churn_definition (str, optional): Definition of churn
        """
        self.companies_df = companies_df.copy() if companies_df is not None else None
        self.deals_df = deals_df.copy() if deals_df is not None else None
        self.tickets_df = tickets_df.copy() if tickets_df is not None else None
        self.churn_definition = churn_definition or Config.DEFAULT_CHURN_DEFINITION
        
    def calculate_tenure(self):
        """
        Calculate client tenure in days.
        
        Returns:
            self: For method chaining
        """
        if 'contract_start_date' not in self.companies_df.columns:
            print("Warning: 'contract_start_date' not found in companies data")
            self.companies_df['tenure_days'] = 0
            return self
        
        # Convert to datetime
        self.companies_df['contract_start_date'] = pd.to_datetime(
            self.companies_df['contract_start_date'], 
            errors='coerce'
        )
        
        # Calculate tenure based on current date
        current_date = datetime.now()
        self.companies_df['tenure_days'] = (
            current_date - self.companies_df['contract_start_date']
        ).dt.days
        
        # Handle missing or invalid values
        median_tenure = self.companies_df['tenure_days'].median()
        # Use non-inplace version to avoid pandas warning
        self.companies_df['tenure_days'] = self.companies_df['tenure_days'].fillna(median_tenure)
        
        # Handle negative values (future dates)
        self.companies_df.loc[self.companies_df['tenure_days'] < 0, 'tenure_days'] = 0
        
        return self
    
    def calculate_time_to_renewal(self):
        """
        Calculate days until contract renewal.
        
        Returns:
            self: For method chaining
        """
        if 'contract_end_date' not in self.companies_df.columns:
            print("Warning: 'contract_end_date' not found in companies data")
            self.companies_df['days_to_renewal'] = 365  # Default to 1 year
            return self
        
        # Convert to datetime
        self.companies_df['contract_end_date'] = pd.to_datetime(
            self.companies_df['contract_end_date'], 
            errors='coerce'
        )
        
        # Calculate days to renewal
        current_date = datetime.now()
        self.companies_df['days_to_renewal'] = (
            self.companies_df['contract_end_date'] - current_date
        ).dt.days
        
        # Handle missing or invalid values
        median_renewal = self.companies_df['days_to_renewal'].median()
        # Use non-inplace version to avoid pandas warning
        self.companies_df['days_to_renewal'] = self.companies_df['days_to_renewal'].fillna(median_renewal)
        
        return self
    
    def create_churn_label(self):
        """
        Create churn label based on defined churn definition.
        
        Returns:
            self: For method chaining
        """
        print(f"Creating churn label using definition: {self.churn_definition}")
        
        # Default - if we can't determine churn status, assume not churned
        self.companies_df['churned'] = 0
        
        # Check column existence for default definition
        if self.churn_definition == "client_status == 'churned'":
            if 'client_status' in self.companies_df.columns:
                self.companies_df['churned'] = (self.companies_df['client_status'] == 'churned').astype(int)
            else:
                print("Warning: 'client_status' field not found in data. You may need to specify a different churn definition.")
                print("Available fields:", self.companies_df.columns.tolist())
                
                # Try to find alternative fields
                if 'hs_lead_status' in self.companies_df.columns:
                    print("Using 'hs_lead_status' as alternative - assuming CLOSED means churned")
                    self.companies_df['churned'] = (self.companies_df['hs_lead_status'] == 'CLOSED').astype(int)
                elif 'lifecyclestage' in self.companies_df.columns:
                    print("Using 'lifecyclestage' as alternative - assuming values other than 'customer' means churned")
                    self.companies_df['churned'] = (self.companies_df['lifecyclestage'] != 'customer').astype(int)
        else:
            # For custom churn definitions, use safer evaluation
            try:
                # Replace field references with dataframe column references
                eval_expr = self.churn_definition
                for col in self.companies_df.columns:
                    if col in eval_expr:
                        eval_expr = eval_expr.replace(col, f"self.companies_df['{col}']")
                
                # Evaluate the expression
                print(f"Evaluating churn expression: {eval_expr}")
                churn_mask = eval(eval_expr)
                self.companies_df['churned'] = churn_mask.astype(int)
            except Exception as e:
                print(f"Error evaluating churn definition: {e}")
                print("Using default value of 0 (not churned) for all records")
        
        # Report churn rate
        churn_rate = self.companies_df['churned'].mean()
        print(f"Created churn label with {self.companies_df['churned'].sum()} churned companies ({churn_rate:.1%} churn rate)")
        
        return self
    
    def calculate_additional_features(self):
        """
        Calculate additional features that might be predictive of churn.
        
        Returns:
            self: For method chaining
        """
        # Recency of last meeting
        if 'last_meeting_date' in self.companies_df.columns:
            self.companies_df['last_meeting_date'] = pd.to_datetime(
                self.companies_df['last_meeting_date'],
                errors='coerce'
            )
            current_date = datetime.now()
            self.companies_df['days_since_last_meeting'] = (
                current_date - self.companies_df['last_meeting_date']
            ).dt.days
            
            # Fill missing values with a high number (indicating no meeting)
            # Use non-inplace version to avoid pandas warning
            self.companies_df['days_since_last_meeting'] = self.companies_df['days_since_last_meeting'].fillna(365)
        
        return self
    
    def enrich_from_deals(self):
        """
        Enrich company data with information from deals.
        
        Returns:
            self: For method chaining
        """
        if self.deals_df is None:
            print("No deals data available for enrichment")
            return self
        
        # Ensure we have a way to join deals to companies
        if 'company_id' not in self.deals_df.columns:
            print("Warning: 'company_id' not found in deals data, unable to join to companies")
            return self
        
        # Convert amount to numeric if it exists
        if 'amount' in self.deals_df.columns:
            self.deals_df['amount'] = pd.to_numeric(self.deals_df['amount'], errors='coerce')
        
        try:
            # Group deals by company and calculate metrics
            deal_metrics = {}
            
            # Calculate total deal value per company
            if 'amount' in self.deals_df.columns:
                deal_metrics['deal_value_total'] = self.deals_df.groupby('company_id')['amount'].sum()
            
            # Calculate deal count per company
            deal_metrics['deal_count'] = self.deals_df.groupby('company_id').size()
            
            # Create a DataFrame from the metrics
            if deal_metrics:
                metrics_df = pd.DataFrame(deal_metrics)
                metrics_df.reset_index(inplace=True)
                
                # Join to companies
                self.companies_df = self.companies_df.merge(
                    metrics_df,
                    left_on='id',
                    right_on='company_id',
                    how='left'
                )
                
                # Fill NaN values with 0 (companies with no deals)
                for col in metrics_df.columns:
                    if col != 'company_id' and col in self.companies_df.columns:
                        self.companies_df[col] = self.companies_df[col].fillna(0)
            
            print(f"Enriched company data with {len(deal_metrics)} deal metrics")
            
        except Exception as e:
            print(f"Error enriching from deals: {e}")
        
        return self
    
    def prepare_for_modeling(self):
        """
        Prepare the final dataset for modeling.
        
        Returns:
            tuple: (X, y) - Feature matrix and target vector
        """
        # Get available columns
        available_columns = self.companies_df.columns.tolist()
        print(f"Available columns for modeling: {available_columns}")
        
        # Preferred columns for modeling
        preferred_columns = [
            'tenure_days', 'days_to_renewal', 'numberofemployees', 'num_locations',
            'contract_value', 'industry', 'days_since_last_meeting', 'deal_value_total',
            'deal_count'
        ]
        
        # Select columns that exist in the data
        model_columns = [col for col in preferred_columns if col in available_columns]
        
        # If we have too few columns, try to use any numeric columns
        if len(model_columns) < 3:
            print("Warning: Too few preferred columns available. Adding additional numeric columns.")
            numeric_cols = self.companies_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            for col in numeric_cols:
                if col not in model_columns and col != 'churned' and col != 'id':
                    model_columns.append(col)
        
        print(f"Using {len(model_columns)} columns for modeling: {model_columns}")
        
        # Create feature dataframe
        X = self.companies_df[model_columns].copy()
        y = self.companies_df['churned'].copy()
        
        return X, y
    
    def process(self):
        """
        Run the full preprocessing pipeline.
        
        Returns:
            tuple: (X, y) - Feature matrix and target vector
        """
        return (self
                .calculate_tenure()
                .calculate_time_to_renewal()
                .create_churn_label()
                .calculate_additional_features()
                .enrich_from_deals()
                .prepare_for_modeling())