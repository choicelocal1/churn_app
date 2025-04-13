#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module handles data preprocessing for deal-level churn prediction.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from config.config import Config


class DealChurnPreprocessor:
    """Class to preprocess and engineer features for deal-level churn prediction"""
    
    def __init__(self, deals_df, companies_df=None, engagements_df=None, churn_definition=None):
        """
        Initialize the preprocessor.
        
        Args:
            deals_df (DataFrame): DataFrame containing deal data
            companies_df (DataFrame, optional): DataFrame containing company data
            engagements_df (DataFrame, optional): DataFrame containing engagement data
            churn_definition (str, optional): Definition of churn
        """
        self.deals_df = deals_df.copy() if deals_df is not None else None
        self.companies_df = companies_df.copy() if companies_df is not None else None
        self.engagements_df = engagements_df.copy() if engagements_df is not None else None
        self.churn_definition = churn_definition or Config.DEFAULT_CHURN_DEFINITION
    
    def clean_data(self):
        """
        Clean and prepare the data for modeling.
        
        Returns:
            self: For method chaining
        """
        if self.deals_df is None:
            print("No deals data available")
            return self
            
        print(f"Cleaning deal data with {len(self.deals_df)} records")
        
        # Convert date columns to datetime
        date_columns = [
            'create_date', 'close_date', 'cancellation_date', 'billing_start_date', 'createdate', 'closedate'
        ]
        
        for col in date_columns:
            if col in self.deals_df.columns:
                self.deals_df[col] = pd.to_datetime(self.deals_df[col], errors='coerce')
        
        # Convert numeric columns to proper types
        numeric_columns = ['amount', 'notes_count', 'meetings_count']
        for col in numeric_columns:
            if col in self.deals_df.columns:
                self.deals_df[col] = pd.to_numeric(self.deals_df[col], errors='coerce')
        
        return self
    
    def create_churn_label(self):
        """
        Create or use existing churn label.
        
        Returns:
            self: For method chaining
        """
        print(f"Creating churn label using definition: {self.churn_definition}")
        
        # Default - if we can't determine churn status, assume not churned
        self.deals_df['churned'] = 0
        
        # Check if 'is_churned' already exists
        if 'is_churned' in self.deals_df.columns:
            print("Using existing 'is_churned' field from deals data")
            
            # Convert boolean to int (0/1)
            if self.deals_df['is_churned'].dtype == bool:
                self.deals_df['churned'] = self.deals_df['is_churned'].astype(int)
            # If it's already numeric, just copy it
            elif pd.api.types.is_numeric_dtype(self.deals_df['is_churned']):
                self.deals_df['churned'] = self.deals_df['is_churned']
            # Otherwise convert strings to int (assuming "True"/"False" or similar)
            else:
                self.deals_df['churned'] = self.deals_df['is_churned'].map(
                    lambda x: 1 if str(x).lower() in ('true', 't', 'yes', 'y', '1') else 0
                )
        
        # For HubSpot data - use dealstage
        elif 'dealstage' in self.deals_df.columns:
            print("Creating churn label based on dealstage")
            deal_stages = self.deals_df['dealstage'].unique()
            print(f"Available dealstages: {deal_stages}")
            
            # Look for closed lost stages
            lost_stages = [stage for stage in deal_stages if 'lost' in str(stage).lower()]
            if lost_stages:
                print(f"Found lost stages: {lost_stages}")
                
                # Mark deals with lost stages as churned
                for stage in lost_stages:
                    self.deals_df.loc[self.deals_df['dealstage'] == stage, 'churned'] = 1
            else:
                print("No 'lost' stages found in the data")
            
        else:
            print("Using custom churn definition")
            try:
                # Try to evaluate the definition literally
                if "dealstage" in self.churn_definition and "dealstage" in self.deals_df.columns:
                    # Extract the target stage
                    if "==" in self.churn_definition:
                        target_stage = self.churn_definition.split("==")[1].strip().strip("'").strip('"')
                        print(f"Looking for deals with stage: {target_stage}")
                        self.deals_df['churned'] = (self.deals_df['dealstage'] == target_stage).astype(int)
                else:
                    # Replace field references with dataframe column references
                    eval_expr = self.churn_definition
                    for col in self.deals_df.columns:
                        if col in eval_expr:
                            eval_expr = eval_expr.replace(col, f"self.deals_df['{col}']")
                    
                    # Evaluate the expression
                    print(f"Evaluating churn expression: {eval_expr}")
                    churn_mask = eval(eval_expr)
                    self.deals_df['churned'] = churn_mask.astype(int)
            except Exception as e:
                print(f"Error evaluating churn definition: {e}")
        
        # Report churn rate
        churn_rate = self.deals_df['churned'].mean()
        print(f"Created churn label with {self.deals_df['churned'].sum()} churned deals ({churn_rate:.1%} churn rate)")
        
        return self
    
    def calculate_features(self):
        """
        Calculate additional features from deal data.
        
        Returns:
            self: For method chaining
        """
        # Deal age (days from creation to now or close)
        create_date_col = None
        for col_name in ['create_date', 'createdate']:
            if col_name in self.deals_df.columns:
                create_date_col = col_name
                break
                
        if create_date_col:
            # Use datetime without timezone info
            current_date = pd.Timestamp(datetime.now().date())
            
            # Ensure dates don't have timezone information to avoid tz conflicts
            self.deals_df[create_date_col] = self.deals_df[create_date_col].dt.tz_localize(None)
            
            # Find close date column
            close_date_col = None
            for col_name in ['close_date', 'closedate']:
                if col_name in self.deals_df.columns:
                    close_date_col = col_name
                    break
            
            # If close_date exists, use it, otherwise use current date
            if close_date_col:
                # Remove timezone info
                self.deals_df[close_date_col] = self.deals_df[close_date_col].dt.tz_localize(None)
                
                # Calculate age
                self.deals_df['deal_age_days'] = np.where(
                    self.deals_df[close_date_col].notna(),
                    (self.deals_df[close_date_col] - self.deals_df[create_date_col]).dt.days,
                    (current_date - self.deals_df[create_date_col]).dt.days
                )
            else:
                self.deals_df['deal_age_days'] = (current_date - self.deals_df[create_date_col]).dt.days
            
            # Handle negative values (data errors)
            self.deals_df.loc[self.deals_df['deal_age_days'] < 0, 'deal_age_days'] = 0
        
        # Interaction rates from notes and meetings
        if 'notes_count' in self.deals_df.columns and 'deal_age_days' in self.deals_df.columns:
            # Avoid division by zero
            days_for_calc = self.deals_df['deal_age_days'].copy()
            days_for_calc = days_for_calc.replace(0, 1)  # Replace 0 with 1 to avoid div/0
            
            self.deals_df['notes_per_day'] = self.deals_df['notes_count'] / days_for_calc
        
        if 'meetings_count' in self.deals_df.columns and 'deal_age_days' in self.deals_df.columns:
            # Avoid division by zero
            days_for_calc = self.deals_df['deal_age_days'].copy()
            days_for_calc = days_for_calc.replace(0, 1)  # Replace 0 with 1 to avoid div/0
            
            self.deals_df['meetings_per_day'] = self.deals_df['meetings_count'] / days_for_calc
            
        # Calculate deal velocity
        self.calculate_deal_velocity()
        
        # Calculate engagement metrics
        self.calculate_engagement_metrics()
        
        return self
        
    def calculate_deal_velocity(self):
        """
        Calculate how quickly deals move through stages.
        
        Returns:
            self: For method chaining
        """
        # Check if we have the necessary date fields
        if 'createdate' in self.deals_df.columns and 'closedate' in self.deals_df.columns:
            # Calculate days from creation to close
            self.deals_df['createdate'] = pd.to_datetime(self.deals_df['createdate'], errors='coerce')
            self.deals_df['closedate'] = pd.to_datetime(self.deals_df['closedate'], errors='coerce')
            
            # For closed deals, calculate velocity
            closed_mask = self.deals_df['closedate'].notna()
            
            if closed_mask.any():
                # Calculate time to close in days
                self.deals_df.loc[closed_mask, 'days_to_close'] = (
                    self.deals_df.loc[closed_mask, 'closedate'] - 
                    self.deals_df.loc[closed_mask, 'createdate']
                ).dt.days
                
                # Calculate velocity (inverse of time to close)
                # A higher velocity means faster closing
                self.deals_df.loc[closed_mask, 'deal_velocity'] = 1 / self.deals_df.loc[closed_mask, 'days_to_close']
                
                # For negative or zero days (data errors), set to NaN
                self.deals_df.loc[self.deals_df['days_to_close'] <= 0, 'deal_velocity'] = np.nan
                
                # Fill NaN values with median
                median_velocity = self.deals_df['deal_velocity'].median()
                self.deals_df['deal_velocity'] = self.deals_df['deal_velocity'].fillna(median_velocity)
                
                print(f"Calculated deal velocity for {closed_mask.sum()} closed deals")
        
        return self

    def calculate_engagement_metrics(self):
        """
        Calculate engagement metrics like communication frequency.
        
        Returns:
            self: For method chaining
        """
        # For this we'd need engagement data joined to deals
        # Since we don't have that yet, we'll create proxy metrics
        
        # 1. Activity level based on deal age
        if 'deal_age_days' in self.deals_df.columns:
            # Categorize deals by age
            self.deals_df['age_category'] = pd.cut(
                self.deals_df['deal_age_days'],
                bins=[0, 30, 90, 180, 365, float('inf')],
                labels=['new', 'active', 'established', 'mature', 'old']
            )
            
            # For categoricals, convert to string for modeling
            self.deals_df['age_category'] = self.deals_df['age_category'].astype(str)
        
        # 2. For each dealstage, calculate average time spent
        if 'dealstage' in self.deals_df.columns and 'deal_age_days' in self.deals_df.columns:
            # Calculate average age per stage
            stage_ages = self.deals_df.groupby('dealstage')['deal_age_days'].mean()
            
            # Add relative age (compared to stage average)
            self.deals_df['relative_age'] = self.deals_df.apply(
                lambda row: row['deal_age_days'] / stage_ages.get(row['dealstage'], 1) 
                if row['dealstage'] in stage_ages.index else 1,
                axis=1
            )
            
            print("Calculated relative age metric based on deal stages")
        
        return self
    
    def enrich_from_companies(self):
        """
        Enrich deal data with company information.
        
        Returns:
            self: For method chaining
        """
        if self.companies_df is None:
            print("No companies data available for enrichment")
            return self
        
        # Check if there's a company ID field in deals
        company_id_col = None
        possible_cols = ['company_id', 'companyid', 'associated_company_id']
        
        for col in possible_cols:
            if col in self.deals_df.columns:
                company_id_col = col
                break
        
        if company_id_col is None:
            print("No company ID column found in deals data, unable to join to companies")
            return self
        
        try:
            # Select useful company features
            company_features = [
                'id', 'industry', 'numberofemployees', 'total_revenue'
            ]
            
            # Keep only columns that exist
            company_features = [col for col in company_features if col in self.companies_df.columns]
            
            if not company_features:
                print("No useful company features found")
                return self
            
            # Join company data to deals
            self.deals_df = self.deals_df.merge(
                self.companies_df[company_features],
                left_on=company_id_col,
                right_on='id',
                how='left',
                suffixes=('', '_company')
            )
            
            print(f"Enriched deal data with {len(company_features)} company features")
            
        except Exception as e:
            print(f"Error enriching from companies: {e}")
        
        return self
        
    def enrich_from_engagements(self):
        """
        Enrich deal data with engagement metrics.
        
        Returns:
            self: For method chaining
        """
        if self.engagements_df is None:
            print("No engagement data available for enrichment")
            return self
            
        # Check if the engagements can be linked to deals
        if 'deal_id' not in self.engagements_df.columns:
            print("No deal_id field in engagement data, unable to join to deals")
            return self
            
        try:
            # Group engagements by deal ID and count
            engagement_counts = self.engagements_df.groupby('deal_id').size().reset_index(name='engagement_count')
            
            # Join engagement counts to deals
            self.deals_df = self.deals_df.merge(
                engagement_counts,
                left_on='id',
                right_on='deal_id',
                how='left'
            )
            
            # Fill missing values with 0 (deals with no engagements)
            self.deals_df['engagement_count'] = self.deals_df['engagement_count'].fillna(0)
            
            # Calculate engagement rate (engagements per day)
            if 'deal_age_days' in self.deals_df.columns:
                # Avoid division by zero
                days_for_calc = self.deals_df['deal_age_days'].copy()
                days_for_calc = days_for_calc.replace(0, 1)  # Replace 0 with 1 to avoid div/0
                
                self.deals_df['engagement_rate'] = self.deals_df['engagement_count'] / days_for_calc
            
            print(f"Enriched deal data with engagement metrics")
            
        except Exception as e:
            print(f"Error enriching from engagements: {e}")
        
        return self
    
    def prepare_for_modeling(self):
        """
        Prepare the final dataset for modeling.
        
        Returns:
            tuple: (X, y) - Feature matrix and target vector
        """
        # Get available columns
        available_columns = self.deals_df.columns.tolist()
        print(f"Available columns for modeling: {available_columns}")
        
        # Preferred columns for modeling (based on your data sample)
        preferred_columns = [
            'amount', 'notes_count', 'meetings_count', 'deal_age_days', 
            'notes_per_day', 'meetings_per_day', 'vertical', 'deal_velocity',
            'relative_age', 'engagement_count', 'engagement_rate'
        ]
        
        # Add company features if available
        company_features = ['industry', 'numberofemployees', 'total_revenue']
        preferred_columns.extend([col for col in company_features if col in available_columns])
        
        # Select columns that exist in the data
        model_columns = [col for col in preferred_columns if col in available_columns]
        
        # If we have too few columns, try to use any numeric columns
        if len(model_columns) < 3:
            print("Warning: Too few preferred columns available. Adding additional numeric columns.")
            numeric_cols = self.deals_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            for col in numeric_cols:
                if col not in model_columns and col != 'churned' and col != 'id' and col != 'deal_id':
                    model_columns.append(col)
        
        print(f"Using {len(model_columns)} columns for modeling: {model_columns}")
        
        # Create feature dataframe
        X = self.deals_df[model_columns].copy()
        y = self.deals_df['churned'].copy()
        
        return X, y
    
    def process(self):
        """
        Run the full preprocessing pipeline.
        
        Returns:
            tuple: (X, y) - Feature matrix and target vector
        """
        return (self
                .clean_data()
                .create_churn_label()
                .calculate_features()
                .enrich_from_companies()
                .enrich_from_engagements()
                .prepare_for_modeling())