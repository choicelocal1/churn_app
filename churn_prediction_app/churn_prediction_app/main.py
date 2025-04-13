#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main application for churn prediction using HubSpot data.
This script orchestrates the entire pipeline from data extraction to prediction.
"""

import argparse
import os
import pandas as pd
import joblib
import shutil
from datetime import datetime
import glob

# Import components
from config.config import Config
from config.env_handler import get_api_key
from data.hubspot_extractor import HubSpotDataExtractor
from data.data_loader import DataLoader
from preprocessing.preprocessor import ChurnDataPreprocessor
from preprocessing.deal_preprocessor import DealChurnPreprocessor
from modeling.model_builder import ChurnModelBuilder
from prediction.predictor import ChurnPredictor
from utils.visualization import create_feature_importance_plot, create_evaluation_plots
from utils.generate_sample_data import generate_all_sample_data


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='HubSpot Churn Prediction Application')
    
    # Define command groups
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Extract data command
    extract_parser = subparsers.add_parser('extract', help='Extract data from HubSpot')
    extract_parser.add_argument('--api-key', help='HubSpot API key')
    extract_parser.add_argument('--use-env-api-key', action='store_true', 
                                help='Use API key from .env file')
    extract_parser.add_argument('--output-dir', default='data/raw', 
                                help='Output directory for extracted data')
    
    # Generate sample data command
    sample_parser = subparsers.add_parser('generate-sample', help='Generate sample data for testing')
    sample_parser.add_argument('--output-dir', default='data/sample', 
                               help='Output directory for sample data')
    sample_parser.add_argument('--num-companies', type=int, default=100, 
                               help='Number of companies to generate')
    
    # Train model command
    train_parser = subparsers.add_parser('train', help='Train churn prediction model')
    train_parser.add_argument('--data-dir', default='data/raw', 
                              help='Directory with input data')
    train_parser.add_argument('--model-dir', default='models', 
                              help='Directory to save model')
    train_parser.add_argument('--churn-definition', default=Config.DEFAULT_CHURN_DEFINITION,
                              help='Definition of churn for labeling')
    train_parser.add_argument('--model-type', default='random_forest',
                              choices=['random_forest', 'logistic_regression', 'xgboost'],
                              help='Type of model to train (xgboost will fall back to random_forest)')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions with trained model')
    predict_parser.add_argument('--model-path', default='models/churn_model_latest.joblib',
                                help='Path to trained model')
    predict_parser.add_argument('--input-data', required=True, 
                                help='Path to input data file')
    predict_parser.add_argument('--output-file', 
                                help='Path to save predictions')
    
    return parser.parse_args()


def extract_data(args):
    """Extract data from HubSpot"""
    print("Extracting data from HubSpot...")
    
    # Get API key
    api_key = args.api_key
    if args.use_env_api_key or api_key is None:
        api_key = get_api_key()
        if api_key is None:
            print("Error: No API key provided. Use --api-key or set HUBSPOT_API_KEY in .env file.")
            return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize extractor
    extractor = HubSpotDataExtractor(api_key=api_key)
    
    # Extract data
    companies_df, deals_df, tickets_df, engagements_df = extractor.extract_all_data()
    
    # Save to CSV files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if companies_df is not None:
        companies_path = f"{args.output_dir}/companies_{timestamp}.csv"
        companies_df.to_csv(companies_path, index=False)
        print(f"Saved companies data: {len(companies_df)} records to {companies_path}")
    
    if deals_df is not None:
        deals_path = f"{args.output_dir}/deals_{timestamp}.csv"
        deals_df.to_csv(deals_path, index=False)
        print(f"Saved deals data: {len(deals_df)} records to {deals_path}")
    
    if tickets_df is not None:
        tickets_path = f"{args.output_dir}/tickets_{timestamp}.csv"
        tickets_df.to_csv(tickets_path, index=False)
        print(f"Saved tickets data: {len(tickets_df)} records to {tickets_path}")
        
    if engagements_df is not None:
        engagements_path = f"{args.output_dir}/engagements_{timestamp}.csv"
        engagements_df.to_csv(engagements_path, index=False)
        print(f"Saved engagements data: {len(engagements_df)} records to {engagements_path}")
    
    print("Data extraction complete!")


def generate_sample_data(args):
    """Generate sample data for testing"""
    print(f"Generating {args.num_companies} sample companies...")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate sample data
    companies_path = generate_all_sample_data(args.output_dir, args.num_companies)
    
    print(f"Sample data generation complete!")
    print(f"Companies data saved to: {companies_path}")


def train_model(args):
    """Train a churn prediction model"""
    print("Training churn prediction model...")
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.data_dir}...")
    data_loader = DataLoader(data_dir=args.data_dir)
    companies_df, deals_df, tickets_df, engagements_df = data_loader.load_most_recent_data()
    
    # Check which data is available
    if deals_df is not None and len(deals_df) > 0:
        print(f"Using deal-level churn prediction with {len(deals_df)} deals")
        print("\nAvailable columns in deals data:")
        print(deals_df.columns.tolist())
        
        # Use deal preprocessor
        preprocessor = DealChurnPreprocessor(
            deals_df=deals_df,
            companies_df=companies_df,
            engagements_df=engagements_df,
            churn_definition=args.churn_definition
        )
    elif companies_df is not None and len(companies_df) > 0:
        print(f"Using company-level churn prediction with {len(companies_df)} companies")
        
        # Use company preprocessor (original)
        preprocessor = ChurnDataPreprocessor(
            companies_df=companies_df,
            deals_df=deals_df,
            tickets_df=tickets_df,
            churn_definition=args.churn_definition
        )
    else:
        print("No data found. Please extract data first or check the data directory.")
        return
    
    # Process data
    print("Preprocessing data...")
    X, y = preprocessor.process()
    
    # Check if we have enough data for modeling
    if len(X) == 0:
        print("No data available after preprocessing.")
        return
    
    # Build and train model with specified type
    print(f"Building {args.model_type} model...")
    model_builder = ChurnModelBuilder(X, y)
    model_builder.build(model_type=args.model_type)
    
    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"churn_model_{timestamp}.joblib"
    model_path = os.path.join(args.model_dir, model_filename)
    model_builder.save_model(filepath=model_path)
    
    # Create copy of latest model
    latest_model_path = os.path.join(args.model_dir, "churn_model_latest.joblib")
    if os.path.exists(latest_model_path):
        os.remove(latest_model_path)
    
    # Always use a copy (more reliable than symlinks)
    shutil.copy2(model_path, latest_model_path)
    
    print(f"Model training complete! Model saved to {model_path}")
    print(f"Latest model copy created at {latest_model_path}")


def predict(args):
    """Make predictions using a trained model and provide explanations"""
    print(f"Loading model from {args.model_path}...")
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} not found.")
        # Try to find any model file in the models directory
        model_files = glob.glob('models/churn_model_*.joblib')
        if model_files:
            latest_model = max(model_files, key=os.path.getctime)
            print(f"Using found model instead: {latest_model}")
            args.model_path = latest_model
        else:
            print("No model files found. Please train a model first.")
            return
    
    # Handle wildcard in input path
    input_files = []
    if '*' in args.input_data:
        # Find matching files
        matching_files = glob.glob(args.input_data)
        if not matching_files:
            print(f"Error: No files match pattern {args.input_data}")
            return
        input_files = matching_files
    else:
        if not os.path.exists(args.input_data):
            print(f"Error: Input file {args.input_data} not found.")
            return
        input_files = [args.input_data]
    
    # Load and combine all input data
    all_data = []
    for input_file in input_files:
        print(f"Loading input data from {os.path.basename(input_file)}...")
        data = pd.read_csv(input_file)
        all_data.append(data)
    
    if not all_data:
        print("No data loaded.")
        return
        
    input_df = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(input_df)} records from {len(input_files)} files")
    
    # Initialize predictor
    try:
        predictor = ChurnPredictor(model_path=args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Make predictions
    print("Making predictions with explanations...")
    result_df = predictor.predict(input_df)
    
    # Save or display results
    if args.output_file:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        result_df.to_csv(args.output_file, index=False)
        print(f"Predictions saved to {args.output_file}")
    else:
        # Display summary
        print("\nPrediction Summary:")
        print(f"Total deals: {len(result_df)}")
        predicted_to_churn = result_df['churn_prediction'].sum()
        churn_rate = result_df['churn_prediction'].mean() * 100
        print(f"Predicted to churn: {predicted_to_churn} ({churn_rate:.1f}%)")
        
        # Display top 10 at-risk deals
        print("\nTop 10 deals at risk of churning:")
        display_cols = ['deal_name', 'churn_probability', 'explanation']
        
        # Add amount if available
        if 'amount' in result_df.columns:
            display_cols.insert(2, 'amount')
            
        # Add deal stage if available
        if 'dealstage' in result_df.columns:
            display_cols.insert(1, 'dealstage')
        
        # Sort by churn probability and display
        at_risk = result_df.sort_values('churn_probability', ascending=False).head(10)
        pd.set_option('display.max_colwidth', 100)  # Set to display full explanation text
        print(at_risk[display_cols])


def main():
    """Main application entry point"""
    args = parse_arguments()
    
    # Create necessary directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/sample', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    if args.command == 'extract':
        extract_data(args)
    elif args.command == 'generate-sample':
        generate_sample_data(args)
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'predict':
        predict(args)
    else:
        print("Please specify a command: extract, generate-sample, train, or predict")
        print("Use --help for more information")


if __name__ == "__main__":
    main()