#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility script to generate sample data for testing and development.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_sample_companies(num_records=100, churn_rate=0.15):
    """
    Generate sample company data.
    
    Args:
        num_records (int): Number of company records to generate
        churn_rate (float): Proportion of churned clients
        
    Returns:
        DataFrame: Sample companies data
    """
    # Generate base records
    companies = []
    
    industries = ['Technology', 'Retail', 'Healthcare', 'Finance', 
                  'Manufacturing', 'Education', 'Hospitality', 'Real Estate']
    
    locations_range = [1, 3, 5, 10, 15, 20, 50, 100]
    
    current_date = datetime.now()
    
    for i in range(num_records):
        # Determine if this company has churned
        is_churned = random.random() < churn_rate
        
        # Contract dates
        if is_churned:
            # Churned companies have older start dates and ended contracts
            start_date = current_date - timedelta(days=random.randint(365, 1825))  # 1-5 years ago
            end_date = start_date + timedelta(days=random.randint(180, 730))  # 6-24 months contract
            
            # Ensure end date is in the past
            if end_date > current_date:
                end_date = current_date - timedelta(days=random.randint(30, 180))
                
            last_meeting = min(end_date - timedelta(days=random.randint(15, 90)), current_date)
        else:
            # Active companies have more recent start dates
            start_date = current_date - timedelta(days=random.randint(30, 1095))  # 1-36 months ago
            end_date = start_date + timedelta(days=random.randint(365, 1095))  # 12-36 months contract
            
            # Some active contracts might be near renewal
            if random.random() < 0.3:  # 30% of active companies near renewal
                end_date = current_date + timedelta(days=random.randint(1, 60))
                
            last_meeting = current_date - timedelta(days=random.randint(1, 60))
        
        # Create company record
        company = {
            'id': str(i + 1),
            'name': f"Company {i + 1}",
            'industry': random.choice(industries),
            'hs_lead_status': 'CLOSED' if is_churned else 'OPEN',
            'lifecyclestage': 'customer',
            'numberofemployees': random.randint(10, 5000),
            'total_revenue': random.randint(100000, 50000000),
            'num_locations': random.choice(locations_range),
            'contract_value': random.randint(5000, 500000),
            'contract_start_date': start_date.strftime('%Y-%m-%d'),
            'contract_end_date': end_date.strftime('%Y-%m-%d'),
            'last_meeting_date': last_meeting.strftime('%Y-%m-%d'),
            'client_status': 'churned' if is_churned else 'active'
        }
        
        companies.append(company)
    
    return pd.DataFrame(companies)


def generate_all_sample_data(output_dir, num_companies=100):
    """
    Generate all sample data files and save to specified directory.
    
    Args:
        output_dir (str): Directory to save output files
        num_companies (int): Number of companies to generate
        
    Returns:
        tuple: Paths to generated files (companies_path)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate company data
    print(f"Generating {num_companies} sample companies...")
    companies_df = generate_sample_companies(num_records=num_companies)
    
    # Save to CSV files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    companies_path = os.path.join(output_dir, f"companies_{timestamp}.csv")
    
    companies_df.to_csv(companies_path, index=False)
    
    print(f"Generated {len(companies_df)} companies")
    print(f"Files saved to {output_dir}")
    
    return companies_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample data for churn prediction')
    parser.add_argument('--output-dir', default='data/raw', help='Output directory for sample data')
    parser.add_argument('--num-companies', type=int, default=100, help='Number of companies to generate')
    
    args = parser.parse_args()
    
    generate_all_sample_data(args.output_dir, args.num_companies)