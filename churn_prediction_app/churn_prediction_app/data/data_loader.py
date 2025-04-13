#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module handles loading data from CSV files.
"""

import os
import glob
import pandas as pd


class DataLoader:
    """Class to load data from files"""
    
    def __init__(self, data_dir):
        """
        Initialize the data loader.
        
        Args:
            data_dir (str): Directory containing data files
        """
        self.data_dir = data_dir
    
    def _get_most_recent_file(self, prefix):
        """
        Get the most recent file with a given prefix.
        
        Args:
            prefix (str): File prefix to match
        
        Returns:
            str: Path to the most recent file, or None if no files found
        """
        pattern = os.path.join(self.data_dir, f"{prefix}_*.csv")
        files = glob.glob(pattern)
        
        if not files:
            return None
        
        # Sort by file modification time (most recent first)
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0]
    
    def load_most_recent_data(self):
        """
        Load the most recent company, deal, ticket, and engagement data.
        
        Returns:
            tuple: (companies_df, deals_df, tickets_df, engagements_df) - Pandas DataFrames
        """
        # Find most recent files
        companies_file = self._get_most_recent_file("companies")
        deals_file = self._get_most_recent_file("deals")
        tickets_file = self._get_most_recent_file("tickets")
        engagements_file = self._get_most_recent_file("engagements")
        
        # Load company data
        companies_df = None
        if companies_file:
            print(f"Loading company data from {os.path.basename(companies_file)}")
            companies_df = pd.read_csv(companies_file)
        
        # Load deal data
        deals_df = None
        if deals_file:
            print(f"Loading deal data from {os.path.basename(deals_file)}")
            deals_df = pd.read_csv(deals_file)
        
        # Load ticket data
        tickets_df = None
        if tickets_file:
            print(f"Loading ticket data from {os.path.basename(tickets_file)}")
            tickets_df = pd.read_csv(tickets_file)
            
        # Load engagement data
        engagements_df = None
        if engagements_file:
            print(f"Loading engagement data from {os.path.basename(engagements_file)}")
            engagements_df = pd.read_csv(engagements_file)
        
        return companies_df, deals_df, tickets_df, engagements_df
    
    def load_data_from_files(self, companies_file=None, deals_file=None, tickets_file=None, engagements_file=None):
        """
        Load data from specific files.
        
        Args:
            companies_file (str, optional): Path to companies CSV file
            deals_file (str, optional): Path to deals CSV file
            tickets_file (str, optional): Path to tickets CSV file
            engagements_file (str, optional): Path to engagements CSV file
        
        Returns:
            tuple: (companies_df, deals_df, tickets_df, engagements_df) - Pandas DataFrames
        """
        # Load company data
        companies_df = None
        if companies_file and os.path.exists(companies_file):
            companies_df = pd.read_csv(companies_file)
        
        # Load deal data
        deals_df = None
        if deals_file and os.path.exists(deals_file):
            deals_df = pd.read_csv(deals_file)
        
        # Load ticket data
        tickets_df = None
        if tickets_file and os.path.exists(tickets_file):
            tickets_df = pd.read_csv(tickets_file)
            
        # Load engagement data
        engagements_df = None
        if engagements_file and os.path.exists(engagements_file):
            engagements_df = pd.read_csv(engagements_file)
        
        return companies_df, deals_df, tickets_df, engagements_df