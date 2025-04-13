#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module handles data extraction from HubSpot API.
"""

import pandas as pd
import requests
import json
from requests.exceptions import RequestException
from config.config import Config


class HubSpotDataExtractor:
    """Class to extract data from HubSpot API"""
    
    def __init__(self, api_key, base_url=None):
        """
        Initialize the HubSpot data extractor.
        
        Args:
            api_key (str): HubSpot API key
            base_url (str, optional): HubSpot API base URL. Defaults to Config.HUBSPOT_BASE_URL.
        """
        self.api_key = api_key
        self.base_url = base_url or Config.HUBSPOT_BASE_URL
        
        # Private App Tokens use a different authorization header format
        if api_key and api_key.startswith('pat-'):
            self.headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
        else:
            # Legacy API keys use a different format
            self.headers = {
                'Authorization': f'apikey {self.api_key}',
                'Content-Type': 'application/json'
            }
        
        # Print masked API key for verification (show only first 4 and last 4 chars)
        if api_key and len(api_key) > 8:
            masked_key = api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]
            print(f"Using API key: {masked_key}")
        else:
            print("Warning: API key seems too short or empty")
            
        print(f"Initialized HubSpot API client with base URL: {self.base_url}")
    
    def get_companies(self, limit=None, properties=None):
        """
        Extract company data from HubSpot.
        
        Args:
            limit (int, optional): Maximum number of companies to extract. 
                                  Defaults to Config.COMPANIES_LIMIT.
            properties (list, optional): Company properties to extract. 
                                       Defaults to Config.COMPANY_PROPERTIES.
        
        Returns:
            dict: JSON response from HubSpot API
        """
        # Ensure limit doesn't exceed 100 (HubSpot API limit)
        if limit and limit > 100:
            print(f"Warning: HubSpot API limits requests to 100 objects. Reducing limit from {limit} to 100.")
            limit = 100
        
        limit = limit or 100  # Default to 100 if not specified
        properties = properties or Config.COMPANY_PROPERTIES
        
        companies_endpoint = f"{self.base_url}/crm/v3/objects/companies"
        
        # Format properties as a single comma-separated string parameter
        params = {
            'limit': limit,
            'properties': ','.join(properties),  # Join properties with commas
            'archived': False
        }
        
        print(f"Requesting up to {limit} companies with {len(properties)} properties")
        print(f"API endpoint: {companies_endpoint}")
        print(f"First few properties: {', '.join(properties[:3])}...")

        try:
            print("Sending API request...")
            response = requests.get(companies_endpoint, headers=self.headers, params=params)
            
            print(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Error details: {response.text[:500]}")  # Print first 500 chars of error
                
                # Try to parse error for more details
                try:
                    error_json = response.json()
                    if 'message' in error_json:
                        print(f"Error message: {error_json['message']}")
                    if 'errors' in error_json:
                        for error in error_json['errors']:
                            print(f"- {error.get('message', 'Unknown error')}")
                except:
                    print("Could not parse error details as JSON")
                    
            response.raise_for_status()
            result = response.json()
            print(f"Successfully retrieved {len(result.get('results', []))} companies")
            return result
            
        except RequestException as e:
            print(f"Error fetching companies: {e}")
            print(f"Request URL: {companies_endpoint}")
            print(f"Request parameters: {params}")
            
            # Check specifically for authentication issues
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code == 401:
                    print("Authentication failed: Please check your API key")
                elif e.response.status_code == 403:
                    print("Access forbidden: Your API key may not have sufficient permissions")
                
            return None
    
    def get_deals(self, limit=None, properties=None):
        """
        Extract deal data from HubSpot.
        
        Args:
            limit (int, optional): Maximum number of deals to extract. 
                                  Defaults to Config.DEALS_LIMIT.
            properties (list, optional): Deal properties to extract. 
                                       Defaults to Config.DEAL_PROPERTIES.
        
        Returns:
            dict: JSON response from HubSpot API
        """
        # Ensure limit doesn't exceed 100 (HubSpot API limit)
        if limit and limit > 100:
            print(f"Warning: HubSpot API limits requests to 100 objects. Reducing limit from {limit} to 100.")
            limit = 100
            
        limit = limit or 100  # Default to 100 if not specified
        properties = properties or Config.DEAL_PROPERTIES
        
        deals_endpoint = f"{self.base_url}/crm/v3/objects/deals"
        
        # Format properties as a single comma-separated string parameter
        params = {
            'limit': limit,
            'properties': ','.join(properties),  # Join properties with commas
            'archived': False
        }
        
        print(f"Requesting up to {limit} deals with {len(properties)} properties")
        
        try:
            response = requests.get(deals_endpoint, headers=self.headers, params=params)
            
            print(f"Deals response status code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Deals error details: {response.text[:500]}")
                
            response.raise_for_status()
            result = response.json()
            print(f"Successfully retrieved {len(result.get('results', []))} deals")
            return result
            
        except RequestException as e:
            print(f"Error fetching deals: {e}")
            return None
    
    def get_tickets(self, limit=None, properties=None):
        """
        Extract support ticket data from HubSpot.
        
        Args:
            limit (int, optional): Maximum number of tickets to extract. 
                                  Defaults to Config.TICKETS_LIMIT.
            properties (list, optional): Ticket properties to extract. 
                                       Defaults to Config.TICKET_PROPERTIES.
        
        Returns:
            dict: JSON response from HubSpot API
        """
        # Ensure limit doesn't exceed 100 (HubSpot API limit)
        if limit and limit > 100:
            print(f"Warning: HubSpot API limits requests to 100 objects. Reducing limit from {limit} to 100.")
            limit = 100
            
        limit = limit or 100  # Default to 100 if not specified
        properties = properties or Config.TICKET_PROPERTIES
        
        tickets_endpoint = f"{self.base_url}/crm/v3/objects/tickets"
        
        # Format properties as a single comma-separated string parameter
        params = {
            'limit': limit,
            'properties': ','.join(properties),  # Join properties with commas
            'archived': False
        }
        
        print(f"Requesting up to {limit} tickets with {len(properties)} properties")
        
        try:
            response = requests.get(tickets_endpoint, headers=self.headers, params=params)
            
            print(f"Tickets response status code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Tickets error details: {response.text[:500]}")
                
            response.raise_for_status()
            result = response.json()
            print(f"Successfully retrieved {len(result.get('results', []))} tickets")
            return result
            
        except RequestException as e:
            print(f"Error fetching tickets: {e}")
            return None
            
    def get_engagement_data(self, limit=None):
        """
        Extract engagement data (calls, emails, meetings) from HubSpot.
        
        Args:
            limit (int, optional): Maximum number of engagements to extract
            
        Returns:
            dict: JSON response from HubSpot API
        """
        limit = limit or 100  # Default to 100 if not specified
        
        engagement_endpoint = f"{self.base_url}/crm/v3/objects/engagements"
        
        # Format properties as a single comma-separated string parameter
        engagement_properties = [
            'hs_activity_type', 'hs_timestamp', 'hs_email_direction',
            'hs_email_subject', 'hs_email_status', 'hs_meeting_outcome',
            'hs_call_direction', 'hs_call_disposition'
        ]
        
        params = {
            'limit': limit,
            'properties': ','.join(engagement_properties),
            'archived': False
        }
        
        print(f"Requesting up to {limit} engagements")
        
        try:
            response = requests.get(engagement_endpoint, headers=self.headers, params=params)
            
            print(f"Engagements response status code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Engagements error details: {response.text[:500]}")
                
            response.raise_for_status()
            result = response.json()
            print(f"Successfully retrieved {len(result.get('results', []))} engagements")
            return result
            
        except RequestException as e:
            print(f"Error fetching engagements: {e}")
            return None
    
    def test_api_connection(self):
        """
        Test basic API connection with a minimal request.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        test_endpoint = f"{self.base_url}/crm/v3/objects/companies"
        params = {
            'limit': 1,
            'properties': 'name'  # Just request the name property
        }
        
        print("Testing HubSpot API connection...")
        
        try:
            response = requests.get(test_endpoint, headers=self.headers, params=params)
            
            if response.status_code == 200:
                print("API connection test successful!")
                return True
            else:
                print(f"API connection test failed with status code: {response.status_code}")
                print(f"Error details: {response.text[:500]}")
                return False
                
        except RequestException as e:
            print(f"API connection test failed with error: {e}")
            return False
    
    def extract_all_data(self):
        """
        Extract all relevant data and combine into DataFrames.
        
        Returns:
            tuple: (companies_df, deals_df, tickets_df, engagements_df) - Pandas DataFrames
        """
        # First, test the API connection
        if not self.test_api_connection():
            print("Cannot proceed with data extraction due to API connection issues")
            return None, None, None, None
            
        # Get company data
        print("\nExtracting company data...")
        companies_data = self.get_companies()
        companies_df = None
        if companies_data:
            companies_df = pd.DataFrame([
                {**{'id': company['id']}, **company['properties']} 
                for company in companies_data['results']
            ])
        else:
            print("Failed to extract company data")
        
        # Get deals data
        print("\nExtracting deal data...")
        deals_data = self.get_deals()
        deals_df = None
        if deals_data:
            deals_df = pd.DataFrame([
                {**{'id': deal['id']}, **deal['properties']} 
                for deal in deals_data['results']
            ])
        else:
            print("Failed to extract deals data")
        
        # Get ticket data
        print("\nExtracting ticket data...")
        tickets_data = self.get_tickets()
        tickets_df = None
        if tickets_data:
            tickets_df = pd.DataFrame([
                {**{'id': ticket['id']}, **ticket['properties']} 
                for ticket in tickets_data['results']
            ])
        else:
            print("Failed to extract ticket data")
            
        # Get engagement data
        print("\nExtracting engagement data...")
        engagements_data = self.get_engagement_data()
        engagements_df = None
        if engagements_data:
            engagements_df = pd.DataFrame([
                {**{'id': engagement['id']}, **engagement['properties']} 
                for engagement in engagements_data['results']
            ])
        else:
            print("Failed to extract engagement data")
        
        # Print summary
        print("\nExtraction summary:")
        if companies_df is not None:
            print(f"Companies: {len(companies_df)}")
        if deals_df is not None:
            print(f"Deals: {len(deals_df)}")
        if tickets_df is not None:
            print(f"Tickets: {len(tickets_df)}")
        if engagements_df is not None:
            print(f"Engagements: {len(engagements_df)}")
        
        return companies_df, deals_df, tickets_df, engagements_df