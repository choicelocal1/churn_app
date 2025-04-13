#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration settings for the churn prediction application.
"""

class Config:
    """Configuration settings for the application"""
    
    # HubSpot API settings
    HUBSPOT_BASE_URL = "https://api.hubapi.com"
    
    # Data extraction settings
    COMPANIES_LIMIT = 100
    DEALS_LIMIT = 100
    TICKETS_LIMIT = 100
    ENGAGEMENTS_LIMIT = 100
    
    # Company properties to extract
    COMPANY_PROPERTIES = [
        "name", 
        "industry", 
        "hs_lead_status", 
        "lifecyclestage", 
        "total_revenue", 
        "numberofemployees", 
        "createdate", 
        "num_locations", 
        "contract_value", 
        "last_meeting_date",
        "contract_end_date", 
        "contract_start_date", 
        "client_status"
    ]
    
    # Deal properties to extract
    DEAL_PROPERTIES = [
        "dealname", 
        "amount", 
        "dealstage", 
        "closedate", 
        "createdate", 
        "hs_lastmodifieddate", 
        "pipeline",
        "hs_deal_stage_probability",
        "notes_last_updated", 
        "notes_last_contacted", 
        "num_contacted_notes",
        "num_notes", 
        "hs_analytics_source", 
        "hs_analytics_source_data_1"
    ]
    
    # Ticket properties to extract
    TICKET_PROPERTIES = [
        "subject", 
        "content", 
        "hs_pipeline", 
        "hs_pipeline_stage", 
        "hs_ticket_priority", 
        "createdate", 
        "hs_lastmodifieddate",
        "hs_ticket_category"
    ]
    
    # Engagement properties to extract
    ENGAGEMENT_PROPERTIES = [
        "hs_activity_type", 
        "hs_timestamp", 
        "hs_email_direction",
        "hs_email_subject", 
        "hs_email_status", 
        "hs_meeting_outcome",
        "hs_call_direction", 
        "hs_call_disposition"
    ]
    
    # Model training settings
    TEST_SIZE = 0.2
    RANDOM_SEED = 42
    
    # Model feature settings
    DEFAULT_MODEL_FEATURES = [
        'tenure_days', 
        'days_to_renewal', 
        'numberofemployees', 
        'num_locations',
        'contract_value', 
        'industry', 
        'days_since_last_meeting'
    ]
    
    # Default churn definitions
    DEFAULT_CHURN_DEFINITION = "client_status == 'churned'"
    DEFAULT_DEAL_CHURN_DEFINITION = "dealstage == 'closedlost'"
    
    # File paths
    DEFAULT_MODEL_PATH = "models/churn_model_latest.joblib"