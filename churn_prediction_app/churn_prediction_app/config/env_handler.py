#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Environment variable handler for the churn prediction application.
Supports loading API keys and other configuration from .env file.
"""

import os
from dotenv import load_dotenv, find_dotenv

def load_environment():
    """
    Load environment variables from .env file.
    
    Returns:
        bool: True if .env file was found and loaded, False otherwise
    """
    # Look for .env file in current directory and parent directories
    env_path = find_dotenv(usecwd=True)
    
    if env_path:
        load_dotenv(env_path)
        return True
    else:
        return False

def get_api_key():
    """
    Get HubSpot API key from environment variables.
    
    Returns:
        str: HubSpot API key if found, None otherwise
    """
    # Try to load from .env file
    load_environment()
    
    # Get API key from environment
    return os.environ.get("HUBSPOT_API_KEY")

def get_config_value(key, default=None):
    """
    Get a configuration value from environment variables.
    
    Args:
        key (str): Environment variable name
        default: Default value to return if not found
        
    Returns:
        Value of environment variable if found, default otherwise
    """
    # Try to load from .env file
    load_environment()
    
    # Get value from environment
    return os.environ.get(key, default)