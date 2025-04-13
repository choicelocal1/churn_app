def __init__(self, api_key, base_url=None):
    """
    Initialize the HubSpot data extractor.
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
    
    print(f"Initialized HubSpot API client with base URL: {self.base_url}")
    # Rest of the method...