"""
Example usage of the OSE v3 API.

This file demonstrates how to interact with the API programmatically.
"""

import requests
import json


def example_single_prediction(base_url: str = "http://localhost:8000"):
    """Example: Single prediction request."""
    url = f"{base_url}/predict"
    payload = {"siren": "123456789"}
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    result = response.json()
    print("Single Prediction Result:")
    print(json.dumps(result, indent=2))
    return result


def example_batch_prediction(base_url: str = "http://localhost:8000"):
    """Example: Batch prediction request."""
    url = f"{base_url}/predict/batch"
    payload = {"sirens": ["123456789", "987654321", "555555555"]}
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    result = response.json()
    print("Batch Prediction Result:")
    print(json.dumps(result, indent=2))
    return result


def example_health_check(base_url: str = "http://localhost:8000"):
    """Example: Health check."""
    url = f"{base_url}/health"
    
    response = requests.get(url)
    response.raise_for_status()
    
    result = response.json()
    print("Health Check Result:")
    print(json.dumps(result, indent=2))
    return result


def example_model_info(base_url: str = "http://localhost:8000"):
    """Example: Model information."""
    url = f"{base_url}/info"
    
    response = requests.get(url)
    response.raise_for_status()
    
    result = response.json()
    print("Model Info:")
    print(f"Model Type: {result['model_type']}")
    print(f"Feature Count: {result['feature_count']}")
    print(f"Model Suffix: {result['model_suffix']}")
    print(f"First 10 Features: {result['features'][:10]}")
    return result


if __name__ == "__main__":
    # Make sure the API is running before running these examples
    print("OSE v3 API Examples")
    print("=" * 50)
    
    try:
        # Health check
        example_health_check()
        print("\n")
        
        # Model info
        example_model_info()
        print("\n")
        
        # Single prediction (replace with actual SIREN from your data)
        # example_single_prediction()
        # print("\n")
        
        # Batch prediction
        # example_batch_prediction()
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Make sure it's running on http://localhost:8000")
        print("Start the API with: python run_api.py")
    except Exception as e:
        print(f"Error: {e}")

