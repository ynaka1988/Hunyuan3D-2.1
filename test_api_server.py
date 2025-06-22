#!/usr/bin/env python3
"""
Test script to demonstrate the API documentation features of the Hunyuan3D API server.

This script shows how to use the API endpoints with proper parameter documentation.
"""

import requests
import base64
import json
from PIL import Image
import io
import time

# API server URL (adjust as needed)
API_BASE_URL = "http://localhost:8081"

def create_test_image():
    """Create a simple test image for API testing"""
    # Create a simple 256x256 test image
    img = Image.new('RGB', (256, 256), color='red')
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return img_base64

def test_generation_request():
    """Test the generation request with simplified parameters"""
    print("Creating test image...")
    # Create test image
    test_image = create_test_image()
    
    # Simplified request payload with only the parameters the worker actually uses
    request_data = {
        "image": test_image,
        "type": "glb"
    }
    
    print("Testing /generate endpoint...")
    print("Request parameters:")
    for key, value in request_data.items():
        if key == "image":
            print(f"  {key}: [base64 encoded image data]")
        else:
            print(f"  {key}: {value}")
    
    try:
        response = requests.post(f"{API_BASE_URL}/generate", json=request_data)
        print(f"Response status: {response.status_code}")
        if response.status_code == 200:
            print("Success! Generated 3D model file received.")
        else:
            print(f"Error: {response.text}")
    except requests.exceptions.ConnectionError:
        print("Could not connect to API server. Make sure it's running on localhost:8081")

def test_async_generation():
    """Test the asynchronous generation endpoint"""
    
    test_image = create_test_image()
    
    request_data = {
        "image": test_image,
        "type": "glb"
    }
    
    print("\nTesting /send endpoint (async)...")
    try:
        response = requests.post(f"{API_BASE_URL}/send", json=request_data)
        print(f"Response status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            uid = result.get("uid")
            print(f"Task ID: {uid}")
            
            # Check status
            print("Checking task status...")
            status_response = requests.get(f"{API_BASE_URL}/status/{uid}")
            print(f"Status: {status_response.json()}")
            # Poll status until completed
            while True:
                status_response = requests.get(f"{API_BASE_URL}/status/{uid}")
                status_data = status_response.json()
                print(f"Status: {status_data['status']}")
                
                if status_data['status'] == 'completed':
                    print("Generation completed!")
                    print("Model data received in base64 format")
                    break
                elif status_data['status'] == 'error':
                    print(f"Error: {status_data.get('message', 'Unknown error')}")
                    break
                    
                time.sleep(2)  # Wait 2 seconds between checks
        else:
            print(f"Error: {response.text}")
    except requests.exceptions.ConnectionError:
        print("Could not connect to API server.")

def test_health_check():
    """Test the health check endpoint"""
    
    print("\nTesting /health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Response status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Health: {result}")
        else:
            print(f"Error: {response.text}")
    except requests.exceptions.ConnectionError:
        print("Could not connect to API server.")

def show_api_documentation_info():
    """Show information about the API documentation"""
    
    print("=" * 60)
    print("HUNYUAN3D API DOCUMENTATION")
    print("=" * 60)
    print()
    print("The API server now includes comprehensive documentation:")
    print()
    print("1. Pydantic Models:")
    print("   - GenerationRequest: Documents all input parameters")
    print("   - GenerationResponse: Documents response format")
    print("   - StatusResponse: Documents status endpoint response")
    print("   - HealthResponse: Documents health check response")
    print()
    print("2. Parameter Documentation:")
    print("   - All parameters have descriptions and examples")
    print("   - Parameter types and constraints are defined")
    print("   - Default values are specified")
    print("   - Note: Only 'image' and 'type' parameters are currently used")
    print()
    print("3. API Organization:")
    print("   - Endpoints are tagged (generation, status)")
    print("   - Comprehensive descriptions for each endpoint")
    print("   - Example requests and responses")
    print()
    print("4. Access Documentation:")
    print(f"   - Interactive docs: {API_BASE_URL}/docs")
    print(f"   - Alternative docs: {API_BASE_URL}/redoc")
    print()
    print("5. Available Endpoints:")
    print("   - POST /generate - Immediate 3D generation")
    print("   - POST /send - Async 3D generation")
    print("   - GET /status/{uid} - Check task status")
    print("   - GET /health - Service health check")
    print()
    print("6. Simplified Parameters:")
    print("   - image: Base64 encoded input image (required)")
    print("   - type: Output file format - 'glb' or 'obj' (optional, default: 'glb')")
    print()

if __name__ == "__main__":
    show_api_documentation_info()
    
    # Run tests if server is available
    test_health_check()
    #test_generation_request()
    test_async_generation()
    
    print("\n" + "=" * 60)
    print("To view the interactive API documentation:")
    print(f"1. Start the API server: python api_server.py")
    print(f"2. Open your browser to: {API_BASE_URL}/docs")
    print("3. Explore the documented endpoints and parameters")
    print("=" * 60) 