#!/usr/bin/env python3
"""
Test script to verify PDF endpoint functionality.
Run this after starting the FastAPI server to test the PDF serving.
"""

import requests
import sys
from pathlib import Path

def test_pdf_endpoint():
    """Test the PDF endpoint to ensure it's working correctly."""
    base_url = "http://localhost:8080"
    pdf_endpoint = f"{base_url}/assignment-pdf"
    
    print("Testing PDF endpoint...")
    print(f"URL: {pdf_endpoint}")
    
    try:
        # Test the endpoint
        response = requests.get(pdf_endpoint, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type', 'Not set')}")
        print(f"Content-Length: {response.headers.get('content-length', 'Not set')}")
        print(f"Content-Disposition: {response.headers.get('content-disposition', 'Not set')}")
        
        if response.status_code == 200:
            print("✅ PDF endpoint is working correctly!")
            print(f"PDF size: {len(response.content)} bytes")
            
            # Check if it's actually a PDF
            if response.content.startswith(b'%PDF'):
                print("✅ Response is a valid PDF file")
            else:
                print("❌ Response doesn't appear to be a PDF file")
                print(f"First 100 bytes: {response.content[:100]}")
                
        else:
            print(f"❌ PDF endpoint returned status {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the server. Make sure the FastAPI server is running on port 8080")
        return False
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        return False
    except Exception as e:
        print(f"❌ Error testing PDF endpoint: {e}")
        return False
    
    return response.status_code == 200

def test_health_endpoint():
    """Test the health endpoint to ensure the server is running."""
    base_url = "http://localhost:8080"
    health_endpoint = f"{base_url}/health"
    
    print("\nTesting health endpoint...")
    print(f"URL: {health_endpoint}")
    
    try:
        response = requests.get(health_endpoint, timeout=5)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Health endpoint is working")
            return True
        else:
            print(f"❌ Health endpoint returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the server. Make sure the FastAPI server is running on port 8080")
        return False
    except Exception as e:
        print(f"❌ Error testing health endpoint: {e}")
        return False

if __name__ == "__main__":
    print("PDF Endpoint Test Script")
    print("=" * 50)
    
    # Test health first
    health_ok = test_health_endpoint()
    
    if health_ok:
        # Test PDF endpoint
        pdf_ok = test_pdf_endpoint()
        
        if pdf_ok:
            print("\n🎉 All tests passed! PDF functionality should work correctly.")
            sys.exit(0)
        else:
            print("\n❌ PDF endpoint test failed.")
            sys.exit(1)
    else:
        print("\n❌ Server is not running or not accessible.")
        print("Please start the FastAPI server first:")
        print("python -m src.enterprise_llmops.main --host 0.0.0.0 --port 8080")
        sys.exit(1)
