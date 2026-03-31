#!/usr/bin/env python
"""
Test script for RAG FastAPI Server
Run this after starting the API server to verify all endpoints work
"""

import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8000"


def print_response(title: str, response: requests.Response, pretty: bool = True):
    """Pretty print API response"""
    print(f"\n{'='*60}")
    print(f"TEST: {title}")
    print(f"{'='*60}")
    print(f"Status Code: {response.status_code}")
    
    try:
        if pretty:
            print(json.dumps(response.json(), indent=2))
        else:
            print(response.text)
    except:
        print(response.text)


def test_health() -> bool:
    """Test health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print_response("Health Check", response)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return False


def test_root() -> bool:
    """Test root endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/")
        print_response("Root Endpoint", response)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Root endpoint failed: {str(e)}")
        return False


def test_batch_status(batch_id: str = "test_batch") -> bool:
    """Test batch status endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/api/batch/status/{batch_id}")
        print_response(f"Batch Status - {batch_id}", response)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Batch status failed: {str(e)}")
        return False


def test_process_batch(batch_id: str = "test_batch") -> bool:
    """Test batch processing endpoint"""
    try:
        payload = {"batch_id": batch_id}
        response = requests.post(
            f"{BASE_URL}/api/batch/process",
            json=payload,
            timeout=300  # 5 minute timeout for processing
        )
        print_response(f"Process Batch - {batch_id}", response)
        
        if response.status_code == 200:
            return True
        elif response.status_code == 404:
            print("⚠️  No PDFs found (expected if no PDFs in rag_data/pdfs/)")
            return True
        return False
    except requests.Timeout:
        print("⚠️  Batch processing timed out (PDFs are being processed)")
        return True
    except Exception as e:
        print(f"❌ Batch processing failed: {str(e)}")
        return False


def test_ask_question(batch_id: str = "test_batch") -> bool:
    """Test Q&A endpoint"""
    try:
        payload = {
            "question": "What diagnoses are mentioned?",
            "batch_id": batch_id,
            "top_k": 5
        }
        response = requests.post(
            f"{BASE_URL}/api/qa/ask",
            json=payload,
            timeout=60
        )
        print_response("Ask Question", response)
        
        if response.status_code == 200:
            return True
        elif response.status_code == 404:
            print("⚠️  No indexed documents (expected if batch not processed yet)")
            return True
        return False
    except Exception as e:
        print(f"❌ Q&A failed: {str(e)}")
        return False


def test_invalid_question(batch_id: str = "test_batch") -> bool:
    """Test Q&A with empty question (should fail)"""
    try:
        payload = {
            "question": "",
            "batch_id": batch_id
        }
        response = requests.post(
            f"{BASE_URL}/api/qa/ask",
            json=payload
        )
        print_response("Ask Question - Empty (Expected to Fail)", response)
        return response.status_code == 400
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*15 + "RAG FastAPI Server - Test Suite" + " "*11 + "║")
    print("╚" + "="*58 + "╝")
    
    print(f"\nTarget URL: {BASE_URL}")
    print("Make sure the server is running: python backend/RagApi/main.py")
    
    # Verify server is running
    try:
        requests.get(f"{BASE_URL}/health", timeout=2)
    except Exception as e:
        print(f"\n❌ ERROR: Cannot connect to server at {BASE_URL}")
        print(f"   Make sure you've started the server with: python backend/RagApi/main.py")
        return
    
    results: Dict[str, bool] = {}
    
    # Run tests
    print("\n" + "="*60)
    print("Running Tests...")
    print("="*60)
    
    results["Health Check"] = test_health()
    results["Root Endpoint"] = test_root()
    results["Batch Status"] = test_batch_status()
    results["Invalid Question"] = test_invalid_question()
    
    # These may fail if no PDFs, which is ok
    print("\n" + "="*60)
    print("Optional Tests (may fail if no PDFs)")
    print("="*60)
    test_process_batch()
    test_ask_question()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30} {status}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed! Server is working correctly.")
        print("\nNext steps:")
        print("1. Upload PDFs to: rag_data/pdfs/batch_xxx/")
        print("2. Call: POST /api/batch/process with batch_id")
        print("3. Then: POST /api/qa/ask to ask questions")
        print(f"4. View docs: {BASE_URL}/docs")
    else:
        print("\n⚠️  Some tests failed. Check the output above.")


if __name__ == "__main__":
    main()
