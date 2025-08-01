#!/usr/bin/env python3
"""
Quick verification script to test all implemented features.
"""

import os
import sys
import requests
import subprocess
import time
import pandas as pd

def test_dataset_balancing():
    """Test the dataset balancing utility."""
    print("🧪 Testing dataset balancing utility...")
    result = subprocess.run([
        'python', 'data/balance_dataset.py', 
        '--input', 'data/tox21_sr-p53.csv', 
        '--method', 'weights'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Dataset balancing utility works correctly")
        return True
    else:
        print(f"❌ Dataset balancing failed: {result.stderr}")
        return False

def test_model_training():
    """Test model training with class weights."""
    print("🧪 Testing model training with class weights...")
    result = subprocess.run([
        'python', 'backend/train_gnn.py', 
        '--epochs', '2'
    ], capture_output=True, text=True)
    
    if result.returncode == 0 and 'Model saved' in result.stdout:
        print("✅ Model training works correctly")
        return True
    else:
        print(f"❌ Model training failed: {result.stderr}")
        return False

def test_model_evaluation():
    """Test comprehensive model evaluation."""
    print("🧪 Testing comprehensive model evaluation...")
    result = subprocess.run([
        'python', 'backend/eval_gnn.py'
    ], capture_output=True, text=True)
    
    if result.returncode == 0 and 'COMPREHENSIVE MODEL EVALUATION' in result.stdout:
        print("✅ Model evaluation works correctly")
        return True
    else:
        print(f"❌ Model evaluation failed: {result.stderr}")
        return False

def test_api_functionality():
    """Test API functionality."""
    print("🧪 Testing API functionality...")
    
    # Start API in background
    import subprocess
    import time
    
    api_process = subprocess.Popen([
        'python', 'backend/gnn_api_advanced.py'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    time.sleep(5)  # Wait for API to start
    
    try:
        # Test health endpoint
        response = requests.get('http://localhost:8001/health', timeout=5)
        if response.status_code == 200:
            print("✅ API health check works")
        else:
            print("❌ API health check failed")
            return False
        
        # Test prediction endpoint
        response = requests.post(
            'http://localhost:8001/predict',
            json={'smiles': 'CCO'},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'toxicity' in data and 'prediction' in data:
                print("✅ API prediction works correctly")
                return True
            else:
                print("❌ API prediction response malformed")
                return False
        else:
            print(f"❌ API prediction failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False
    finally:
        # Cleanup
        api_process.terminate()
        api_process.wait()

def test_file_structure():
    """Test that all required files are present."""
    print("🧪 Testing file structure...")
    
    required_files = [
        'README.md',
        'requirements.txt',
        'docker-compose.yml',
        'Dockerfile.backend',
        'Dockerfile.frontend',
        'backend/train_gnn.py',
        'backend/eval_gnn.py',
        'backend/gnn_api.py',
        'backend/gnn_api_advanced.py',
        'data/balance_dataset.py',
        'data/tox21_sr-p53.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files are present")
        return True

def main():
    """Run all tests."""
    print("🚀 Starting comprehensive feature verification...\n")
    
    # Change to project directory
    os.chdir('/home/runner/work/Test/Test')
    
    tests = [
        test_file_structure,
        test_dataset_balancing,
        test_model_training,
        test_model_evaluation,
        test_api_functionality
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
        print()  # Add spacing
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("📊 VERIFICATION SUMMARY")
    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! Implementation is complete.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())