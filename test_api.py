"""
Test script for the Baby Cry Analyzer API.
"""

import requests
import os
import time
import json
from typing import Dict, Any


class BabyCryAnalyzerTester:
    """Test client for the Baby Cry Analyzer API."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        """
        Initialize the tester.
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health_endpoint(self) -> Dict[str, Any]:
        """Test the health endpoint."""
        print("Testing /health endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            result = {
                'status_code': response.status_code,
                'response': response.json() if response.status_code == 200 else response.text,
                'success': response.status_code == 200
            }
            print(f"Health check: {'✓ PASSED' if result['success'] else '✗ FAILED'}")
            return result
        except requests.exceptions.RequestException as e:
            print(f"Health check: ✗ FAILED - {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def test_categories_endpoint(self) -> Dict[str, Any]:
        """Test the categories endpoint."""
        print("Testing /categories endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/categories")
            result = {
                'status_code': response.status_code,
                'response': response.json() if response.status_code == 200 else response.text,
                'success': response.status_code == 200
            }
            print(f"Categories: {'✓ PASSED' if result['success'] else '✗ FAILED'}")
            return result
        except requests.exceptions.RequestException as e:
            print(f"Categories: ✗ FAILED - {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def test_predict_endpoint(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Test the predict endpoint with an audio file.
        
        Args:
            audio_file_path: Path to the audio file to test
            
        Returns:
            Test results
        """
        print(f"Testing /predict endpoint with {audio_file_path}...")
        
        if not os.path.exists(audio_file_path):
            print(f"Predict: ✗ FAILED - File not found: {audio_file_path}")
            return {'success': False, 'error': 'File not found'}
        
        try:
            with open(audio_file_path, 'rb') as audio_file:
                files = {'audio': audio_file}
                response = self.session.post(f"{self.base_url}/predict", files=files)
            
            result = {
                'status_code': response.status_code,
                'success': response.status_code == 200
            }
            
            if response.status_code == 200:
                result['response'] = response.json()
                print(f"Predict: ✓ PASSED")
                print(f"  Predicted need: {result['response']['prediction']['predicted_need']}")
                print(f"  Confidence: {result['response']['prediction']['confidence']:.3f}")
            else:
                result['response'] = response.text
                print(f"Predict: ✗ FAILED - Status {response.status_code}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"Predict: ✗ FAILED - {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def test_invalid_file(self) -> Dict[str, Any]:
        """Test the predict endpoint with an invalid file."""
        print("Testing /predict endpoint with invalid file...")
        
        try:
            # Create a dummy text file
            dummy_content = b"This is not an audio file"
            files = {'audio': ('test.txt', dummy_content, 'text/plain')}
            response = self.session.post(f"{self.base_url}/predict", files=files)
            
            result = {
                'status_code': response.status_code,
                'response': response.json() if response.headers.get('content-type') == 'application/json' else response.text,
                'success': response.status_code == 400  # Should return 400 for invalid file
            }
            
            print(f"Invalid file test: {'✓ PASSED' if result['success'] else '✗ FAILED'}")
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"Invalid file test: ✗ FAILED - {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def test_no_file(self) -> Dict[str, Any]:
        """Test the predict endpoint without a file."""
        print("Testing /predict endpoint without file...")
        
        try:
            response = self.session.post(f"{self.base_url}/predict")
            
            result = {
                'status_code': response.status_code,
                'response': response.json() if response.headers.get('content-type') == 'application/json' else response.text,
                'success': response.status_code == 400  # Should return 400 for no file
            }
            
            print(f"No file test: {'✓ PASSED' if result['success'] else '✗ FAILED'}")
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"No file test: ✗ FAILED - {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def run_all_tests(self, audio_file_path: str = None) -> Dict[str, Any]:
        """
        Run all tests.
        
        Args:
            audio_file_path: Optional path to an audio file for testing
            
        Returns:
            Complete test results
        """
        print("=" * 50)
        print("Baby Cry Analyzer API Test Suite")
        print("=" * 50)
        
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'base_url': self.base_url,
            'tests': {}
        }
        
        # Test health endpoint
        results['tests']['health'] = self.test_health_endpoint()
        
        # Test categories endpoint
        results['tests']['categories'] = self.test_categories_endpoint()
        
        # Test predict endpoint with valid file (if provided)
        if audio_file_path:
            results['tests']['predict_valid'] = self.test_predict_endpoint(audio_file_path)
        
        # Test predict endpoint with invalid file
        results['tests']['predict_invalid'] = self.test_invalid_file()
        
        # Test predict endpoint without file
        results['tests']['predict_no_file'] = self.test_no_file()
        
        # Summary
        total_tests = len(results['tests'])
        passed_tests = sum(1 for test in results['tests'].values() if test.get('success', False))
        
        print("\n" + "=" * 50)
        print(f"Test Summary: {passed_tests}/{total_tests} tests passed")
        print("=" * 50)
        
        results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_file: str = "test_results.json"):
        """Save test results to a file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Test results saved to {output_file}")


def main():
    """Main function to run tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the Baby Cry Analyzer API")
    parser.add_argument('--url', default='http://localhost:5000', help='API base URL')
    parser.add_argument('--audio', help='Path to audio file for testing')
    parser.add_argument('--output', default='test_results.json', help='Output file for results')
    
    args = parser.parse_args()
    
    tester = BabyCryAnalyzerTester(args.url)
    results = tester.run_all_tests(args.audio)
    tester.save_results(results, args.output)


if __name__ == "__main__":
    main()
