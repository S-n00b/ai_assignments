#!/usr/bin/env python3
"""
Comprehensive Test Execution Script

This script provides a comprehensive test execution framework for the GitHub Pages
frontend, platform architecture layers, and Phase 7 demonstration flow.

Features:
- Unit, integration, and end-to-end testing
- Service health validation
- Test coverage reporting
- Parallel test execution
- Custom test filtering
"""

import argparse
import subprocess
import sys
import time
import requests
from pathlib import Path
from typing import List, Dict, Optional
import json


class TestExecutor:
    """Comprehensive test execution framework."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.services = {
            "fastapi": {"port": 8080, "health_url": "/health"},
            "gradio": {"port": 7860, "health_url": "/"},
            "mlflow": {"port": 5000, "health_url": "/"},
            "chromadb": {"port": 8000, "health_url": "/"}
        }
    
    def check_service_health(self, service: str) -> bool:
        """Check if a service is running and healthy."""
        if service not in self.services:
            return False
        
        try:
            port = self.services[service]["port"]
            health_url = self.services[service]["health_url"]
            response = requests.get(f"http://localhost:{port}{health_url}", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def validate_services(self) -> Dict[str, bool]:
        """Validate all required services are running."""
        print("🔍 Checking service health...")
        service_status = {}
        
        for service in self.services:
            is_healthy = self.check_service_health(service)
            service_status[service] = is_healthy
            status_icon = "✅" if is_healthy else "❌"
            print(f"  {status_icon} {service.capitalize()}: {'Healthy' if is_healthy else 'Not Running'}")
        
        return service_status
    
    def run_command(self, command: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        """Run a command and return the result."""
        if cwd is None:
            cwd = self.project_root
        
        print(f"🚀 Running: {' '.join(command)}")
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        return result
    
    def run_unit_tests(self, verbose: bool = True, coverage: bool = False) -> bool:
        """Run unit tests."""
        print("\n📋 Running Unit Tests...")
        
        command = ["python", "-m", "pytest", "tests/unit/"]
        if verbose:
            command.append("-v")
        if coverage:
            command.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
        
        result = self.run_command(command)
        
        if result.returncode == 0:
            print("✅ Unit tests passed")
            return True
        else:
            print("❌ Unit tests failed")
            print(f"Error: {result.stderr}")
            return False
    
    def run_integration_tests(self, verbose: bool = True, live_services: bool = False) -> bool:
        """Run integration tests."""
        print("\n🔗 Running Integration Tests...")
        
        command = ["python", "-m", "pytest", "tests/integration/"]
        if verbose:
            command.append("-v")
        if live_services:
            command.append("--live-services")
        
        result = self.run_command(command)
        
        if result.returncode == 0:
            print("✅ Integration tests passed")
            return True
        else:
            print("❌ Integration tests failed")
            print(f"Error: {result.stderr}")
            return False
    
    def run_e2e_tests(self, verbose: bool = True, production_urls: bool = False) -> bool:
        """Run end-to-end tests."""
        print("\n🌐 Running End-to-End Tests...")
        
        command = ["python", "-m", "pytest", "tests/e2e/"]
        if verbose:
            command.append("-v")
        if production_urls:
            command.append("--production-urls")
        
        result = self.run_command(command)
        
        if result.returncode == 0:
            print("✅ End-to-end tests passed")
            return True
        else:
            print("❌ End-to-end tests failed")
            print(f"Error: {result.stderr}")
            return False
    
    def run_phase7_tests(self, verbose: bool = True) -> bool:
        """Run Phase 7 demonstration tests."""
        print("\n🎯 Running Phase 7 Demonstration Tests...")
        
        command = ["python", "-m", "pytest", "tests/e2e/test_phase7_complete_demonstration.py"]
        if verbose:
            command.append("-v")
        
        result = self.run_command(command)
        
        if result.returncode == 0:
            print("✅ Phase 7 demonstration tests passed")
            return True
        else:
            print("❌ Phase 7 demonstration tests failed")
            print(f"Error: {result.stderr}")
            return False
    
    def run_github_pages_tests(self, verbose: bool = True, hosted: bool = False) -> bool:
        """Run GitHub Pages frontend tests."""
        print("\n📄 Running GitHub Pages Tests...")
        
        if hosted:
            command = ["python", "-m", "pytest", "tests/e2e/test_github_pages_frontend_integration.py", "--production-urls"]
        else:
            command = ["python", "-m", "pytest", "tests/unit/test_github_pages_integration.py"]
        
        if verbose:
            command.append("-v")
        
        result = self.run_command(command)
        
        if result.returncode == 0:
            print("✅ GitHub Pages tests passed")
            return True
        else:
            print("❌ GitHub Pages tests failed")
            print(f"Error: {result.stderr}")
            return False
    
    def run_comprehensive_suite(self, verbose: bool = True, parallel: bool = False, coverage: bool = False) -> Dict[str, bool]:
        """Run the complete test suite."""
        print("\n🎯 Running Comprehensive Test Suite...")
        
        results = {}
        
        # Run tests in sequence
        results["unit"] = self.run_unit_tests(verbose=verbose, coverage=coverage)
        results["integration"] = self.run_integration_tests(verbose=verbose, live_services=True)
        results["e2e"] = self.run_e2e_tests(verbose=verbose)
        results["phase7"] = self.run_phase7_tests(verbose=verbose)
        results["github_pages"] = self.run_github_pages_tests(verbose=verbose)
        
        return results
    
    def generate_report(self, results: Dict[str, bool]) -> None:
        """Generate a test execution report."""
        print("\n📊 Test Execution Report")
        print("=" * 50)
        
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result)
        
        for test_category, passed in results.items():
            status_icon = "✅" if passed else "❌"
            print(f"{status_icon} {test_category.replace('_', ' ').title()}: {'PASSED' if passed else 'FAILED'}")
        
        print(f"\nOverall: {passed_tests}/{total_tests} test categories passed")
        
        if passed_tests == total_tests:
            print("🎉 All tests passed successfully!")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
    
    def activate_virtual_environment(self) -> bool:
        """Activate the virtual environment."""
        venv_script = self.project_root / "venv" / "Scripts" / "Activate.ps1"
        
        if not venv_script.exists():
            print("❌ Virtual environment not found. Please create it first.")
            return False
        
        print("✅ Virtual environment found")
        return True


def main():
    """Main entry point for the test execution script."""
    parser = argparse.ArgumentParser(description="Comprehensive Test Execution Script")
    parser.add_argument("--test-type", choices=["unit", "integration", "e2e", "phase7", "github-pages", "all"], 
                       default="all", help="Type of tests to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--parallel", "-p", action="store_true", help="Run tests in parallel")
    parser.add_argument("--live-services", action="store_true", help="Use live services for testing")
    parser.add_argument("--production-urls", action="store_true", help="Use production URLs for testing")
    parser.add_argument("--skip-health-check", action="store_true", help="Skip service health validation")
    
    args = parser.parse_args()
    
    # Initialize test executor
    project_root = Path(__file__).parent.parent
    executor = TestExecutor(project_root)
    
    # Check virtual environment
    if not executor.activate_virtual_environment():
        sys.exit(1)
    
    # Validate services unless skipped
    if not args.skip_health_check:
        service_status = executor.validate_services()
        if not any(service_status.values()):
            print("⚠️  No services are running. Some tests may fail.")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)
    
    # Run tests based on selection
    results = {}
    
    if args.test_type == "unit":
        results["unit"] = executor.run_unit_tests(verbose=args.verbose, coverage=args.coverage)
    elif args.test_type == "integration":
        results["integration"] = executor.run_integration_tests(verbose=args.verbose, live_services=args.live_services)
    elif args.test_type == "e2e":
        results["e2e"] = executor.run_e2e_tests(verbose=args.verbose, production_urls=args.production_urls)
    elif args.test_type == "phase7":
        results["phase7"] = executor.run_phase7_tests(verbose=args.verbose)
    elif args.test_type == "github-pages":
        results["github_pages"] = executor.run_github_pages_tests(verbose=args.verbose, hosted=args.production_urls)
    elif args.test_type == "all":
        results = executor.run_comprehensive_suite(verbose=args.verbose, parallel=args.parallel, coverage=args.coverage)
    
    # Generate report
    executor.generate_report(results)
    
    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
