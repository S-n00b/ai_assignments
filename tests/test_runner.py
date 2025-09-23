#!/usr/bin/env python3
"""
Comprehensive test runner for the AI Assignments test suite.

This script provides a unified interface for running all types of tests:
- Unit tests
- Integration tests  
- End-to-end tests
- GitHub Pages frontend tests
- Platform architecture tests
- Service integration tests

Usage:
    python tests/test_runner.py --help
    python tests/test_runner.py --unit
    python tests/test_runner.py --integration
    python tests/test_runner.py --e2e
    python tests/test_runner.py --all
    python tests/test_runner.py --phase7
    python tests/test_runner.py --github-pages
"""

import argparse
import sys
import subprocess
import os
from pathlib import Path
from datetime import datetime
import json
from typing import List, Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestRunner:
    """Comprehensive test runner for all test types."""
    
    def __init__(self):
        self.project_root = project_root
        self.tests_dir = self.project_root / "tests"
        self.src_dir = self.project_root / "src"
        self.venv_dir = self.project_root / "venv"
        self.results_dir = self.project_root / "test_results"
        
        # Ensure results directory exists
        self.results_dir.mkdir(exist_ok=True)
        
        # Test configurations
        self.test_configs = {
            "unit": {
                "path": "tests/unit",
                "markers": ["unit"],
                "description": "Unit tests for individual components",
                "timeout": 300
            },
            "integration": {
                "path": "tests/integration", 
                "markers": ["integration"],
                "description": "Integration tests for module interactions",
                "timeout": 600
            },
            "e2e": {
                "path": "tests/e2e",
                "markers": ["e2e"],
                "description": "End-to-end tests for complete workflows", 
                "timeout": 900
            },
            "github_pages": {
                "path": "tests/e2e/test_github_pages_frontend_integration.py",
                "markers": ["github_pages", "frontend"],
                "description": "GitHub Pages frontend integration tests",
                "timeout": 600
            },
            "phase7": {
                "path": "tests/e2e/test_phase7_complete_demonstration.py",
                "markers": ["phase7", "e2e"],
                "description": "Phase 7 demonstration tests",
                "timeout": 1200
            },
            "platform_architecture": {
                "path": "tests/integration/test_platform_architecture_layers.py",
                "markers": ["platform_architecture", "integration"],
                "description": "Platform architecture layer tests",
                "timeout": 600
            },
            "service_integration": {
                "path": "tests/integration/test_service_level_interactions.py", 
                "markers": ["service_integration", "integration"],
                "description": "Service-level integration tests",
                "timeout": 600
            }
        }
    
    def activate_virtual_environment(self):
        """Activate the virtual environment."""
        if os.name == 'nt':  # Windows
            activate_script = self.venv_dir / "Scripts" / "activate.bat"
            python_exe = self.venv_dir / "Scripts" / "python.exe"
        else:  # Unix/Linux/MacOS
            activate_script = self.venv_dir / "bin" / "activate"
            python_exe = self.venv_dir / "bin" / "python"
        
        if not python_exe.exists():
            print(f"Virtual environment not found at {self.venv_dir}")
            print("Please run: python -m venv venv")
            sys.exit(1)
        
        return python_exe
    
    def run_pytest_command(self, 
                          test_path: str, 
                          markers: List[str] = None,
                          timeout: int = 300,
                          verbose: bool = True,
                          coverage: bool = False,
                          parallel: bool = False) -> Dict[str, Any]:
        """Run pytest with specified parameters."""
        python_exe = self.activate_virtual_environment()
        
        # Build pytest command
        cmd = [str(python_exe), "-m", "pytest"]
        
        # Add test path
        cmd.append(test_path)
        
        # Add markers
        if markers:
            for marker in markers:
                cmd.extend(["-m", marker])
        
        # Add timeout
        cmd.extend(["--timeout", str(timeout)])
        
        # Add verbose output
        if verbose:
            cmd.append("-v")
        
        # Add coverage
        if coverage:
            cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term-missing"])
        
        # Add parallel execution
        if parallel:
            cmd.extend(["-n", "auto"])
        
        # Add additional pytest options
        cmd.extend([
            "--tb=short",
            "--strict-markers",
            "--disable-warnings",
            "--color=yes",
            "--durations=10"
        ])
        
        # Run command
        start_time = datetime.now()
        print(f"Running command: {' '.join(cmd)}")
        print("-" * 80)
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout + 60  # Add buffer for pytest timeout
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": duration,
                "command": " ".join(cmd)
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": f"Test timed out after {timeout + 60} seconds",
                "duration": timeout + 60,
                "command": " ".join(cmd)
            }
    
    def run_test_suite(self, 
                      test_type: str, 
                      verbose: bool = True,
                      coverage: bool = False,
                      parallel: bool = False) -> Dict[str, Any]:
        """Run a specific test suite."""
        if test_type not in self.test_configs:
            raise ValueError(f"Unknown test type: {test_type}")
        
        config = self.test_configs[test_type]
        print(f"\n{'='*80}")
        print(f"Running {test_type.upper()} tests")
        print(f"Description: {config['description']}")
        print(f"Path: {config['path']}")
        print(f"Markers: {config['markers']}")
        print(f"Timeout: {config['timeout']} seconds")
        print(f"{'='*80}")
        
        result = self.run_pytest_command(
            test_path=config["path"],
            markers=config["markers"],
            timeout=config["timeout"],
            verbose=verbose,
            coverage=coverage,
            parallel=parallel
        )
        
        # Save results
        self.save_test_results(test_type, result)
        
        return result
    
    def run_all_tests(self, 
                     verbose: bool = True,
                     coverage: bool = False,
                     parallel: bool = False) -> Dict[str, Any]:
        """Run all test suites."""
        print(f"\n{'='*80}")
        print("Running ALL test suites")
        print(f"{'='*80}")
        
        all_results = {}
        overall_success = True
        
        # Run each test suite
        for test_type in self.test_configs.keys():
            print(f"\n--- Running {test_type} tests ---")
            result = self.run_test_suite(
                test_type=test_type,
                verbose=verbose,
                coverage=coverage,
                parallel=parallel
            )
            
            all_results[test_type] = result
            if not result["success"]:
                overall_success = False
            
            # Print summary for this test suite
            status = "✓ PASSED" if result["success"] else "✗ FAILED"
            print(f"{status} {test_type} tests completed in {result['duration']:.2f}s")
        
        # Overall summary
        print(f"\n{'='*80}")
        print("OVERALL TEST SUMMARY")
        print(f"{'='*80}")
        
        for test_type, result in all_results.items():
            status = "✓ PASSED" if result["success"] else "✗ FAILED"
            print(f"{status} {test_type:20} {result['duration']:8.2f}s")
        
        overall_status = "✓ ALL TESTS PASSED" if overall_success else "✗ SOME TESTS FAILED"
        print(f"\n{overall_status}")
        
        # Save overall results
        self.save_test_results("all", {
            "success": overall_success,
            "results": all_results,
            "timestamp": datetime.now().isoformat()
        })
        
        return all_results
    
    def run_phase7_demonstration_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run Phase 7 demonstration tests specifically."""
        print(f"\n{'='*80}")
        print("Running Phase 7 Demonstration Tests")
        print("Following the complete Phase 7 demonstration workflow")
        print(f"{'='*80}")
        
        # Run Phase 7 specific tests
        result = self.run_test_suite("phase7", verbose=verbose)
        
        # Also run related integration tests
        integration_result = self.run_test_suite("service_integration", verbose=verbose)
        architecture_result = self.run_test_suite("platform_architecture", verbose=verbose)
        
        phase7_results = {
            "phase7_demonstration": result,
            "service_integration": integration_result,
            "platform_architecture": architecture_result,
            "overall_success": result["success"] and integration_result["success"] and architecture_result["success"]
        }
        
        # Save Phase 7 results
        self.save_test_results("phase7_complete", phase7_results)
        
        return phase7_results
    
    def run_github_pages_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run GitHub Pages frontend tests specifically."""
        print(f"\n{'='*80}")
        print("Running GitHub Pages Frontend Tests")
        print("Testing local development and hosted deployment")
        print(f"{'='*80}")
        
        # Run GitHub Pages specific tests
        result = self.run_test_suite("github_pages", verbose=verbose)
        
        # Also run related frontend tests
        frontend_result = self.run_test_suite("e2e", verbose=verbose)
        
        github_pages_results = {
            "github_pages": result,
            "frontend_integration": frontend_result,
            "overall_success": result["success"] and frontend_result["success"]
        }
        
        # Save GitHub Pages results
        self.save_test_results("github_pages_complete", github_pages_results)
        
        return github_pages_results
    
    def save_test_results(self, test_type: str, results: Dict[str, Any]):
        """Save test results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{test_type}_test_results_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Prepare results for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, datetime):
                json_results[key] = value.isoformat()
            elif isinstance(value, Path):
                json_results[key] = str(value)
            else:
                json_results[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Test results saved to: {filepath}")
    
    def print_help(self):
        """Print help information."""
        print("AI Assignments Test Runner")
        print("=" * 50)
        print()
        print("Available test suites:")
        print()
        
        for test_type, config in self.test_configs.items():
            print(f"  {test_type:20} - {config['description']}")
            print(f"  {'':20}   Path: {config['path']}")
            print(f"  {'':20}   Markers: {', '.join(config['markers'])}")
            print(f"  {'':20}   Timeout: {config['timeout']}s")
            print()
        
        print("Usage examples:")
        print("  python tests/test_runner.py --unit")
        print("  python tests/test_runner.py --integration")
        print("  python tests/test_runner.py --e2e")
        print("  python tests/test_runner.py --phase7")
        print("  python tests/test_runner.py --github-pages")
        print("  python tests/test_runner.py --all")
        print("  python tests/test_runner.py --all --coverage")
        print("  python tests/test_runner.py --all --parallel")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Assignments Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/test_runner.py --unit
  python tests/test_runner.py --integration --coverage
  python tests/test_runner.py --e2e --parallel
  python tests/test_runner.py --phase7
  python tests/test_runner.py --github-pages
  python tests/test_runner.py --all
  python tests/test_runner.py --all --coverage --parallel
        """
    )
    
    # Test suite options
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--e2e", action="store_true", help="Run end-to-end tests")
    parser.add_argument("--phase7", action="store_true", help="Run Phase 7 demonstration tests")
    parser.add_argument("--github-pages", action="store_true", help="Run GitHub Pages frontend tests")
    parser.add_argument("--platform-architecture", action="store_true", help="Run platform architecture tests")
    parser.add_argument("--service-integration", action="store_true", help="Run service integration tests")
    parser.add_argument("--all", action="store_true", help="Run all test suites")
    
    # Execution options
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    parser.add_argument("--quiet", action="store_true", help="Quiet output")
    
    # Other options
    parser.add_argument("--help-detailed", action="store_true", help="Show detailed help")
    
    args = parser.parse_args()
    
    # Handle help
    if args.help_detailed or (not any([
        args.unit, args.integration, args.e2e, args.phase7, 
        args.github_pages, args.platform_architecture, 
        args.service_integration, args.all
    ])):
        runner = TestRunner()
        runner.print_help()
        return
    
    # Set verbose mode
    verbose = args.verbose and not args.quiet
    
    # Create test runner
    runner = TestRunner()
    
    # Run requested tests
    if args.all:
        runner.run_all_tests(
            verbose=verbose,
            coverage=args.coverage,
            parallel=args.parallel
        )
    elif args.phase7:
        runner.run_phase7_demonstration_tests(verbose=verbose)
    elif args.github_pages:
        runner.run_github_pages_tests(verbose=verbose)
    else:
        # Run individual test suites
        if args.unit:
            runner.run_test_suite("unit", verbose=verbose, coverage=args.coverage, parallel=args.parallel)
        
        if args.integration:
            runner.run_test_suite("integration", verbose=verbose, coverage=args.coverage, parallel=args.parallel)
        
        if args.e2e:
            runner.run_test_suite("e2e", verbose=verbose, coverage=args.coverage, parallel=args.parallel)
        
        if args.platform_architecture:
            runner.run_test_suite("platform_architecture", verbose=verbose, coverage=args.coverage, parallel=args.parallel)
        
        if args.service_integration:
            runner.run_test_suite("service_integration", verbose=verbose, coverage=args.coverage, parallel=args.parallel)


if __name__ == "__main__":
    main()
