#!/usr/bin/env python3
"""
Documentation Site Functionality and Responsiveness Testing

This script performs comprehensive testing of the Jekyll documentation site
to ensure all features work correctly and the site is responsive.
"""

import os
import sys
import subprocess
import json
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
import yaml
import re
from urllib.parse import urljoin, urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SiteTester:
    """Comprehensive testing for the documentation site."""
    
    def __init__(self, site_url: str = "https://samne.github.io/ai_assignments"):
        self.site_url = site_url
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'site_url': site_url,
            'tests': {},
            'summary': {
                'total_tests': 0,
                'passed_tests': 0,
                'failed_tests': 0,
                'warnings': 0
            }
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return results."""
        logger.info("Starting comprehensive site testing...")
        
        # Test categories
        test_categories = [
            ('site_accessibility', self.test_site_accessibility),
            ('navigation', self.test_navigation),
            ('content_validation', self.test_content_validation),
            ('responsive_design', self.test_responsive_design),
            ('performance', self.test_performance),
            ('seo_validation', self.test_seo_validation),
            ('social_media', self.test_social_media_integration),
            ('search_functionality', self.test_search_functionality),
            ('comments_system', self.test_comments_system),
            ('analytics', self.test_analytics_integration),
            ('security', self.test_security_headers),
            ('accessibility', self.test_accessibility_compliance)
        ]
        
        for test_name, test_func in test_categories:
            try:
                logger.info(f"Running {test_name} tests...")
                result = test_func()
                self.test_results['tests'][test_name] = result
                self.update_summary(result)
            except Exception as e:
                logger.error(f"Error running {test_name} tests: {e}")
                self.test_results['tests'][test_name] = {
                    'status': 'error',
                    'message': str(e),
                    'tests': []
                }
        
        logger.info("Site testing completed.")
        return self.test_results
    
    def test_site_accessibility(self) -> Dict[str, Any]:
        """Test basic site accessibility."""
        tests = []
        
        # Test main page accessibility
        try:
            response = requests.get(self.site_url, timeout=10)
            tests.append({
                'name': 'Main page accessibility',
                'status': 'pass' if response.status_code == 200 else 'fail',
                'details': f"Status code: {response.status_code}",
                'response_time': response.elapsed.total_seconds()
            })
        except Exception as e:
            tests.append({
                'name': 'Main page accessibility',
                'status': 'fail',
                'details': f"Error: {str(e)}"
            })
        
        # Test HTTPS
        try:
            if self.site_url.startswith('https://'):
                response = requests.get(self.site_url, timeout=10, verify=True)
                tests.append({
                    'name': 'HTTPS accessibility',
                    'status': 'pass' if response.status_code == 200 else 'fail',
                    'details': f"HTTPS working: {response.status_code == 200}"
                })
        except Exception as e:
            tests.append({
                'name': 'HTTPS accessibility',
                'status': 'fail',
                'details': f"HTTPS error: {str(e)}"
            })
        
        return {
            'status': 'pass' if all(t['status'] == 'pass' for t in tests) else 'fail',
            'tests': tests
        }
    
    def test_navigation(self) -> Dict[str, Any]:
        """Test site navigation functionality."""
        tests = []
        
        # Test main navigation pages
        navigation_pages = [
            '/',
            '/about/',
            '/archives/',
            '/search/'
        ]
        
        for page in navigation_pages:
            try:
                url = urljoin(self.site_url, page)
                response = requests.get(url, timeout=10)
                tests.append({
                    'name': f'Navigation: {page}',
                    'status': 'pass' if response.status_code == 200 else 'fail',
                    'details': f"Status: {response.status_code}, Size: {len(response.content)} bytes"
                })
            except Exception as e:
                tests.append({
                    'name': f'Navigation: {page}',
                    'status': 'fail',
                    'details': f"Error: {str(e)}"
                })
        
        return {
            'status': 'pass' if all(t['status'] == 'pass' for t in tests) else 'fail',
            'tests': tests
        }
    
    def test_content_validation(self) -> Dict[str, Any]:
        """Test content validation and structure."""
        tests = []
        
        # Test main page content
        try:
            response = requests.get(self.site_url, timeout=10)
            content = response.text
            
            # Check for essential elements
            essential_elements = [
                ('title', '<title>'),
                ('meta description', '<meta name="description"'),
                ('favicon', '<link rel="icon"'),
                ('CSS', '<link rel="stylesheet"'),
                ('JavaScript', '<script'),
                ('navigation', '<nav'),
                ('footer', '<footer')
            ]
            
            for element_name, element_tag in essential_elements:
                if element_tag in content:
                    tests.append({
                        'name': f'Content: {element_name}',
                        'status': 'pass',
                        'details': f"Found {element_tag}"
                    })
                else:
                    tests.append({
                        'name': f'Content: {element_name}',
                        'status': 'fail',
                        'details': f"Missing {element_tag}"
                    })
            
            # Check for broken links
            broken_links = self.check_broken_links(content)
            tests.append({
                'name': 'Broken links check',
                'status': 'pass' if not broken_links else 'fail',
                'details': f"Found {len(broken_links)} broken links: {broken_links[:5]}"
            })
            
        except Exception as e:
            tests.append({
                'name': 'Content validation',
                'status': 'fail',
                'details': f"Error: {str(e)}"
            })
        
        return {
            'status': 'pass' if all(t['status'] == 'pass' for t in tests) else 'fail',
            'tests': tests
        }
    
    def test_responsive_design(self) -> Dict[str, Any]:
        """Test responsive design elements."""
        tests = []
        
        # Test viewport meta tag
        try:
            response = requests.get(self.site_url, timeout=10)
            content = response.text
            
            if 'viewport' in content:
                tests.append({
                    'name': 'Responsive: Viewport meta tag',
                    'status': 'pass',
                    'details': 'Viewport meta tag found'
                })
            else:
                tests.append({
                    'name': 'Responsive: Viewport meta tag',
                    'status': 'fail',
                    'details': 'Viewport meta tag missing'
                })
            
            # Check for responsive CSS
            if 'media=' in content or '@media' in content:
                tests.append({
                    'name': 'Responsive: CSS media queries',
                    'status': 'pass',
                    'details': 'Media queries found in CSS'
                })
            else:
                tests.append({
                    'name': 'Responsive: CSS media queries',
                    'status': 'warn',
                    'details': 'No media queries found'
                })
            
        except Exception as e:
            tests.append({
                'name': 'Responsive design test',
                'status': 'fail',
                'details': f"Error: {str(e)}"
            })
        
        return {
            'status': 'pass' if all(t['status'] in ['pass', 'warn'] for t in tests) else 'fail',
            'tests': tests
        }
    
    def test_performance(self) -> Dict[str, Any]:
        """Test site performance metrics."""
        tests = []
        
        try:
            response = requests.get(self.site_url, timeout=10)
            
            # Response time
            response_time = response.elapsed.total_seconds()
            tests.append({
                'name': 'Performance: Response time',
                'status': 'pass' if response_time < 3.0 else 'warn' if response_time < 5.0 else 'fail',
                'details': f"Response time: {response_time:.2f}s"
            })
            
            # Content size
            content_size = len(response.content)
            tests.append({
                'name': 'Performance: Content size',
                'status': 'pass' if content_size < 1024*1024 else 'warn' if content_size < 2*1024*1024 else 'fail',
                'details': f"Content size: {content_size/1024:.1f} KB"
            })
            
            # Check for compression
            if 'gzip' in response.headers.get('content-encoding', ''):
                tests.append({
                    'name': 'Performance: Compression',
                    'status': 'pass',
                    'details': 'Gzip compression enabled'
                })
            else:
                tests.append({
                    'name': 'Performance: Compression',
                    'status': 'warn',
                    'details': 'Gzip compression not detected'
                })
            
        except Exception as e:
            tests.append({
                'name': 'Performance test',
                'status': 'fail',
                'details': f"Error: {str(e)}"
            })
        
        return {
            'status': 'pass' if all(t['status'] in ['pass', 'warn'] for t in tests) else 'fail',
            'tests': tests
        }
    
    def test_seo_validation(self) -> Dict[str, Any]:
        """Test SEO elements and optimization."""
        tests = []
        
        try:
            response = requests.get(self.site_url, timeout=10)
            content = response.text
            
            # Check meta tags
            seo_elements = [
                ('title', '<title>'),
                ('meta description', '<meta name="description"'),
                ('meta keywords', '<meta name="keywords"'),
                ('canonical URL', '<link rel="canonical"'),
                ('Open Graph', '<meta property="og:'),
                ('Twitter Card', '<meta name="twitter:'),
                ('robots', '<meta name="robots"')
            ]
            
            for element_name, element_tag in seo_elements:
                if element_tag in content:
                    tests.append({
                        'name': f'SEO: {element_name}',
                        'status': 'pass',
                        'details': f"Found {element_tag}"
                    })
                else:
                    tests.append({
                        'name': f'SEO: {element_name}',
                        'status': 'warn',
                        'details': f"Missing {element_tag}"
                    })
            
            # Check for structured data
            if 'application/ld+json' in content:
                tests.append({
                    'name': 'SEO: Structured data',
                    'status': 'pass',
                    'details': 'JSON-LD structured data found'
                })
            else:
                tests.append({
                    'name': 'SEO: Structured data',
                    'status': 'warn',
                    'details': 'No structured data found'
                })
            
        except Exception as e:
            tests.append({
                'name': 'SEO validation',
                'status': 'fail',
                'details': f"Error: {str(e)}"
            })
        
        return {
            'status': 'pass' if all(t['status'] in ['pass', 'warn'] for t in tests) else 'fail',
            'tests': tests
        }
    
    def test_social_media_integration(self) -> Dict[str, Any]:
        """Test social media integration."""
        tests = []
        
        try:
            response = requests.get(self.site_url, timeout=10)
            content = response.text
            
            # Check social media meta tags
            social_elements = [
                ('Facebook Open Graph', '<meta property="og:'),
                ('Twitter Card', '<meta name="twitter:'),
                ('LinkedIn', 'linkedin.com'),
                ('Social sharing buttons', 'share')
            ]
            
            for element_name, element_tag in social_elements:
                if element_tag in content:
                    tests.append({
                        'name': f'Social: {element_name}',
                        'status': 'pass',
                        'details': f"Found {element_tag}"
                    })
                else:
                    tests.append({
                        'name': f'Social: {element_name}',
                        'status': 'warn',
                        'details': f"Missing {element_tag}"
                    })
            
        except Exception as e:
            tests.append({
                'name': 'Social media test',
                'status': 'fail',
                'details': f"Error: {str(e)}"
            })
        
        return {
            'status': 'pass' if all(t['status'] in ['pass', 'warn'] for t in tests) else 'fail',
            'tests': tests
        }
    
    def test_search_functionality(self) -> Dict[str, Any]:
        """Test search functionality."""
        tests = []
        
        # Test search page
        try:
            search_url = urljoin(self.site_url, '/search/')
            response = requests.get(search_url, timeout=10)
            
            tests.append({
                'name': 'Search: Page accessibility',
                'status': 'pass' if response.status_code == 200 else 'fail',
                'details': f"Search page status: {response.status_code}"
            })
            
            # Check for search elements
            content = response.text
            if 'search' in content.lower():
                tests.append({
                    'name': 'Search: Search elements',
                    'status': 'pass',
                    'details': 'Search elements found'
                })
            else:
                tests.append({
                    'name': 'Search: Search elements',
                    'status': 'warn',
                    'details': 'No search elements found'
                })
            
        except Exception as e:
            tests.append({
                'name': 'Search functionality',
                'status': 'fail',
                'details': f"Error: {str(e)}"
            })
        
        return {
            'status': 'pass' if all(t['status'] in ['pass', 'warn'] for t in tests) else 'fail',
            'tests': tests
        }
    
    def test_comments_system(self) -> Dict[str, Any]:
        """Test comments system integration."""
        tests = []
        
        try:
            response = requests.get(self.site_url, timeout=10)
            content = response.text
            
            # Check for comments system
            if 'giscus' in content or 'disqus' in content or 'utterances' in content:
                tests.append({
                    'name': 'Comments: System integration',
                    'status': 'pass',
                    'details': 'Comments system found'
                })
            else:
                tests.append({
                    'name': 'Comments: System integration',
                    'status': 'warn',
                    'details': 'No comments system detected'
                })
            
        except Exception as e:
            tests.append({
                'name': 'Comments system test',
                'status': 'fail',
                'details': f"Error: {str(e)}"
            })
        
        return {
            'status': 'pass' if all(t['status'] in ['pass', 'warn'] for t in tests) else 'fail',
            'tests': tests
        }
    
    def test_analytics_integration(self) -> Dict[str, Any]:
        """Test analytics integration."""
        tests = []
        
        try:
            response = requests.get(self.site_url, timeout=10)
            content = response.text
            
            # Check for analytics
            analytics_services = [
                ('Google Analytics', 'google-analytics.com'),
                ('Google Tag Manager', 'googletagmanager.com'),
                ('Facebook Pixel', 'facebook.com/tr'),
                ('Other analytics', 'analytics')
            ]
            
            for service_name, service_tag in analytics_services:
                if service_tag in content:
                    tests.append({
                        'name': f'Analytics: {service_name}',
                        'status': 'pass',
                        'details': f"Found {service_tag}"
                    })
                else:
                    tests.append({
                        'name': f'Analytics: {service_name}',
                        'status': 'warn',
                        'details': f"Missing {service_tag}"
                    })
            
        except Exception as e:
            tests.append({
                'name': 'Analytics test',
                'status': 'fail',
                'details': f"Error: {str(e)}"
            })
        
        return {
            'status': 'pass' if all(t['status'] in ['pass', 'warn'] for t in tests) else 'fail',
            'tests': tests
        }
    
    def test_security_headers(self) -> Dict[str, Any]:
        """Test security headers."""
        tests = []
        
        try:
            response = requests.get(self.site_url, timeout=10)
            headers = response.headers
            
            # Check security headers
            security_headers = [
                ('HTTPS', 'https' in self.site_url),
                ('Content Security Policy', 'content-security-policy' in headers),
                ('X-Frame-Options', 'x-frame-options' in headers),
                ('X-Content-Type-Options', 'x-content-type-options' in headers),
                ('X-XSS-Protection', 'x-xss-protection' in headers),
                ('Referrer-Policy', 'referrer-policy' in headers)
            ]
            
            for header_name, header_present in security_headers:
                tests.append({
                    'name': f'Security: {header_name}',
                    'status': 'pass' if header_present else 'warn',
                    'details': f"{header_name}: {'Present' if header_present else 'Missing'}"
                })
            
        except Exception as e:
            tests.append({
                'name': 'Security headers test',
                'status': 'fail',
                'details': f"Error: {str(e)}"
            })
        
        return {
            'status': 'pass' if all(t['status'] in ['pass', 'warn'] for t in tests) else 'fail',
            'tests': tests
        }
    
    def test_accessibility_compliance(self) -> Dict[str, Any]:
        """Test accessibility compliance."""
        tests = []
        
        try:
            response = requests.get(self.site_url, timeout=10)
            content = response.text
            
            # Check accessibility elements
            accessibility_elements = [
                ('Alt text for images', 'alt='),
                ('Heading structure', '<h1>'),
                ('Form labels', '<label'),
                ('ARIA attributes', 'aria-'),
                ('Skip links', 'skip'),
                ('Focus management', 'tabindex')
            ]
            
            for element_name, element_tag in accessibility_elements:
                if element_tag in content:
                    tests.append({
                        'name': f'Accessibility: {element_name}',
                        'status': 'pass',
                        'details': f"Found {element_tag}"
                    })
                else:
                    tests.append({
                        'name': f'Accessibility: {element_name}',
                        'status': 'warn',
                        'details': f"Missing {element_tag}"
                    })
            
        except Exception as e:
            tests.append({
                'name': 'Accessibility test',
                'status': 'fail',
                'details': f"Error: {str(e)}"
            })
        
        return {
            'status': 'pass' if all(t['status'] in ['pass', 'warn'] for t in tests) else 'fail',
            'tests': tests
        }
    
    def check_broken_links(self, content: str) -> List[str]:
        """Check for broken links in content."""
        broken_links = []
        
        # Extract links from content
        link_pattern = r'href=["\']([^"\']+)["\']'
        links = re.findall(link_pattern, content)
        
        for link in links[:10]:  # Check first 10 links to avoid timeout
            try:
                if link.startswith('http'):
                    response = requests.head(link, timeout=5)
                    if response.status_code >= 400:
                        broken_links.append(link)
                elif link.startswith('/'):
                    full_url = urljoin(self.site_url, link)
                    response = requests.head(full_url, timeout=5)
                    if response.status_code >= 400:
                        broken_links.append(link)
            except:
                broken_links.append(link)
        
        return broken_links
    
    def update_summary(self, test_result: Dict[str, Any]):
        """Update test summary statistics."""
        self.test_results['summary']['total_tests'] += len(test_result.get('tests', []))
        
        for test in test_result.get('tests', []):
            if test['status'] == 'pass':
                self.test_results['summary']['passed_tests'] += 1
            elif test['status'] == 'fail':
                self.test_results['summary']['failed_tests'] += 1
            elif test['status'] == 'warn':
                self.test_results['summary']['warnings'] += 1
    
    def generate_report(self) -> str:
        """Generate a comprehensive test report."""
        report = f"""
# Documentation Site Testing Report

**Generated**: {self.test_results['timestamp']}
**Site URL**: {self.test_results['site_url']}

## Summary

- **Total Tests**: {self.test_results['summary']['total_tests']}
- **Passed**: {self.test_results['summary']['passed_tests']}
- **Failed**: {self.test_results['summary']['failed_tests']}
- **Warnings**: {self.test_results['summary']['warnings']}

## Test Results

"""
        
        for test_category, test_result in self.test_results['tests'].items():
            report += f"### {test_category.replace('_', ' ').title()}\n\n"
            report += f"**Status**: {test_result['status']}\n\n"
            
            for test in test_result.get('tests', []):
                status_icon = "✅" if test['status'] == 'pass' else "❌" if test['status'] == 'fail' else "⚠️"
                report += f"- {status_icon} **{test['name']}**: {test['details']}\n"
            
            report += "\n"
        
        return report
    
    def save_report(self, filename: str = None):
        """Save test report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"site_test_report_{timestamp}.md"
        
        report = self.generate_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Test report saved to: {filename}")
        return filename


def main():
    """Main function to run site testing."""
    logger.info("Starting documentation site testing...")
    
    # Get site URL from environment or use default
    site_url = os.environ.get('SITE_URL', 'https://samne.github.io/ai_assignments')
    
    # Create tester and run tests
    tester = SiteTester(site_url)
    results = tester.run_all_tests()
    
    # Generate and save report
    report_filename = tester.save_report()
    
    # Print summary
    summary = results['summary']
    logger.info(f"Testing completed:")
    logger.info(f"  Total tests: {summary['total_tests']}")
    logger.info(f"  Passed: {summary['passed_tests']}")
    logger.info(f"  Failed: {summary['failed_tests']}")
    logger.info(f"  Warnings: {summary['warnings']}")
    
    # Exit with error code if there are failures
    if summary['failed_tests'] > 0:
        logger.error("Some tests failed. Check the report for details.")
        sys.exit(1)
    else:
        logger.info("All tests passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
