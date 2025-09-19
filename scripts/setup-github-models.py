#!/usr/bin/env python3
"""
GitHub Models Setup Script

This script helps you set up authentication for GitHub Models API.
"""

import os
import sys
import subprocess
import getpass
from pathlib import Path

def print_header():
    """Print setup header."""
    print("=" * 60)
    print("🚀 GitHub Models API Setup")
    print("=" * 60)
    print()

def check_github_cli():
    """Check if GitHub CLI is installed."""
    try:
        result = subprocess.run(["gh", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ GitHub CLI is installed")
            return True
        else:
            print("❌ GitHub CLI is not installed")
            return False
    except FileNotFoundError:
        print("❌ GitHub CLI is not installed")
        return False

def setup_github_cli():
    """Set up GitHub CLI authentication."""
    print("\n📋 Setting up GitHub CLI authentication...")
    print("This will open a browser for authentication.")
    
    try:
        # Run gh auth login
        result = subprocess.run(["gh", "auth", "login"], input="\n\n\n\n\n", text=True)
        
        if result.returncode == 0:
            print("✅ GitHub CLI authentication successful!")
            return True
        else:
            print("❌ GitHub CLI authentication failed")
            return False
    except Exception as e:
        print(f"❌ Error during GitHub CLI setup: {e}")
        return False

def create_pat_instructions():
    """Print instructions for creating a Personal Access Token."""
    print("\n📋 Manual PAT Setup Instructions:")
    print("-" * 40)
    print("1. Go to: https://github.com/settings/tokens")
    print("2. Click 'Generate new token (classic)'")
    print("3. Give it a name: 'Lenovo AAITC Models API'")
    print("4. Select scopes:")
    print("   ✅ models (required for GitHub Models API)")
    print("   ✅ repo (optional, for repository access)")
    print("5. Set expiration: 90 days (recommended)")
    print("6. Click 'Generate token'")
    print("7. Copy the token (it won't be shown again!)")
    print("8. Set environment variable:")
    print("   Windows: set GITHUB_TOKEN=your_token_here")
    print("   Linux/Mac: export GITHUB_TOKEN=your_token_here")
    print()

def test_authentication():
    """Test GitHub Models authentication."""
    print("\n🧪 Testing GitHub Models authentication...")
    
    # Check environment variable
    token = os.getenv("GITHUB_TOKEN")
    if token:
        print("✅ GITHUB_TOKEN environment variable is set")
        if token == "demo_token":
            print("⚠️  Using demo token - limited functionality")
            return False
        else:
            print("✅ Real token detected")
            return True
    
    # Check GitHub CLI
    try:
        result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            print("✅ GitHub CLI token available")
            return True
    except:
        pass
    
    print("❌ No authentication found")
    return False

def create_env_file():
    """Create .env file template."""
    env_file = Path(".env")
    if not env_file.exists():
        print("\n📝 Creating .env file template...")
        with open(env_file, "w") as f:
            f.write("# GitHub Models API Configuration\n")
            f.write("# Replace 'your_token_here' with your actual GitHub token\n")
            f.write("GITHUB_TOKEN=your_token_here\n")
        print(f"✅ Created {env_file}")
        print("   Edit this file and add your GitHub token")
    else:
        print(f"✅ {env_file} already exists")

def main():
    """Main setup function."""
    print_header()
    
    # Check current authentication
    if test_authentication():
        print("\n🎉 GitHub Models authentication is already set up!")
        print("You can now use the GitHub Models backend.")
        return
    
    print("\n🔧 Setting up GitHub Models authentication...")
    
    # Option 1: Try GitHub CLI
    if check_github_cli():
        print("\nOption 1: Use GitHub CLI (Recommended)")
        print("-" * 40)
        response = input("Set up GitHub CLI authentication? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            if setup_github_cli():
                print("\n🎉 GitHub CLI setup complete!")
                if test_authentication():
                    print("✅ Authentication verified!")
                    return
    
    # Option 2: Manual PAT setup
    print("\nOption 2: Manual Personal Access Token Setup")
    print("-" * 40)
    create_pat_instructions()
    
    # Create .env file
    create_env_file()
    
    print("\n📋 Next Steps:")
    print("1. Follow the PAT setup instructions above")
    print("2. Set your GITHUB_TOKEN environment variable")
    print("3. Run this script again to verify setup")
    print("4. Start using GitHub Models in your application!")
    
    print("\n💡 Quick Test:")
    print("   python -c \"from src.github_models_backend import GitHubModelsClient; print('Setup complete!')\"")

if __name__ == "__main__":
    main()
