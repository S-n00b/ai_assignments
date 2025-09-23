"""
Edge Deployment Configurations for Small Models

This module provides deployment configurations for edge devices including
Android, iOS, Edge servers, and embedded systems.
"""

import json
import yaml
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class EdgeDeploymentConfig:
    """
    Edge deployment configuration manager.
    
    Provides platform-specific configurations for deploying small models
    on various edge devices and platforms.
    """
    
    def __init__(self, model_name: str, base_config: Optional[Dict] = None):
        """
        Initialize edge deployment configuration.
        
        Args:
            model_name: Name of the model to deploy
            base_config: Base configuration dictionary
        """
        self.model_name = model_name
        self.base_config = base_config or {}
        self.platform_configs = self._initialize_platform_configs()
        
    def _initialize_platform_configs(self) -> Dict[str, Dict]:
        """Initialize platform-specific configurations."""
        return {
            "android": {
                "platform": "android",
                "architecture": "arm64-v8a",
                "min_sdk": 21,
                "target_sdk": 33,
                "min_ram_mb": 512,
                "max_model_size_mb": 100,
                "quantization": "int8",
                "optimization_level": "mobile",
                "inference_engine": "tflite",
                "deployment_package": "aar",
                "dependencies": [
                    "tensorflow-lite",
                    "tensorflow-lite-gpu",
                    "tensorflow-lite-support"
                ],
                "performance_targets": {
                    "max_inference_time_ms": 50,
                    "max_memory_usage_mb": 256,
                    "min_throughput_tokens_per_second": 20
                }
            },
            "ios": {
                "platform": "ios",
                "architecture": "arm64",
                "min_version": "12.0",
                "target_version": "16.0",
                "min_ram_mb": 512,
                "max_model_size_mb": 100,
                "quantization": "int8",
                "optimization_level": "mobile",
                "inference_engine": "coreml",
                "deployment_package": "framework",
                "dependencies": [
                    "CoreML",
                    "NaturalLanguage",
                    "CreateML"
                ],
                "performance_targets": {
                    "max_inference_time_ms": 50,
                    "max_memory_usage_mb": 256,
                    "min_throughput_tokens_per_second": 20
                }
            },
            "edge": {
                "platform": "edge",
                "architecture": "x86_64",
                "os": "linux",
                "min_ram_gb": 1,
                "max_model_size_mb": 200,
                "quantization": "int8",
                "optimization_level": "balanced",
                "inference_engine": "onnx",
                "deployment_package": "docker",
                "dependencies": [
                    "onnxruntime",
                    "torch",
                    "transformers"
                ],
                "performance_targets": {
                    "max_inference_time_ms": 100,
                    "max_memory_usage_mb": 512,
                    "min_throughput_tokens_per_second": 50
                }
            },
            "embedded": {
                "platform": "embedded",
                "architecture": "armv7",
                "os": "linux",
                "min_ram_mb": 256,
                "max_model_size_mb": 50,
                "quantization": "int8",
                "optimization_level": "aggressive",
                "inference_engine": "tflite",
                "deployment_package": "static_lib",
                "dependencies": [
                    "tensorflow-lite",
                    "tensorflow-lite-micro"
                ],
                "performance_targets": {
                    "max_inference_time_ms": 25,
                    "max_memory_usage_mb": 128,
                    "min_throughput_tokens_per_second": 10
                }
            }
        }
    
    def get_platform_config(self, platform: str) -> Dict:
        """
        Get configuration for specific platform.
        
        Args:
            platform: Target platform name
            
        Returns:
            Platform configuration
        """
        if platform not in self.platform_configs:
            raise ValueError(f"Unsupported platform: {platform}")
        
        config = self.platform_configs[platform].copy()
        config.update(self.base_config)
        return config
    
    def create_deployment_package(self, 
                                 platform: str,
                                 model_path: str,
                                 output_dir: str) -> str:
        """
        Create deployment package for specific platform.
        
        Args:
            platform: Target platform
            model_path: Path to the model
            output_dir: Output directory
            
        Returns:
            Path to deployment package
        """
        try:
            config = self.get_platform_config(platform)
            output_path = Path(output_dir) / platform
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create platform-specific files
            self._create_platform_files(platform, config, model_path, output_path)
            
            # Create deployment manifest
            manifest = {
                "model_name": self.model_name,
                "platform": platform,
                "version": "1.0.0",
                "deployment_date": str(Path().cwd()),
                "configuration": config,
                "model_path": model_path,
                "deployment_ready": True
            }
            
            with open(output_path / "deployment_manifest.json", 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"Deployment package created for {platform} at {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to create deployment package for {platform}: {e}")
            raise
    
    def _create_platform_files(self, 
                              platform: str, 
                              config: Dict, 
                              model_path: str, 
                              output_path: Path):
        """Create platform-specific deployment files."""
        if platform == "android":
            self._create_android_files(config, model_path, output_path)
        elif platform == "ios":
            self._create_ios_files(config, model_path, output_path)
        elif platform == "edge":
            self._create_edge_files(config, model_path, output_path)
        elif platform == "embedded":
            self._create_embedded_files(config, model_path, output_path)
    
    def _create_android_files(self, config: Dict, model_path: str, output_path: Path):
        """Create Android deployment files."""
        # Create build.gradle
        build_gradle = f"""
android {{
    compileSdk {config['target_sdk']}
    
    defaultConfig {{
        minSdk {config['min_sdk']}
        targetSdk {config['target_sdk']}
        ndk {{
            abiFilters '{config['architecture']}'
        }}
    }}
    
    dependencies {{
        {chr(10).join([f"    implementation '{dep}'" for dep in config['dependencies']])}
    }}
}}
"""
        with open(output_path / "build.gradle", 'w') as f:
            f.write(build_gradle)
        
        # Create Android manifest
        manifest = f"""
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <uses-permission android:name="android.permission.INTERNET" />
    <uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
    
    <application
        android:allowBackup="true"
        android:label="{self.model_name}">
        
        <activity
            android:name=".MainActivity"
            android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>
</manifest>
"""
        with open(output_path / "AndroidManifest.xml", 'w') as f:
            f.write(manifest)
    
    def _create_ios_files(self, config: Dict, model_path: str, output_path: Path):
        """Create iOS deployment files."""
        # Create Info.plist
        info_plist = f"""
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDisplayName</key>
    <string>{self.model_name}</string>
    <key>CFBundleIdentifier</key>
    <string>com.lenovo.{self.model_name.lower()}</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>{config['min_version']}</string>
    <key>UIRequiredDeviceCapabilities</key>
    <array>
        <string>arm64</string>
    </array>
</dict>
</plist>
"""
        with open(output_path / "Info.plist", 'w') as f:
            f.write(info_plist)
        
        # Create Podfile
        podfile = f"""
platform :ios, '{config['min_version']}'
use_frameworks!

target '{self.model_name}' do
    {chr(10).join([f"  pod '{dep}'" for dep in config['dependencies']])}
end
"""
        with open(output_path / "Podfile", 'w') as f:
            f.write(podfile)
    
    def _create_edge_files(self, config: Dict, model_path: str, output_path: Path):
        """Create Edge deployment files."""
        # Create Dockerfile
        dockerfile = f"""
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
{chr(10).join([f"RUN pip install {dep}" for dep in config['dependencies']])}

# Copy model and application
COPY {model_path} /app/model/
COPY . /app/

# Set environment variables
ENV MODEL_PATH=/app/model
ENV PLATFORM={config['platform']}
ENV OPTIMIZATION_LEVEL={config['optimization_level']}

# Expose port
EXPOSE 8080

# Run application
CMD ["python", "app.py"]
"""
        with open(output_path / "Dockerfile", 'w') as f:
            f.write(dockerfile)
        
        # Create docker-compose.yml
        docker_compose = f"""
version: '3.8'
services:
  {self.model_name}:
    build: .
    ports:
      - "8080:8080"
    environment:
      - MODEL_PATH=/app/model
      - PLATFORM={config['platform']}
    volumes:
      - ./model:/app/model
    restart: unless-stopped
"""
        with open(output_path / "docker-compose.yml", 'w') as f:
            f.write(docker_compose)
    
    def _create_embedded_files(self, config: Dict, model_path: str, output_path: Path):
        """Create Embedded deployment files."""
        # Create CMakeLists.txt
        cmake = f"""
cmake_minimum_required(VERSION 3.10)
project({self.model_name})

set(CMAKE_CXX_STANDARD 14)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(TFLITE REQUIRED tensorflow-lite)

# Include directories
include_directories(${{TFLITE_INCLUDE_DIRS}})

# Add executable
add_executable({self.model_name} main.cpp)

# Link libraries
target_link_libraries({self.model_name} ${{TFLITE_LIBRARIES}})

# Compiler flags
target_compile_options({self.model_name} PRIVATE ${{TFLITE_CFLAGS_OTHER}})
"""
        with open(output_path / "CMakeLists.txt", 'w') as f:
            f.write(cmake)
        
        # Create embedded configuration
        embedded_config = {
            "platform": config["platform"],
            "architecture": config["architecture"],
            "memory_requirements": {
                "min_ram_mb": config["min_ram_mb"],
                "max_model_size_mb": config["max_model_size_mb"]
            },
            "performance_targets": config["performance_targets"],
            "optimization": {
                "level": config["optimization_level"],
                "quantization": config["quantization"]
            }
        }
        
        with open(output_path / "embedded_config.json", 'w') as f:
            json.dump(embedded_config, f, indent=2)
    
    def validate_deployment(self, 
                           platform: str, 
                           model_metrics: Dict[str, float]) -> Dict[str, bool]:
        """
        Validate deployment against platform requirements.
        
        Args:
            platform: Target platform
            model_metrics: Model performance metrics
            
        Returns:
            Validation results
        """
        try:
            config = self.get_platform_config(platform)
            targets = config["performance_targets"]
            
            validation_results = {
                "inference_time": model_metrics.get("avg_inference_time_ms", 0) <= targets["max_inference_time_ms"],
                "memory_usage": model_metrics.get("memory_usage_mb", 0) <= targets["max_memory_usage_mb"],
                "throughput": model_metrics.get("throughput_tokens_per_second", 0) >= targets["min_throughput_tokens_per_second"],
                "model_size": model_metrics.get("model_size_mb", 0) <= config["max_model_size_mb"]
            }
            
            # Overall validation
            validation_results["deployment_ready"] = all(validation_results.values())
            
            logger.info(f"Deployment validation for {platform}: {validation_results}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Deployment validation failed: {e}")
            raise
    
    def create_deployment_script(self, 
                                platform: str, 
                                output_dir: str) -> str:
        """
        Create deployment script for platform.
        
        Args:
            platform: Target platform
            output_dir: Output directory
            
        Returns:
            Path to deployment script
        """
        try:
            config = self.get_platform_config(platform)
            script_path = Path(output_dir) / f"deploy_{platform}.sh"
            
            if platform == "android":
                script_content = f"""
#!/bin/bash
# Android deployment script for {self.model_name}

echo "Building Android deployment package..."

# Build AAR
./gradlew assembleRelease

# Copy to output directory
cp app/build/outputs/aar/app-release.aar {output_dir}/{self.model_name}.aar

echo "Android deployment package created: {self.model_name}.aar"
"""
            elif platform == "ios":
                script_content = f"""
#!/bin/bash
# iOS deployment script for {self.model_name}

echo "Building iOS deployment package..."

# Install dependencies
pod install

# Build framework
xcodebuild -workspace {self.model_name}.xcworkspace -scheme {self.model_name} -configuration Release

# Create framework package
cp -R build/Release-iphoneos/{self.model_name}.framework {output_dir}/

echo "iOS deployment package created: {self.model_name}.framework"
"""
            elif platform == "edge":
                script_content = f"""
#!/bin/bash
# Edge deployment script for {self.model_name}

echo "Building Edge deployment package..."

# Build Docker image
docker build -t {self.model_name}:latest .

# Save image
docker save {self.model_name}:latest | gzip > {output_dir}/{self.model_name}.tar.gz

echo "Edge deployment package created: {self.model_name}.tar.gz"
"""
            elif platform == "embedded":
                script_content = f"""
#!/bin/bash
# Embedded deployment script for {self.model_name}

echo "Building Embedded deployment package..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

# Copy executable
cp {self.model_name} {output_dir}/

echo "Embedded deployment package created: {self.model_name}"
"""
            else:
                raise ValueError(f"Unsupported platform: {platform}")
            
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Make script executable
            script_path.chmod(0o755)
            
            logger.info(f"Deployment script created: {script_path}")
            return str(script_path)
            
        except Exception as e:
            logger.error(f"Failed to create deployment script: {e}")
            raise
