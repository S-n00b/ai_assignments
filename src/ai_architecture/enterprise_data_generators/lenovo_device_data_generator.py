"""
Lenovo Device Data Generator

Generates realistic Lenovo device data including:
- Moto Edge series mobile devices
- ThinkPad laptop series
- ThinkSystem server infrastructure
- Device specifications, configurations, and support data
"""

import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import faker

@dataclass
class DeviceSpecification:
    """Device specification data structure"""
    model: str
    series: str
    category: str
    processor: str
    memory: str
    storage: str
    display: str
    connectivity: List[str]
    os: str
    price_range: str
    target_market: str
    release_date: str
    warranty_period: str
    support_level: str

@dataclass
class DeviceConfiguration:
    """Device configuration data structure"""
    device_id: str
    model: str
    configuration_name: str
    cpu_spec: str
    ram_spec: str
    storage_spec: str
    gpu_spec: Optional[str]
    network_spec: str
    power_spec: str
    cooling_spec: str
    use_case: str
    performance_tier: str
    cost_estimate: float

class LenovoDeviceDataGenerator:
    """Generates comprehensive Lenovo device data"""
    
    def __init__(self):
        self.fake = faker.Faker()
        self.device_series = {
            "mobile": ["Moto Edge", "Moto G", "Moto E", "Legion Phone"],
            "laptop": ["ThinkPad", "IdeaPad", "Legion", "Yoga"],
            "server": ["ThinkSystem", "ThinkServer", "System x"]
        }
        
        self.processor_options = {
            "mobile": ["Snapdragon 8 Gen 3", "Snapdragon 7 Gen 3", "MediaTek Dimensity 9000", "Snapdragon 6 Gen 1"],
            "laptop": ["Intel Core i7-13700H", "Intel Core i5-13400H", "AMD Ryzen 7 7840HS", "Intel Core i9-13900H"],
            "server": ["Intel Xeon Gold 6338", "AMD EPYC 7543", "Intel Xeon Silver 4314", "AMD EPYC 7763"]
        }
        
        self.memory_options = {
            "mobile": ["8GB LPDDR5", "12GB LPDDR5", "16GB LPDDR5", "6GB LPDDR4X"],
            "laptop": ["16GB DDR5", "32GB DDR5", "64GB DDR5", "8GB DDR4"],
            "server": ["64GB DDR4", "128GB DDR4", "256GB DDR4", "512GB DDR4"]
        }
        
        self.storage_options = {
            "mobile": ["128GB UFS 3.1", "256GB UFS 3.1", "512GB UFS 3.1", "1TB UFS 3.1"],
            "laptop": ["512GB NVMe SSD", "1TB NVMe SSD", "2TB NVMe SSD", "4TB NVMe SSD"],
            "server": ["1TB NVMe SSD", "2TB NVMe SSD", "4TB NVMe SSD", "8TB NVMe SSD"]
        }

    def generate_moto_edge_series(self, count: int = 50) -> List[DeviceSpecification]:
        """Generate Moto Edge series mobile devices"""
        devices = []
        
        edge_models = [
            "Moto Edge 50 Pro", "Moto Edge 50", "Moto Edge 50 Fusion",
            "Moto Edge 40 Neo", "Moto Edge 40", "Moto Edge 40 Pro",
            "Moto Edge 30 Pro", "Moto Edge 30", "Moto Edge 30 Fusion"
        ]
        
        for i in range(count):
            model = random.choice(edge_models)
            devices.append(DeviceSpecification(
                model=model,
                series="Moto Edge",
                category="Mobile",
                processor=random.choice(self.processor_options["mobile"]),
                memory=random.choice(self.memory_options["mobile"]),
                storage=random.choice(self.storage_options["mobile"]),
                display=f"{random.choice(['6.1', '6.3', '6.5', '6.7'])}\" OLED {random.choice(['90Hz', '120Hz', '144Hz'])}",
                connectivity=["5G", "WiFi 6E", "Bluetooth 5.3", "NFC", "USB-C"],
                os="Android 14",
                price_range=random.choice(["$299-$399", "$399-$599", "$599-$799", "$799-$999"]),
                target_market=random.choice(["Consumer", "Business", "Gaming", "Photography"]),
                release_date=self.fake.date_between(start_date='-2y', end_date='today').strftime('%Y-%m-%d'),
                warranty_period="2 years",
                support_level=random.choice(["Standard", "Premium", "Enterprise"])
            ))
        
        return devices

    def generate_thinkpad_series(self, count: int = 50) -> List[DeviceSpecification]:
        """Generate ThinkPad series laptops"""
        devices = []
        
        thinkpad_models = [
            "ThinkPad X1 Carbon Gen 11", "ThinkPad X1 Yoga Gen 8", "ThinkPad X1 Nano Gen 3",
            "ThinkPad T14 Gen 4", "ThinkPad T16 Gen 2", "ThinkPad P16 Gen 2",
            "ThinkPad L14 Gen 4", "ThinkPad L15 Gen 4", "ThinkPad E15 Gen 5"
        ]
        
        for i in range(count):
            model = random.choice(thinkpad_models)
            devices.append(DeviceSpecification(
                model=model,
                series="ThinkPad",
                category="Laptop",
                processor=random.choice(self.processor_options["laptop"]),
                memory=random.choice(self.memory_options["laptop"]),
                storage=random.choice(self.storage_options["laptop"]),
                display=f"{random.choice(['14', '15.6', '16'])}\" {random.choice(['FHD', 'QHD', '4K'])} {random.choice(['IPS', 'OLED'])}",
                connectivity=["WiFi 6E", "Bluetooth 5.3", "USB-C", "Thunderbolt 4", "HDMI", "Ethernet"],
                os=random.choice(["Windows 11 Pro", "Windows 11 Home", "Ubuntu 22.04 LTS"]),
                price_range=random.choice(["$899-$1299", "$1299-$1799", "$1799-$2499", "$2499-$3499"]),
                target_market=random.choice(["Business", "Enterprise", "Professional", "Developer"]),
                release_date=self.fake.date_between(start_date='-2y', end_date='today').strftime('%Y-%m-%d'),
                warranty_period="3 years",
                support_level=random.choice(["Standard", "Premium", "Enterprise", "On-site"])
            ))
        
        return devices

    def generate_thinksystem_series(self, count: int = 30) -> List[DeviceSpecification]:
        """Generate ThinkSystem server infrastructure"""
        devices = []
        
        thinksystem_models = [
            "ThinkSystem SR650 V2", "ThinkSystem SR850 V2", "ThinkSystem SR950 V2",
            "ThinkSystem ST250 V2", "ThinkSystem ST550 V2", "ThinkSystem ST650 V2",
            "ThinkSystem SD530 V2", "ThinkSystem SD650 V2", "ThinkSystem SD850 V2"
        ]
        
        for i in range(count):
            model = random.choice(thinksystem_models)
            devices.append(DeviceSpecification(
                model=model,
                series="ThinkSystem",
                category="Server",
                processor=random.choice(self.processor_options["server"]),
                memory=random.choice(self.memory_options["server"]),
                storage=random.choice(self.storage_options["server"]),
                display="Headless Server",
                connectivity=["10Gb Ethernet", "25Gb Ethernet", "100Gb Ethernet", "InfiniBand"],
                os=random.choice(["VMware vSphere", "Red Hat Enterprise Linux", "SUSE Linux Enterprise", "Windows Server 2022"]),
                price_range=random.choice(["$5000-$15000", "$15000-$50000", "$50000-$100000", "$100000+"]),
                target_market=random.choice(["Enterprise", "Data Center", "Cloud", "HPC"]),
                release_date=self.fake.date_between(start_date='-3y', end_date='today').strftime('%Y-%m-%d'),
                warranty_period="5 years",
                support_level=random.choice(["Standard", "Premium", "Enterprise", "24/7"])
            ))
        
        return devices

    def generate_device_configurations(self, devices: List[DeviceSpecification], count: int = 100) -> List[DeviceConfiguration]:
        """Generate device configurations for deployment scenarios"""
        configurations = []
        
        use_cases = [
            "Development Workstation", "Data Analytics", "Machine Learning Training",
            "Web Server", "Database Server", "File Server", "Application Server",
            "Virtualization Host", "High Performance Computing", "Edge Computing"
        ]
        
        performance_tiers = ["Entry", "Mid-range", "High-end", "Workstation", "Enterprise"]
        
        for i in range(count):
            device = random.choice(devices)
            config_id = f"CONFIG_{i+1:04d}"
            
            configurations.append(DeviceConfiguration(
                device_id=f"DEV_{random.randint(1000, 9999)}",
                model=device.model,
                configuration_name=f"{device.model} - {random.choice(use_cases)}",
                cpu_spec=device.processor,
                ram_spec=device.memory,
                storage_spec=device.storage,
                gpu_spec=random.choice(["NVIDIA RTX 4090", "NVIDIA RTX 4080", "NVIDIA A100", "AMD Radeon Pro W7900", None]),
                network_spec=random.choice(["1Gb Ethernet", "10Gb Ethernet", "25Gb Ethernet", "100Gb Ethernet"]),
                power_spec=random.choice(["300W", "500W", "750W", "1000W", "1500W"]),
                cooling_spec=random.choice(["Air Cooling", "Liquid Cooling", "Hybrid Cooling"]),
                use_case=random.choice(use_cases),
                performance_tier=random.choice(performance_tiers),
                cost_estimate=random.uniform(1000, 50000)
            ))
        
        return configurations

    def generate_support_knowledge_base(self, devices: List[DeviceSpecification]) -> List[Dict[str, Any]]:
        """Generate support knowledge base entries"""
        knowledge_entries = []
        
        common_issues = [
            "Battery not charging", "WiFi connectivity issues", "Bluetooth pairing problems",
            "Display flickering", "Audio not working", "Camera not functioning",
            "Performance degradation", "Overheating issues", "Boot problems",
            "Driver installation", "Software compatibility", "Hardware failure"
        ]
        
        solutions = [
            "Update device drivers", "Reset network settings", "Clear cache and data",
            "Check hardware connections", "Update firmware", "Contact support",
            "Replace hardware component", "Software reinstallation", "Factory reset"
        ]
        
        for device in devices:
            for issue in random.sample(common_issues, random.randint(3, 8)):
                knowledge_entries.append({
                    "device_model": device.model,
                    "issue_category": issue,
                    "symptoms": [
                        f"Device shows {random.choice(['error', 'warning'])} message",
                        f"Performance is {random.choice(['slow', 'unstable'])}",
                        f"Feature {random.choice(['not working', 'malfunctioning'])}"
                    ],
                    "troubleshooting_steps": [
                        f"Step 1: {random.choice(solutions)}",
                        f"Step 2: {random.choice(solutions)}",
                        f"Step 3: {random.choice(solutions)}"
                    ],
                    "resolution": random.choice(solutions),
                    "support_level": device.support_level,
                    "escalation_path": f"Contact {device.support_level} support",
                    "documentation_links": [
                        f"https://support.lenovo.com/{device.model.lower().replace(' ', '-')}",
                        f"https://docs.lenovo.com/{device.series.lower()}"
                    ]
                })
        
        return knowledge_entries

    def generate_all_device_data(self) -> Dict[str, Any]:
        """Generate comprehensive device data"""
        print("Generating Lenovo device data...")
        
        # Generate device specifications
        moto_devices = self.generate_moto_edge_series(50)
        thinkpad_devices = self.generate_thinkpad_series(50)
        thinksystem_devices = self.generate_thinksystem_series(30)
        
        all_devices = moto_devices + thinkpad_devices + thinksystem_devices
        
        # Generate configurations
        configurations = self.generate_device_configurations(all_devices, 100)
        
        # Generate support knowledge
        support_knowledge = self.generate_support_knowledge_base(all_devices)
        
        # Compile results
        result = {
            "generation_timestamp": datetime.now().isoformat(),
            "device_specifications": [asdict(device) for device in all_devices],
            "device_configurations": [asdict(config) for config in configurations],
            "support_knowledge_base": support_knowledge,
            "statistics": {
                "total_devices": len(all_devices),
                "mobile_devices": len(moto_devices),
                "laptop_devices": len(thinkpad_devices),
                "server_devices": len(thinksystem_devices),
                "configurations": len(configurations),
                "support_entries": len(support_knowledge)
            }
        }
        
        return result

    def save_to_json(self, data: Dict[str, Any], filename: str = "lenovo_device_data.json"):
        """Save generated data to JSON file"""
        import os
        os.makedirs("data/lenovo_devices", exist_ok=True)
        filepath = f"data/lenovo_devices/{filename}"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Device data saved to {filepath}")
        return filepath

if __name__ == "__main__":
    generator = LenovoDeviceDataGenerator()
    data = generator.generate_all_device_data()
    filepath = generator.save_to_json(data)
    print(f"Generated {data['statistics']['total_devices']} devices with {data['statistics']['support_entries']} support entries")
