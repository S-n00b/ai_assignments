"""
Enhanced Prompt Registries for Enterprise Focus

This module provides enterprise-focused prompt registries for:
- Lenovo business scenarios
- Device support workflows
- Customer service interactions
- Technical documentation
- Mobile device scenarios
- Factory roster deployment
"""

from .lenovo_business_prompts import LenovoBusinessPrompts
from .device_support_prompts import DeviceSupportPrompts
from .customer_service_prompts import CustomerServicePrompts
from .technical_documentation_prompts import TechnicalDocumentationPrompts
from .mobile_device_prompts import MobileDevicePrompts
from .factory_roster_prompts import FactoryRosterPrompts

__all__ = [
    "LenovoBusinessPrompts",
    "DeviceSupportPrompts",
    "CustomerServicePrompts", 
    "TechnicalDocumentationPrompts",
    "MobileDevicePrompts",
    "FactoryRosterPrompts"
]
