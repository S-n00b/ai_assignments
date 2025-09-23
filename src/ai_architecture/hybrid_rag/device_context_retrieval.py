"""
Device Context Retrieval for Hybrid RAG

This module provides device-specific context retrieval for Lenovo devices
including specifications, support information, and troubleshooting data.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json
import numpy as np

logger = logging.getLogger(__name__)

class DeviceContextRetrieval:
    """
    Device context retrieval for hybrid RAG.
    
    Provides device-specific context retrieval for Lenovo devices
    including specifications, support information, and troubleshooting data.
    """
    
    def __init__(self, 
                 chromadb_client=None,
                 neo4j_client=None,
                 device_database_path: Optional[str] = None):
        """
        Initialize device context retrieval.
        
        Args:
            chromadb_client: ChromaDB client
            neo4j_client: Neo4j client
            device_database_path: Path to device database
        """
        self.chromadb_client = chromadb_client
        self.neo4j_client = neo4j_client
        self.device_database_path = device_database_path
        self.device_database = None
        
        if device_database_path:
            self.load_device_database()
    
    def load_device_database(self) -> None:
        """Load device database from file."""
        try:
            if self.device_database_path and Path(self.device_database_path).exists():
                with open(self.device_database_path, 'r', encoding='utf-8') as f:
                    self.device_database = json.load(f)
                logger.info(f"Loaded device database from {self.device_database_path}")
            else:
                logger.warning(f"Device database not found at {self.device_database_path}")
                
        except Exception as e:
            logger.error(f"Failed to load device database: {e}")
            raise
    
    def get_device_specifications(self, 
                                device_id: str,
                                include_related: bool = True) -> Dict[str, Any]:
        """
        Get device specifications.
        
        Args:
            device_id: Device ID
            include_related: Include related devices
            
        Returns:
            Device specifications
        """
        try:
            # Search in device database
            device_info = None
            if self.device_database:
                for device in self.device_database.get('devices', []):
                    if device.get('id') == device_id:
                        device_info = device
                        break
            
            if not device_info:
                return {'error': f'Device {device_id} not found'}
            
            # Get basic specifications
            specifications = {
                'id': device_info.get('id'),
                'name': device_info.get('name'),
                'type': device_info.get('type'),
                'category': device_info.get('category'),
                'specifications': device_info.get('specifications', {}),
                'support_info': device_info.get('support_info', ''),
                'common_issues': device_info.get('common_issues', []),
                'troubleshooting_guides': device_info.get('troubleshooting_guides', [])
            }
            
            # Include related devices if requested
            if include_related:
                related_devices = self._get_related_devices(device_info)
                specifications['related_devices'] = related_devices
            
            logger.info(f"Retrieved specifications for device: {device_id}")
            return specifications
            
        except Exception as e:
            logger.error(f"Failed to get device specifications: {e}")
            raise
    
    def _get_related_devices(self, device_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get related devices based on type and category."""
        try:
            related_devices = []
            device_type = device_info.get('type', '')
            device_category = device_info.get('category', '')
            
            if self.device_database:
                for device in self.device_database.get('devices', []):
                    if (device.get('id') != device_info.get('id') and 
                        (device.get('type') == device_type or 
                         device.get('category') == device_category)):
                        related_devices.append({
                            'id': device.get('id'),
                            'name': device.get('name'),
                            'type': device.get('type'),
                            'category': device.get('category')
                        })
            
            return related_devices[:5]  # Limit to 5 related devices
            
        except Exception as e:
            logger.error(f"Failed to get related devices: {e}")
            return []
    
    def search_device_support(self, 
                            query: str,
                            device_type: Optional[str] = None,
                            category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search device support information.
        
        Args:
            query: Search query
            device_type: Filter by device type
            category: Filter by category
            
        Returns:
            List of support information
        """
        try:
            support_results = []
            
            if self.device_database:
                for device in self.device_database.get('devices', []):
                    # Apply filters
                    if device_type and device.get('type') != device_type:
                        continue
                    if category and device.get('category') != category:
                        continue
                    
                    # Search in device information
                    device_text = f"{device.get('name', '')} {device.get('type', '')} {device.get('category', '')} {device.get('support_info', '')}"
                    
                    if query.lower() in device_text.lower():
                        support_results.append({
                            'device_id': device.get('id'),
                            'device_name': device.get('name'),
                            'device_type': device.get('type'),
                            'category': device.get('category'),
                            'support_info': device.get('support_info', ''),
                            'common_issues': device.get('common_issues', []),
                            'troubleshooting_guides': device.get('troubleshooting_guides', [])
                        })
            
            logger.info(f"Found {len(support_results)} support results for query: {query}")
            return support_results
            
        except Exception as e:
            logger.error(f"Failed to search device support: {e}")
            raise
    
    def get_troubleshooting_guide(self, 
                                device_id: str,
                                issue_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get troubleshooting guide for device.
        
        Args:
            device_id: Device ID
            issue_type: Type of issue to troubleshoot
            
        Returns:
            List of troubleshooting steps
        """
        try:
            # Get device information
            device_info = None
            if self.device_database:
                for device in self.device_database.get('devices', []):
                    if device.get('id') == device_id:
                        device_info = device
                        break
            
            if not device_info:
                return []
            
            # Get troubleshooting guides
            troubleshooting_guides = device_info.get('troubleshooting_guides', [])
            
            # Filter by issue type if specified
            if issue_type:
                filtered_guides = []
                for guide in troubleshooting_guides:
                    if issue_type.lower() in guide.get('issue_type', '').lower():
                        filtered_guides.append(guide)
                troubleshooting_guides = filtered_guides
            
            logger.info(f"Retrieved {len(troubleshooting_guides)} troubleshooting guides for device: {device_id}")
            return troubleshooting_guides
            
        except Exception as e:
            logger.error(f"Failed to get troubleshooting guide: {e}")
            raise
    
    def get_device_compatibility(self, 
                               device_id: str,
                               accessory_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get device compatibility information.
        
        Args:
            device_id: Device ID
            accessory_type: Type of accessory to check compatibility
            
        Returns:
            List of compatible accessories
        """
        try:
            # Get device information
            device_info = None
            if self.device_database:
                for device in self.device_database.get('devices', []):
                    if device.get('id') == device_id:
                        device_info = device
                        break
            
            if not device_info:
                return []
            
            # Get compatibility information
            compatibility_info = device_info.get('compatibility', [])
            
            # Filter by accessory type if specified
            if accessory_type:
                filtered_compatibility = []
                for item in compatibility_info:
                    if accessory_type.lower() in item.get('accessory_type', '').lower():
                        filtered_compatibility.append(item)
                compatibility_info = filtered_compatibility
            
            logger.info(f"Retrieved {len(compatibility_info)} compatibility items for device: {device_id}")
            return compatibility_info
            
        except Exception as e:
            logger.error(f"Failed to get device compatibility: {e}")
            raise
    
    def get_device_warranty_info(self, device_id: str) -> Dict[str, Any]:
        """
        Get device warranty information.
        
        Args:
            device_id: Device ID
            
        Returns:
            Warranty information
        """
        try:
            # Get device information
            device_info = None
            if self.device_database:
                for device in self.device_database.get('devices', []):
                    if device.get('id') == device_id:
                        device_info = device
                        break
            
            if not device_info:
                return {'error': f'Device {device_id} not found'}
            
            # Get warranty information
            warranty_info = device_info.get('warranty', {})
            
            logger.info(f"Retrieved warranty information for device: {device_id}")
            return warranty_info
            
        except Exception as e:
            logger.error(f"Failed to get device warranty info: {e}")
            raise
    
    def get_device_support_contacts(self, 
                                  device_id: str,
                                  support_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get device support contacts.
        
        Args:
            device_id: Device ID
            support_type: Type of support contact
            
        Returns:
            List of support contacts
        """
        try:
            # Get device information
            device_info = None
            if self.device_database:
                for device in self.device_database.get('devices', []):
                    if device.get('id') == device_id:
                        device_info = device
                        break
            
            if not device_info:
                return []
            
            # Get support contacts
            support_contacts = device_info.get('support_contacts', [])
            
            # Filter by support type if specified
            if support_type:
                filtered_contacts = []
                for contact in support_contacts:
                    if support_type.lower() in contact.get('type', '').lower():
                        filtered_contacts.append(contact)
                support_contacts = filtered_contacts
            
            logger.info(f"Retrieved {len(support_contacts)} support contacts for device: {device_id}")
            return support_contacts
            
        except Exception as e:
            logger.error(f"Failed to get device support contacts: {e}")
            raise
    
    def create_device_context(self, 
                            device_id: str,
                            context_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Create comprehensive device context.
        
        Args:
            device_id: Device ID
            context_type: Type of context (comprehensive, basic, support_only)
            
        Returns:
            Device context
        """
        try:
            # Get device specifications
            specifications = self.get_device_specifications(device_id, include_related=True)
            
            if 'error' in specifications:
                return specifications
            
            # Create context based on type
            if context_type == "comprehensive":
                context = {
                    'device_id': device_id,
                    'specifications': specifications,
                    'troubleshooting_guides': self.get_troubleshooting_guide(device_id),
                    'compatibility': self.get_device_compatibility(device_id),
                    'warranty': self.get_device_warranty_info(device_id),
                    'support_contacts': self.get_device_support_contacts(device_id)
                }
            elif context_type == "basic":
                context = {
                    'device_id': device_id,
                    'specifications': specifications
                }
            elif context_type == "support_only":
                context = {
                    'device_id': device_id,
                    'troubleshooting_guides': self.get_troubleshooting_guide(device_id),
                    'support_contacts': self.get_device_support_contacts(device_id)
                }
            else:
                raise ValueError(f"Unsupported context type: {context_type}")
            
            logger.info(f"Created {context_type} context for device: {device_id}")
            return context
            
        except Exception as e:
            logger.error(f"Failed to create device context: {e}")
            raise
    
    def search_device_issues(self, 
                            issue_description: str,
                            device_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for device issues based on description.
        
        Args:
            issue_description: Description of the issue
            device_type: Filter by device type
            
        Returns:
            List of matching issues
        """
        try:
            matching_issues = []
            
            if self.device_database:
                for device in self.device_database.get('devices', []):
                    # Apply device type filter
                    if device_type and device.get('type') != device_type:
                        continue
                    
                    # Search in common issues
                    common_issues = device.get('common_issues', [])
                    for issue in common_issues:
                        issue_text = f"{issue.get('title', '')} {issue.get('description', '')}"
                        if issue_description.lower() in issue_text.lower():
                            matching_issues.append({
                                'device_id': device.get('id'),
                                'device_name': device.get('name'),
                                'device_type': device.get('type'),
                                'issue': issue,
                                'relevance_score': self._calculate_relevance_score(issue_description, issue_text)
                            })
            
            # Sort by relevance score
            matching_issues.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            logger.info(f"Found {len(matching_issues)} matching issues for: {issue_description}")
            return matching_issues
            
        except Exception as e:
            logger.error(f"Failed to search device issues: {e}")
            raise
    
    def _calculate_relevance_score(self, query: str, text: str) -> float:
        """Calculate relevance score between query and text."""
        try:
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(query_words.intersection(text_words))
            union = len(query_words.union(text_words))
            
            if union == 0:
                return 0.0
            
            return intersection / union
            
        except Exception as e:
            logger.error(f"Failed to calculate relevance score: {e}")
            return 0.0
    
    def get_device_statistics(self) -> Dict[str, Any]:
        """
        Get device database statistics.
        
        Returns:
            Device database statistics
        """
        try:
            if not self.device_database:
                return {'error': 'Device database not loaded'}
            
            devices = self.device_database.get('devices', [])
            
            # Calculate statistics
            device_types = {}
            categories = {}
            total_issues = 0
            total_guides = 0
            
            for device in devices:
                # Count device types
                device_type = device.get('type', 'Unknown')
                device_types[device_type] = device_types.get(device_type, 0) + 1
                
                # Count categories
                category = device.get('category', 'Unknown')
                categories[category] = categories.get(category, 0) + 1
                
                # Count issues and guides
                total_issues += len(device.get('common_issues', []))
                total_guides += len(device.get('troubleshooting_guides', []))
            
            stats = {
                'total_devices': len(devices),
                'device_types': device_types,
                'categories': categories,
                'total_issues': total_issues,
                'total_guides': total_guides,
                'avg_issues_per_device': total_issues / len(devices) if devices else 0,
                'avg_guides_per_device': total_guides / len(devices) if devices else 0
            }
            
            logger.info(f"Device database statistics: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get device statistics: {e}")
            raise
