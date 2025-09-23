"""
Customer Journey RAG for Hybrid RAG

This module provides customer journey-aware RAG capabilities for Lenovo
customer interactions, B2B scenarios, and customer support workflows.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CustomerJourneyRAG:
    """
    Customer journey-aware RAG system.
    
    Provides customer journey-aware retrieval for Lenovo customer interactions,
    B2B scenarios, and customer support workflows.
    """
    
    def __init__(self, 
                 chromadb_client=None,
                 neo4j_client=None,
                 customer_database_path: Optional[str] = None):
        """
        Initialize customer journey RAG.
        
        Args:
            chromadb_client: ChromaDB client
            neo4j_client: Neo4j client
            customer_database_path: Path to customer database
        """
        self.chromadb_client = chromadb_client
        self.neo4j_client = neo4j_client
        self.customer_database_path = customer_database_path
        self.customer_database = None
        
        if customer_database_path:
            self.load_customer_database()
    
    def load_customer_database(self) -> None:
        """Load customer database from file."""
        try:
            if self.customer_database_path and Path(self.customer_database_path).exists():
                with open(self.customer_database_path, 'r', encoding='utf-8') as f:
                    self.customer_database = json.load(f)
                logger.info(f"Loaded customer database from {self.customer_database_path}")
            else:
                logger.warning(f"Customer database not found at {self.customer_database_path}")
                
        except Exception as e:
            logger.error(f"Failed to load customer database: {e}")
            raise
    
    def get_customer_profile(self, 
                           customer_id: str,
                           include_history: bool = True) -> Dict[str, Any]:
        """
        Get customer profile information.
        
        Args:
            customer_id: Customer ID
            include_history: Include interaction history
            
        Returns:
            Customer profile
        """
        try:
            # Search in customer database
            customer_info = None
            if self.customer_database:
                for customer in self.customer_database.get('customers', []):
                    if customer.get('id') == customer_id:
                        customer_info = customer
                        break
            
            if not customer_info:
                return {'error': f'Customer {customer_id} not found'}
            
            # Get basic profile
            profile = {
                'id': customer_info.get('id'),
                'name': customer_info.get('name'),
                'type': customer_info.get('type'),
                'industry': customer_info.get('industry'),
                'contact_info': customer_info.get('contact_info', {}),
                'preferences': customer_info.get('preferences', {}),
                'company_info': customer_info.get('company_info', {})
            }
            
            # Include interaction history if requested
            if include_history:
                profile['interaction_history'] = customer_info.get('interaction_history', [])
                profile['support_tickets'] = customer_info.get('support_tickets', [])
                profile['purchase_history'] = customer_info.get('purchase_history', [])
            
            logger.info(f"Retrieved profile for customer: {customer_id}")
            return profile
            
        except Exception as e:
            logger.error(f"Failed to get customer profile: {e}")
            raise
    
    def get_customer_journey(self, 
                           customer_id: str,
                           journey_type: str = "complete") -> Dict[str, Any]:
        """
        Get customer journey information.
        
        Args:
            customer_id: Customer ID
            journey_type: Type of journey (complete, support, sales, technical)
            
        Returns:
            Customer journey information
        """
        try:
            # Get customer profile
            customer_profile = self.get_customer_profile(customer_id, include_history=True)
            
            if 'error' in customer_profile:
                return customer_profile
            
            # Get journey based on type
            if journey_type == "complete":
                journey = {
                    'customer_id': customer_id,
                    'profile': customer_profile,
                    'journey_stages': self._get_journey_stages(customer_profile),
                    'touchpoints': self._get_customer_touchpoints(customer_profile),
                    'pain_points': self._get_customer_pain_points(customer_profile),
                    'satisfaction_score': self._calculate_satisfaction_score(customer_profile)
                }
            elif journey_type == "support":
                journey = {
                    'customer_id': customer_id,
                    'support_interactions': customer_profile.get('interaction_history', []),
                    'support_tickets': customer_profile.get('support_tickets', []),
                    'common_issues': self._get_common_support_issues(customer_profile),
                    'resolution_times': self._get_resolution_times(customer_profile)
                }
            elif journey_type == "sales":
                journey = {
                    'customer_id': customer_id,
                    'purchase_history': customer_profile.get('purchase_history', []),
                    'sales_interactions': self._get_sales_interactions(customer_profile),
                    'product_interest': self._get_product_interest(customer_profile),
                    'sales_stage': self._get_sales_stage(customer_profile)
                }
            elif journey_type == "technical":
                journey = {
                    'customer_id': customer_id,
                    'technical_interactions': self._get_technical_interactions(customer_profile),
                    'technical_issues': self._get_technical_issues(customer_profile),
                    'technical_preferences': self._get_technical_preferences(customer_profile)
                }
            else:
                raise ValueError(f"Unsupported journey type: {journey_type}")
            
            logger.info(f"Retrieved {journey_type} journey for customer: {customer_id}")
            return journey
            
        except Exception as e:
            logger.error(f"Failed to get customer journey: {e}")
            raise
    
    def _get_journey_stages(self, customer_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get customer journey stages."""
        try:
            stages = []
            
            # Define journey stages
            stage_definitions = {
                'awareness': 'Customer becomes aware of Lenovo products',
                'consideration': 'Customer considers Lenovo solutions',
                'purchase': 'Customer makes a purchase',
                'onboarding': 'Customer onboarding process',
                'usage': 'Customer uses products',
                'support': 'Customer seeks support',
                'renewal': 'Customer considers renewal/upgrade'
            }
            
            # Analyze customer data to determine stages
            interaction_history = customer_profile.get('interaction_history', [])
            purchase_history = customer_profile.get('purchase_history', [])
            support_tickets = customer_profile.get('support_tickets', [])
            
            # Determine which stages the customer has gone through
            for stage, description in stage_definitions.items():
                stage_data = {
                    'stage': stage,
                    'description': description,
                    'completed': False,
                    'completion_date': None,
                    'interactions': []
                }
                
                # Check if stage is completed based on data
                if stage == 'awareness' and interaction_history:
                    stage_data['completed'] = True
                    stage_data['completion_date'] = interaction_history[0].get('date')
                    stage_data['interactions'] = [i for i in interaction_history if i.get('type') == 'awareness']
                
                elif stage == 'purchase' and purchase_history:
                    stage_data['completed'] = True
                    stage_data['completion_date'] = purchase_history[0].get('date')
                    stage_data['interactions'] = purchase_history
                
                elif stage == 'support' and support_tickets:
                    stage_data['completed'] = True
                    stage_data['completion_date'] = support_tickets[0].get('date')
                    stage_data['interactions'] = support_tickets
                
                stages.append(stage_data)
            
            return stages
            
        except Exception as e:
            logger.error(f"Failed to get journey stages: {e}")
            return []
    
    def _get_customer_touchpoints(self, customer_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get customer touchpoints."""
        try:
            touchpoints = []
            
            # Get all interactions
            interaction_history = customer_profile.get('interaction_history', [])
            support_tickets = customer_profile.get('support_tickets', [])
            purchase_history = customer_profile.get('purchase_history', [])
            
            # Combine all touchpoints
            all_touchpoints = []
            all_touchpoints.extend(interaction_history)
            all_touchpoints.extend(support_tickets)
            all_touchpoints.extend(purchase_history)
            
            # Sort by date
            all_touchpoints.sort(key=lambda x: x.get('date', ''), reverse=True)
            
            # Format touchpoints
            for touchpoint in all_touchpoints:
                formatted_touchpoint = {
                    'date': touchpoint.get('date'),
                    'type': touchpoint.get('type', 'interaction'),
                    'channel': touchpoint.get('channel', 'unknown'),
                    'description': touchpoint.get('description', ''),
                    'outcome': touchpoint.get('outcome', ''),
                    'satisfaction': touchpoint.get('satisfaction', None)
                }
                touchpoints.append(formatted_touchpoint)
            
            return touchpoints
            
        except Exception as e:
            logger.error(f"Failed to get customer touchpoints: {e}")
            return []
    
    def _get_customer_pain_points(self, customer_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get customer pain points."""
        try:
            pain_points = []
            
            # Analyze support tickets for pain points
            support_tickets = customer_profile.get('support_tickets', [])
            for ticket in support_tickets:
                if ticket.get('status') == 'open' or ticket.get('priority') == 'high':
                    pain_points.append({
                        'type': 'support_issue',
                        'description': ticket.get('description', ''),
                        'priority': ticket.get('priority', 'medium'),
                        'date': ticket.get('date'),
                        'status': ticket.get('status')
                    })
            
            # Analyze interaction history for pain points
            interaction_history = customer_profile.get('interaction_history', [])
            for interaction in interaction_history:
                if interaction.get('satisfaction', 5) < 3:  # Low satisfaction
                    pain_points.append({
                        'type': 'satisfaction_issue',
                        'description': interaction.get('description', ''),
                        'satisfaction_score': interaction.get('satisfaction'),
                        'date': interaction.get('date')
                    })
            
            return pain_points
            
        except Exception as e:
            logger.error(f"Failed to get customer pain points: {e}")
            return []
    
    def _calculate_satisfaction_score(self, customer_profile: Dict[str, Any]) -> float:
        """Calculate customer satisfaction score."""
        try:
            # Get satisfaction scores from interactions
            interaction_history = customer_profile.get('interaction_history', [])
            satisfaction_scores = [i.get('satisfaction', 5) for i in interaction_history if i.get('satisfaction')]
            
            if not satisfaction_scores:
                return 5.0  # Default score
            
            # Calculate average satisfaction
            average_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores)
            
            # Adjust based on support tickets
            support_tickets = customer_profile.get('support_tickets', [])
            open_tickets = len([t for t in support_tickets if t.get('status') == 'open'])
            
            # Reduce score for open tickets
            if open_tickets > 0:
                average_satisfaction -= open_tickets * 0.5
            
            return max(0.0, min(10.0, average_satisfaction))  # Clamp between 0 and 10
            
        except Exception as e:
            logger.error(f"Failed to calculate satisfaction score: {e}")
            return 5.0
    
    def _get_common_support_issues(self, customer_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get common support issues for customer."""
        try:
            support_tickets = customer_profile.get('support_tickets', [])
            
            # Count issue types
            issue_counts = {}
            for ticket in support_tickets:
                issue_type = ticket.get('issue_type', 'unknown')
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
            
            # Sort by count
            common_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
            
            return [{'issue_type': issue, 'count': count} for issue, count in common_issues]
            
        except Exception as e:
            logger.error(f"Failed to get common support issues: {e}")
            return []
    
    def _get_resolution_times(self, customer_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Get resolution times for support tickets."""
        try:
            support_tickets = customer_profile.get('support_tickets', [])
            
            resolution_times = []
            for ticket in support_tickets:
                if ticket.get('status') == 'closed' and ticket.get('resolution_time'):
                    resolution_times.append(ticket.get('resolution_time'))
            
            if not resolution_times:
                return {'average': 0, 'min': 0, 'max': 0, 'count': 0}
            
            return {
                'average': sum(resolution_times) / len(resolution_times),
                'min': min(resolution_times),
                'max': max(resolution_times),
                'count': len(resolution_times)
            }
            
        except Exception as e:
            logger.error(f"Failed to get resolution times: {e}")
            return {'average': 0, 'min': 0, 'max': 0, 'count': 0}
    
    def _get_sales_interactions(self, customer_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get sales interactions for customer."""
        try:
            interaction_history = customer_profile.get('interaction_history', [])
            sales_interactions = [i for i in interaction_history if i.get('type') == 'sales']
            return sales_interactions
            
        except Exception as e:
            logger.error(f"Failed to get sales interactions: {e}")
            return []
    
    def _get_product_interest(self, customer_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get product interest for customer."""
        try:
            preferences = customer_profile.get('preferences', {})
            product_interest = preferences.get('product_interest', [])
            return product_interest
            
        except Exception as e:
            logger.error(f"Failed to get product interest: {e}")
            return []
    
    def _get_sales_stage(self, customer_profile: Dict[str, Any]) -> str:
        """Get current sales stage for customer."""
        try:
            purchase_history = customer_profile.get('purchase_history', [])
            sales_interactions = self._get_sales_interactions(customer_profile)
            
            if purchase_history:
                return 'customer'
            elif sales_interactions:
                return 'prospect'
            else:
                return 'lead'
                
        except Exception as e:
            logger.error(f"Failed to get sales stage: {e}")
            return 'unknown'
    
    def _get_technical_interactions(self, customer_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get technical interactions for customer."""
        try:
            interaction_history = customer_profile.get('interaction_history', [])
            technical_interactions = [i for i in interaction_history if i.get('type') == 'technical']
            return technical_interactions
            
        except Exception as e:
            logger.error(f"Failed to get technical interactions: {e}")
            return []
    
    def _get_technical_issues(self, customer_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get technical issues for customer."""
        try:
            support_tickets = customer_profile.get('support_tickets', [])
            technical_issues = [t for t in support_tickets if t.get('category') == 'technical']
            return technical_issues
            
        except Exception as e:
            logger.error(f"Failed to get technical issues: {e}")
            return []
    
    def _get_technical_preferences(self, customer_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Get technical preferences for customer."""
        try:
            preferences = customer_profile.get('preferences', {})
            technical_preferences = preferences.get('technical', {})
            return technical_preferences
            
        except Exception as e:
            logger.error(f"Failed to get technical preferences: {e}")
            return {}
    
    def search_customer_context(self, 
                              query: str,
                              customer_id: Optional[str] = None,
                              context_type: str = "comprehensive") -> List[Dict[str, Any]]:
        """
        Search customer context based on query.
        
        Args:
            query: Search query
            customer_id: Customer ID to search within
            context_type: Type of context to search
            
        Returns:
            List of relevant context
        """
        try:
            results = []
            
            if customer_id:
                # Search within specific customer
                customer_profile = self.get_customer_profile(customer_id, include_history=True)
                if 'error' not in customer_profile:
                    # Search in customer data
                    customer_text = f"{customer_profile.get('name', '')} {customer_profile.get('type', '')} {customer_profile.get('industry', '')}"
                    if query.lower() in customer_text.lower():
                        results.append({
                            'type': 'customer_profile',
                            'customer_id': customer_id,
                            'content': customer_text,
                            'relevance_score': 1.0
                        })
                    
                    # Search in interaction history
                    interaction_history = customer_profile.get('interaction_history', [])
                    for interaction in interaction_history:
                        interaction_text = f"{interaction.get('description', '')} {interaction.get('type', '')}"
                        if query.lower() in interaction_text.lower():
                            results.append({
                                'type': 'interaction',
                                'customer_id': customer_id,
                                'content': interaction_text,
                                'relevance_score': 0.8
                            })
            else:
                # Search across all customers
                if self.customer_database:
                    for customer in self.customer_database.get('customers', []):
                        customer_text = f"{customer.get('name', '')} {customer.get('type', '')} {customer.get('industry', '')}"
                        if query.lower() in customer_text.lower():
                            results.append({
                                'type': 'customer_profile',
                                'customer_id': customer.get('id'),
                                'content': customer_text,
                                'relevance_score': 1.0
                            })
            
            # Sort by relevance score
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            logger.info(f"Found {len(results)} customer context results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search customer context: {e}")
            raise
    
    def get_customer_statistics(self) -> Dict[str, Any]:
        """
        Get customer database statistics.
        
        Returns:
            Customer database statistics
        """
        try:
            if not self.customer_database:
                return {'error': 'Customer database not loaded'}
            
            customers = self.customer_database.get('customers', [])
            
            # Calculate statistics
            customer_types = {}
            industries = {}
            total_interactions = 0
            total_support_tickets = 0
            
            for customer in customers:
                # Count customer types
                customer_type = customer.get('type', 'Unknown')
                customer_types[customer_type] = customer_types.get(customer_type, 0) + 1
                
                # Count industries
                industry = customer.get('industry', 'Unknown')
                industries[industry] = industries.get(industry, 0) + 1
                
                # Count interactions and tickets
                total_interactions += len(customer.get('interaction_history', []))
                total_support_tickets += len(customer.get('support_tickets', []))
            
            stats = {
                'total_customers': len(customers),
                'customer_types': customer_types,
                'industries': industries,
                'total_interactions': total_interactions,
                'total_support_tickets': total_support_tickets,
                'avg_interactions_per_customer': total_interactions / len(customers) if customers else 0,
                'avg_tickets_per_customer': total_support_tickets / len(customers) if customers else 0
            }
            
            logger.info(f"Customer database statistics: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get customer statistics: {e}")
            raise
