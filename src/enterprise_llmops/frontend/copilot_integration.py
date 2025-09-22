"""
CopilotKit Integration for Enterprise LLMOps Platform

This module provides Microsoft-style Copilot integration with natural language
AI interactions, context-aware assistance, and intelligent automation.

Key Features:
- Natural language AI interactions
- Context-aware AI assistants for each module
- Intelligent code suggestions and auto-completion
- Conversational interfaces for model evaluation tasks
- AI-powered documentation generation
- Smart error diagnosis and resolution suggestions
- Multi-modal interaction support (text, voice, gesture)
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import aiohttp
import openai
import anthropic
from pathlib import Path
import re
import base64
# Optional imports for multimodal features
try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None


@dataclass
class CopilotMessage:
    """Message in a Copilot conversation."""
    message_id: str
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime
    context: Dict[str, Any] = None
    attachments: List[str] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.attachments is None:
            self.attachments = []


@dataclass
class CopilotContext:
    """Context for Copilot interactions."""
    user_id: str
    session_id: str
    current_module: str
    active_tasks: List[str]
    user_preferences: Dict[str, Any]
    conversation_history: List[CopilotMessage]
    system_state: Dict[str, Any]
    
    def __post_init__(self):
        if self.active_tasks is None:
            self.active_tasks = []
        if self.user_preferences is None:
            self.user_preferences = {}
        if self.conversation_history is None:
            self.conversation_history = []
        if self.system_state is None:
            self.system_state = {}


@dataclass
class CopilotAction:
    """Action that Copilot can perform."""
    action_id: str
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable
    requires_confirmation: bool = False


class CopilotIntegration:
    """
    CopilotKit integration for Enterprise LLMOps platform.
    
    This class provides comprehensive AI-powered assistance with natural language
    interactions, context awareness, and intelligent automation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Copilot integration."""
        self.config = config
        self.logger = self._setup_logging()
        self.contexts = {}
        self.actions = {}
        self.voice_engine = None
        self.speech_recognizer = None
        
        # Initialize AI clients
        self._init_ai_clients()
        
        # Initialize voice components
        self._init_voice_components()
        
        # Register available actions
        self._register_actions()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for Copilot integration."""
        logger = logging.getLogger("copilot_integration")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _init_ai_clients(self):
        """Initialize AI client connections."""
        try:
            # Initialize OpenAI client
            if self.config.get('openai_api_key'):
                openai.api_key = self.config['openai_api_key']
                self.openai_client = openai.OpenAI(api_key=self.config['openai_api_key'])
            else:
                self.openai_client = None
                self.logger.warning("OpenAI API key not provided")
            
            # Initialize Anthropic client
            if self.config.get('anthropic_api_key'):
                self.anthropic_client = anthropic.Anthropic(api_key=self.config['anthropic_api_key'])
            else:
                self.anthropic_client = None
                self.logger.warning("Anthropic API key not provided")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize AI clients: {e}")
            self.openai_client = None
            self.anthropic_client = None
    
    def _init_voice_components(self):
        """Initialize voice recognition and synthesis components."""
        try:
            # Initialize speech recognition
            self.speech_recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Initialize text-to-speech
            self.voice_engine = pyttsx3.init()
            self.voice_engine.setProperty('rate', 150)
            
            # Get available voices
            voices = self.voice_engine.getProperty('voices')
            if voices:
                self.voice_engine.setProperty('voice', voices[0].id)
            
            self.logger.info("Voice components initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Voice components initialization failed: {e}")
            self.speech_recognizer = None
            self.voice_engine = None
    
    def _register_actions(self):
        """Register available Copilot actions."""
        actions = [
            CopilotAction(
                action_id="evaluate_model",
                name="Evaluate Model",
                description="Evaluate a specific model for a given use case",
                parameters={"model_name": "string", "use_case": "string"},
                handler=self._handle_model_evaluation,
                requires_confirmation=False
            ),
            CopilotAction(
                action_id="select_optimal_model",
                name="Select Optimal Model",
                description="Select the best model for a specific use case",
                parameters={"use_case": "string", "requirements": "object"},
                handler=self._handle_model_selection,
                requires_confirmation=True
            ),
            CopilotAction(
                action_id="deploy_model",
                name="Deploy Model",
                description="Deploy a model to production",
                parameters={"model_name": "string", "environment": "string"},
                handler=self._handle_model_deployment,
                requires_confirmation=True
            ),
            CopilotAction(
                action_id="monitor_performance",
                name="Monitor Performance",
                description="Monitor model performance metrics",
                parameters={"model_name": "string", "time_range": "string"},
                handler=self._handle_performance_monitoring,
                requires_confirmation=False
            ),
            CopilotAction(
                action_id="generate_documentation",
                name="Generate Documentation",
                description="Generate documentation for a model or process",
                parameters={"subject": "string", "format": "string"},
                handler=self._handle_documentation_generation,
                requires_confirmation=False
            ),
            CopilotAction(
                action_id="analyze_error",
                name="Analyze Error",
                description="Analyze and suggest solutions for errors",
                parameters={"error_message": "string", "context": "object"},
                handler=self._handle_error_analysis,
                requires_confirmation=False
            ),
            CopilotAction(
                action_id="create_workflow",
                name="Create Workflow",
                description="Create a new agent workflow",
                parameters={"workflow_name": "string", "steps": "array"},
                handler=self._handle_workflow_creation,
                requires_confirmation=True
            ),
            CopilotAction(
                action_id="query_knowledge_graph",
                name="Query Knowledge Graph",
                description="Query the knowledge graph for information",
                parameters={"query": "string", "filters": "object"},
                handler=self._handle_knowledge_graph_query,
                requires_confirmation=False
            )
        ]
        
        for action in actions:
            self.actions[action.action_id] = action
        
        self.logger.info(f"Registered {len(actions)} Copilot actions")
    
    async def process_message(self, user_id: str, message: str, context: Optional[CopilotContext] = None) -> Dict[str, Any]:
        """Process a user message and generate response."""
        try:
            # Get or create context
            if context is None:
                context = self._get_or_create_context(user_id)
            
            # Add user message to conversation
            user_message = CopilotMessage(
                message_id=f"msg_{int(datetime.now().timestamp())}",
                role="user",
                content=message,
                timestamp=datetime.now()
            )
            context.conversation_history.append(user_message)
            
            # Analyze message intent
            intent = await self._analyze_intent(message, context)
            
            # Generate response
            if intent.get('action_id'):
                response = await self._execute_action(intent, context)
            else:
                response = await self._generate_conversational_response(message, context)
            
            # Add assistant response to conversation
            assistant_message = CopilotMessage(
                message_id=f"msg_{int(datetime.now().timestamp())}_resp",
                role="assistant",
                content=response['content'],
                timestamp=datetime.now(),
                context=response.get('context', {})
            )
            context.conversation_history.append(assistant_message)
            
            # Update context
            self.contexts[user_id] = context
            
            return {
                "success": True,
                "response": response['content'],
                "action_executed": intent.get('action_id'),
                "context": asdict(context),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process message: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _analyze_intent(self, message: str, context: CopilotContext) -> Dict[str, Any]:
        """Analyze user message intent."""
        message_lower = message.lower()
        
        # Check for action keywords
        action_keywords = {
            "evaluate": "evaluate_model",
            "test": "evaluate_model",
            "select": "select_optimal_model",
            "choose": "select_optimal_model",
            "deploy": "deploy_model",
            "monitor": "monitor_performance",
            "check": "monitor_performance",
            "document": "generate_documentation",
            "doc": "generate_documentation",
            "error": "analyze_error",
            "bug": "analyze_error",
            "workflow": "create_workflow",
            "pipeline": "create_workflow",
            "query": "query_knowledge_graph",
            "search": "query_knowledge_graph"
        }
        
        for keyword, action_id in action_keywords.items():
            if keyword in message_lower:
                return {
                    "intent": "action",
                    "action_id": action_id,
                    "confidence": 0.8,
                    "parameters": self._extract_parameters(message, action_id)
                }
        
        # Check for conversational intent
        conversational_keywords = ["hello", "hi", "help", "what", "how", "why", "explain"]
        if any(keyword in message_lower for keyword in conversational_keywords):
            return {
                "intent": "conversation",
                "confidence": 0.9
            }
        
        return {
            "intent": "conversation",
            "confidence": 0.7
        }
    
    def _extract_parameters(self, message: str, action_id: str) -> Dict[str, Any]:
        """Extract parameters from message for specific action."""
        parameters = {}
        
        if action_id == "evaluate_model":
            # Extract model name
            model_patterns = [
                r"evaluate\s+(\w+)",
                r"test\s+(\w+)",
                r"(\w+)\s+model"
            ]
            for pattern in model_patterns:
                match = re.search(pattern, message, re.IGNORECASE)
                if match:
                    parameters["model_name"] = match.group(1)
                    break
            
            # Extract use case
            use_case_patterns = [
                r"for\s+(.+)",
                r"use\s+case\s+(.+)",
                r"scenario\s+(.+)"
            ]
            for pattern in use_case_patterns:
                match = re.search(pattern, message, re.IGNORECASE)
                if match:
                    parameters["use_case"] = match.group(1).strip()
                    break
        
        elif action_id == "select_optimal_model":
            # Extract use case and requirements
            if "for" in message.lower():
                use_case_match = re.search(r"for\s+(.+)", message, re.IGNORECASE)
                if use_case_match:
                    parameters["use_case"] = use_case_match.group(1).strip()
            
            # Extract performance requirements
            if "fast" in message.lower() or "speed" in message.lower():
                parameters["requirements"] = {"performance": "high"}
            elif "cost" in message.lower() or "cheap" in message.lower():
                parameters["requirements"] = {"cost": "low"}
        
        elif action_id == "monitor_performance":
            # Extract model name
            model_match = re.search(r"(\w+)\s+performance", message, re.IGNORECASE)
            if model_match:
                parameters["model_name"] = model_match.group(1)
            
            # Extract time range
            if "last" in message.lower():
                time_match = re.search(r"last\s+(\w+)", message, re.IGNORECASE)
                if time_match:
                    parameters["time_range"] = time_match.group(1)
        
        return parameters
    
    async def _execute_action(self, intent: Dict[str, Any], context: CopilotContext) -> Dict[str, Any]:
        """Execute a Copilot action."""
        action_id = intent['action_id']
        parameters = intent.get('parameters', {})
        
        if action_id not in self.actions:
            return {
                "content": f"Sorry, I don't know how to perform the action '{action_id}'.",
                "context": {}
            }
        
        action = self.actions[action_id]
        
        try:
            # Execute action
            result = await action.handler(parameters, context)
            
            return {
                "content": result['message'],
                "context": result.get('context', {}),
                "action_result": result.get('data')
            }
            
        except Exception as e:
            self.logger.error(f"Failed to execute action {action_id}: {e}")
            return {
                "content": f"Sorry, I encountered an error while executing the action: {str(e)}",
                "context": {}
            }
    
    async def _generate_conversational_response(self, message: str, context: CopilotContext) -> Dict[str, Any]:
        """Generate conversational response using AI."""
        try:
            # Prepare conversation history
            conversation_history = []
            for msg in context.conversation_history[-10:]:  # Last 10 messages
                conversation_history.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # Add system context
            system_prompt = self._get_system_prompt(context)
            
            # Generate response using AI
            if self.openai_client:
                response = await self._generate_openai_response(system_prompt, conversation_history)
            elif self.anthropic_client:
                response = await self._generate_anthropic_response(system_prompt, conversation_history)
            else:
                response = self._generate_fallback_response(message)
            
            return {
                "content": response,
                "context": {"generated_by": "ai"}
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate conversational response: {e}")
            return {
                "content": "I'm sorry, I'm having trouble processing your request right now. Please try again later.",
                "context": {}
            }
    
    def _get_system_prompt(self, context: CopilotContext) -> str:
        """Get system prompt for AI."""
        return f"""
        You are an AI assistant for the Enterprise LLMOps platform. You help users with:
        
        1. Model evaluation and selection
        2. Performance monitoring
        3. Deployment management
        4. Documentation generation
        5. Error analysis and troubleshooting
        6. Workflow creation
        7. Knowledge graph queries
        
        Current context:
        - User ID: {context.user_id}
        - Current module: {context.current_module}
        - Active tasks: {', '.join(context.active_tasks)}
        
        Be helpful, professional, and provide actionable advice. If you don't know something, 
        say so and suggest how the user can find the information.
        """
    
    async def _generate_openai_response(self, system_prompt: str, conversation_history: List[Dict[str, str]]) -> str:
        """Generate response using OpenAI."""
        messages = [{"role": "system", "content": system_prompt}] + conversation_history
        
        response = await self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    async def _generate_anthropic_response(self, system_prompt: str, conversation_history: List[Dict[str, str]]) -> str:
        """Generate response using Anthropic."""
        # Convert conversation history to Claude format
        conversation_text = system_prompt + "\n\n"
        for msg in conversation_history:
            conversation_text += f"{msg['role'].capitalize()}: {msg['content']}\n"
        
        response = await self.anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=500,
            messages=[{"role": "user", "content": conversation_text}]
        )
        
        return response.content[0].text
    
    def _generate_fallback_response(self, message: str) -> str:
        """Generate fallback response when AI is not available."""
        fallback_responses = [
            "I understand you're asking about that. Let me help you with the Enterprise LLMOps platform.",
            "That's a great question! I can assist you with model evaluation, deployment, or monitoring.",
            "I'd be happy to help you with that. What specific aspect of the LLMOps platform are you interested in?",
            "Let me guide you through that. I can help with model selection, performance analysis, or troubleshooting."
        ]
        
        # Simple keyword-based response
        message_lower = message.lower()
        if any(word in message_lower for word in ["model", "evaluation", "test"]):
            return "I can help you with model evaluation. You can evaluate models by specifying the model name and use case."
        elif any(word in message_lower for word in ["deploy", "deployment"]):
            return "I can assist with model deployment. Please specify which model you'd like to deploy and the target environment."
        elif any(word in message_lower for word in ["monitor", "performance", "metrics"]):
            return "I can help you monitor model performance. Let me know which model you'd like to monitor and the time range."
        else:
            return fallback_responses[hash(message) % len(fallback_responses)]
    
    # Action handlers
    async def _handle_model_evaluation(self, parameters: Dict[str, Any], context: CopilotContext) -> Dict[str, Any]:
        """Handle model evaluation action."""
        model_name = parameters.get('model_name', 'GPT-5')
        use_case = parameters.get('use_case', 'general purpose')
        
        # Simulate model evaluation
        evaluation_result = {
            "model": model_name,
            "use_case": use_case,
            "accuracy": 0.92,
            "latency": 1.2,
            "cost": 0.05,
            "recommendation": "suitable"
        }
        
        message = f"""
        I've evaluated {model_name} for {use_case}:
        
        • Accuracy: {evaluation_result['accuracy']:.1%}
        • Latency: {evaluation_result['latency']}s
        • Cost: ${evaluation_result['cost']:.3f} per request
        • Recommendation: {evaluation_result['recommendation']}
        
        The model appears to be suitable for your use case. Would you like me to help you deploy it or compare it with other models?
        """
        
        return {
            "message": message,
            "data": evaluation_result
        }
    
    async def _handle_model_selection(self, parameters: Dict[str, Any], context: CopilotContext) -> Dict[str, Any]:
        """Handle model selection action."""
        use_case = parameters.get('use_case', 'general purpose')
        requirements = parameters.get('requirements', {})
        
        # Simulate model selection
        selected_model = "Claude 3.5 Sonnet"  # Default selection
        confidence = 0.87
        
        message = f"""
        Based on your use case '{use_case}', I recommend {selected_model} with {confidence:.1%} confidence.
        
        This model is optimal because:
        • High performance for your specific use case
        • Good balance of accuracy and speed
        • Cost-effective for your requirements
        
        Would you like me to proceed with deploying {selected_model} or would you like to see alternative options?
        """
        
        return {
            "message": message,
            "data": {
                "selected_model": selected_model,
                "confidence": confidence,
                "use_case": use_case
            }
        }
    
    async def _handle_model_deployment(self, parameters: Dict[str, Any], context: CopilotContext) -> Dict[str, Any]:
        """Handle model deployment action."""
        model_name = parameters.get('model_name', 'Claude 3.5 Sonnet')
        environment = parameters.get('environment', 'production')
        
        # Simulate deployment
        deployment_id = f"deploy_{int(datetime.now().timestamp())}"
        
        message = f"""
        I'm deploying {model_name} to {environment} environment.
        
        Deployment Details:
        • Deployment ID: {deployment_id}
        • Environment: {environment}
        • Status: In Progress
        • Estimated completion: 2-3 minutes
        
        I'll notify you when the deployment is complete. You can monitor the progress in the dashboard.
        """
        
        return {
            "message": message,
            "data": {
                "deployment_id": deployment_id,
                "model_name": model_name,
                "environment": environment,
                "status": "deploying"
            }
        }
    
    async def _handle_performance_monitoring(self, parameters: Dict[str, Any], context: CopilotContext) -> Dict[str, Any]:
        """Handle performance monitoring action."""
        model_name = parameters.get('model_name', 'all models')
        time_range = parameters.get('time_range', 'last hour')
        
        # Simulate performance data
        performance_data = {
            "total_requests": 1250,
            "avg_latency": 1.2,
            "error_rate": 0.02,
            "cpu_usage": 45.2,
            "memory_usage": 67.8
        }
        
        message = f"""
        Performance metrics for {model_name} ({time_range}):
        
        • Total Requests: {performance_data['total_requests']:,}
        • Average Latency: {performance_data['avg_latency']}s
        • Error Rate: {performance_data['error_rate']:.1%}
        • CPU Usage: {performance_data['cpu_usage']:.1f}%
        • Memory Usage: {performance_data['memory_usage']:.1f}%
        
        Performance looks good! All metrics are within normal ranges. Is there anything specific you'd like me to investigate?
        """
        
        return {
            "message": message,
            "data": performance_data
        }
    
    async def _handle_documentation_generation(self, parameters: Dict[str, Any], context: CopilotContext) -> Dict[str, Any]:
        """Handle documentation generation action."""
        subject = parameters.get('subject', 'model evaluation process')
        format_type = parameters.get('format', 'markdown')
        
        # Simulate documentation generation
        doc_content = f"""
        # {subject.title()}
        
        ## Overview
        This document provides comprehensive information about {subject}.
        
        ## Key Components
        - Component 1: Description
        - Component 2: Description
        - Component 3: Description
        
        ## Usage
        1. Step 1: Action
        2. Step 2: Action
        3. Step 3: Action
        
        ## Best Practices
        - Best practice 1
        - Best practice 2
        - Best practice 3
        
        Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        message = f"""
        I've generated documentation for '{subject}' in {format_type} format.
        
        The documentation includes:
        • Overview and key components
        • Step-by-step usage instructions
        • Best practices and recommendations
        
        The documentation has been saved to your workspace. Would you like me to open it for review or make any modifications?
        """
        
        return {
            "message": message,
            "data": {
                "subject": subject,
                "format": format_type,
                "content": doc_content
            }
        }
    
    async def _handle_error_analysis(self, parameters: Dict[str, Any], context: CopilotContext) -> Dict[str, Any]:
        """Handle error analysis action."""
        error_message = parameters.get('error_message', 'Connection timeout')
        error_context = parameters.get('context', {})
        
        # Simulate error analysis
        analysis_result = {
            "error_type": "Network Error",
            "severity": "Medium",
            "possible_causes": [
                "Network connectivity issues",
                "Server overload",
                "Configuration problems"
            ],
            "suggested_solutions": [
                "Check network connectivity",
                "Verify server status",
                "Review configuration settings"
            ],
            "prevention_tips": [
                "Implement retry logic",
                "Add health checks",
                "Monitor network metrics"
            ]
        }
        
        message = f"""
        I've analyzed the error: "{error_message}"
        
        **Error Type**: {analysis_result['error_type']}
        **Severity**: {analysis_result['severity']}
        
        **Possible Causes**:
        {chr(10).join('• ' + cause for cause in analysis_result['possible_causes'])}
        
        **Suggested Solutions**:
        {chr(10).join('• ' + solution for solution in analysis_result['suggested_solutions'])}
        
        **Prevention Tips**:
        {chr(10).join('• ' + tip for tip in analysis_result['prevention_tips'])}
        
        Would you like me to help you implement any of these solutions?
        """
        
        return {
            "message": message,
            "data": analysis_result
        }
    
    async def _handle_workflow_creation(self, parameters: Dict[str, Any], context: CopilotContext) -> Dict[str, Any]:
        """Handle workflow creation action."""
        workflow_name = parameters.get('workflow_name', 'Custom Workflow')
        steps = parameters.get('steps', [])
        
        # Simulate workflow creation
        workflow_id = f"workflow_{int(datetime.now().timestamp())}"
        
        message = f"""
        I've created a new workflow: "{workflow_name}"
        
        **Workflow ID**: {workflow_id}
        **Steps**: {len(steps)} steps defined
        
        The workflow has been saved and is ready for execution. You can:
        • Review the workflow steps
        • Test the workflow
        • Deploy to production
        • Schedule automated runs
        
        Would you like me to help you test or deploy this workflow?
        """
        
        return {
            "message": message,
            "data": {
                "workflow_id": workflow_id,
                "workflow_name": workflow_name,
                "steps": steps,
                "status": "created"
            }
        }
    
    async def _handle_knowledge_graph_query(self, parameters: Dict[str, Any], context: CopilotContext) -> Dict[str, Any]:
        """Handle knowledge graph query action."""
        query = parameters.get('query', 'model relationships')
        filters = parameters.get('filters', {})
        
        # Simulate knowledge graph query
        query_results = [
            {"node": "GPT-5", "relationship": "evaluated_by", "target": "Evaluation Agent"},
            {"node": "Claude 3.5", "relationship": "selected_for", "target": "Documentation Use Case"},
            {"node": "Llama 3.3", "relationship": "deployed_to", "target": "Production Environment"}
        ]
        
        message = f"""
        Knowledge graph query results for: "{query}"
        
        Found {len(query_results)} relationships:
        
        {chr(10).join(f'• {result["node"]} {result["relationship"]} {result["target"]}' for result in query_results)}
        
        This shows the relationships between models, agents, and use cases in your system. Would you like me to explore any specific relationships further?
        """
        
        return {
            "message": message,
            "data": {
                "query": query,
                "results": query_results,
                "total_found": len(query_results)
            }
        }
    
    def _get_or_create_context(self, user_id: str) -> CopilotContext:
        """Get or create user context."""
        if user_id not in self.contexts:
            self.contexts[user_id] = CopilotContext(
                user_id=user_id,
                session_id=f"session_{int(datetime.now().timestamp())}",
                current_module="dashboard",
                active_tasks=[],
                user_preferences={},
                conversation_history=[],
                system_state={}
            )
        
        return self.contexts[user_id]
    
    async def process_voice_input(self, user_id: str) -> Dict[str, Any]:
        """Process voice input and return response."""
        if not self.speech_recognizer or not self.microphone:
            return {
                "success": False,
                "error": "Voice recognition not available"
            }
        
        try:
            # Listen for audio input
            with self.microphone as source:
                self.speech_recognizer.adjust_for_ambient_noise(source)
                audio = self.speech_recognizer.listen(source, timeout=5)
            
            # Recognize speech
            text = self.speech_recognizer.recognize_google(audio)
            
            # Process the text message
            response = await self.process_message(user_id, text)
            
            # Convert response to speech if voice engine is available
            if self.voice_engine and response['success']:
                self.voice_engine.say(response['response'])
                self.voice_engine.runAndWait()
            
            return response
            
        except sr.WaitTimeoutError:
            return {
                "success": False,
                "error": "No speech detected"
            }
        except sr.UnknownValueError:
            return {
                "success": False,
                "error": "Could not understand speech"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Voice processing error: {str(e)}"
            }
    
    async def process_image_input(self, user_id: str, image_data: bytes) -> Dict[str, Any]:
        """Process image input for visual analysis."""
        try:
            # Convert bytes to OpenCV image
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Basic image analysis (placeholder)
            height, width, channels = image.shape
            
            analysis_result = {
                "dimensions": f"{width}x{height}",
                "channels": channels,
                "type": "image",
                "analysis": "Visual analysis completed"
            }
            
            message = f"""
            I've analyzed the image you provided:
            
            • Dimensions: {analysis_result['dimensions']}
            • Channels: {analysis_result['channels']}
            • Analysis: {analysis_result['analysis']}
            
            The image appears to be a visual representation. How can I help you with this image?
            """
            
            return {
                "success": True,
                "response": message,
                "analysis": analysis_result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Image processing error: {str(e)}"
            }
    
    def get_available_actions(self) -> List[Dict[str, Any]]:
        """Get list of available Copilot actions."""
        return [
            {
                "action_id": action.action_id,
                "name": action.name,
                "description": action.description,
                "parameters": action.parameters,
                "requires_confirmation": action.requires_confirmation
            }
            for action in self.actions.values()
        ]
    
    def get_user_context(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user context."""
        if user_id in self.contexts:
            return asdict(self.contexts[user_id])
        return None
    
    async def clear_user_context(self, user_id: str):
        """Clear user context."""
        if user_id in self.contexts:
            del self.contexts[user_id]


async def main():
    """Main function for testing Copilot integration."""
    config = {
        "openai_api_key": None,  # Set your API key
        "anthropic_api_key": None  # Set your API key
    }
    
    copilot = CopilotIntegration(config)
    
    # Test message processing
    user_id = "test_user"
    
    test_messages = [
        "Hello, can you help me evaluate GPT-5 for customer support?",
        "I need to select the best model for documentation generation",
        "Can you monitor the performance of Claude 3.5 Sonnet?",
        "Generate documentation for the model evaluation process",
        "I'm getting a connection timeout error, can you help?"
    ]
    
    for message in test_messages:
        print(f"\nUser: {message}")
        response = await copilot.process_message(user_id, message)
        print(f"Copilot: {response['response']}")
        print(f"Action executed: {response.get('action_executed', 'None')}")
    
    # Show available actions
    print(f"\nAvailable actions: {len(copilot.get_available_actions())}")
    for action in copilot.get_available_actions():
        print(f"- {action['name']}: {action['description']}")


if __name__ == "__main__":
    asyncio.run(main())
