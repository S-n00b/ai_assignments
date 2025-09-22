"""
Simplified Evaluation Interface for Gradio App

This module provides a streamlined evaluation interface that works with the
unified registry system and supports both local and remote model evaluation.
"""

import gradio as gr
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
import time

from ..unified_registry import UnifiedRegistryManager, UnifiedModelObject, ModelServingInterface, ServingRequest

logger = logging.getLogger(__name__)


class SimplifiedEvaluationInterface:
    """Simplified evaluation interface for Gradio app."""
    
    def __init__(self, registry_manager: UnifiedRegistryManager):
        """Initialize the simplified evaluation interface."""
        self.registry_manager = registry_manager
        self.serving_interface = None
        self.current_model = None
        self.evaluation_history = []
        
    async def initialize_serving(self):
        """Initialize the serving interface."""
        if not self.serving_interface:
            self.serving_interface = ModelServingInterface(self.registry_manager)
            await self.serving_interface.__aenter__()
    
    def create_evaluation_interface(self) -> Tuple[gr.Textbox, gr.Dropdown, gr.Button, gr.HTML, gr.Plot]:
        """Create the simplified evaluation interface."""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 🧪 Model Evaluation")
                gr.Markdown("""
                **Streamlined Evaluation:**
                - Unified interface for local and remote models
                - Real-time inference and results
                - Performance metrics tracking
                - Evaluation history
                """)
                
                # Input text
                input_text = gr.Textbox(
                    label="Evaluation Input",
                    placeholder="Enter your prompt or question here...",
                    lines=5,
                    info="Enter the text you want to evaluate with the selected model"
                )
                
                # Model parameters
                with gr.Accordion("Model Parameters", open=False):
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                        info="Controls randomness in generation"
                    )
                    
                    max_tokens = gr.Slider(
                        minimum=1,
                        maximum=4000,
                        value=1000,
                        step=1,
                        label="Max Tokens",
                        info="Maximum number of tokens to generate"
                    )
                    
                    top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Top P",
                        info="Controls diversity via nucleus sampling"
                    )
                
                # Evaluate button
                evaluate_button = gr.Button("🚀 Evaluate Model", variant="primary")
                
                # Clear button
                clear_button = gr.Button("🗑️ Clear", variant="secondary")
            
            with gr.Column(scale=2):
                # Results display
                results_output = gr.HTML(
                    value="<div style='text-align: center; padding: 2rem; color: #666;'>Select a model and enter input text to start evaluation</div>"
                )
                
                # Performance metrics chart
                metrics_chart = gr.Plot(
                    label="Performance Metrics"
                )
        
        return input_text, temperature, max_tokens, top_p, evaluate_button, clear_button, results_output, metrics_chart
    
    async def evaluate_model(
        self,
        input_text: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        top_p: float
    ) -> Tuple[str, Any]:
        """Evaluate the selected model with given input."""
        try:
            if not input_text.strip():
                return "<div style='color: orange; padding: 1rem;'>Please enter some text to evaluate</div>", None
            
            if not model_id:
                return "<div style='color: orange; padding: 1rem;'>Please select a model first</div>", None
            
            # Initialize serving interface if needed
            await self.initialize_serving()
            
            # Get model from registry
            model = await self.registry_manager.get_model(model_id)
            if not model:
                return f"<div style='color: red; padding: 1rem;'>Model {model_id} not found</div>", None
            
            self.current_model = model
            
            logger.info(f"Evaluating model {model_id} with input: {input_text[:100]}...")
            
            # Prepare serving request
            request = ServingRequest(
                model_id=model_id,
                messages=[{"role": "user", "content": input_text}],
                parameters={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p
                }
            )
            
            # Make inference
            response = await self.serving_interface.make_inference(request)
            
            # Record evaluation
            evaluation_record = {
                "timestamp": time.time(),
                "model_id": model_id,
                "input": input_text,
                "response": response.response,
                "latency_ms": response.latency_ms,
                "success": response.success,
                "error": response.error,
                "serving_type": response.serving_type
            }
            self.evaluation_history.append(evaluation_record)
            
            # Create results HTML
            results_html = self._create_results_html(response, model, input_text)
            
            # Create performance chart
            chart = self._create_performance_chart()
            
            return results_html, chart
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            error_html = f"""
            <div style='color: red; padding: 1rem; border: 1px solid #F44336; border-radius: 4px;'>
                <strong>❌ Evaluation Failed</strong><br>
                Error: {str(e)}
            </div>
            """
            return error_html, None
    
    def _create_results_html(self, response: Any, model: UnifiedModelObject, input_text: str) -> str:
        """Create HTML display for evaluation results."""
        if response.success:
            # Extract response content
            if isinstance(response.response, dict):
                if "choices" in response.response and len(response.response["choices"]) > 0:
                    output_text = response.response["choices"][0].get("message", {}).get("content", str(response.response))
                elif "response" in response.response:
                    output_text = response.response["response"]
                else:
                    output_text = str(response.response)
            else:
                output_text = str(response.response)
            
            # Create success HTML
            results_html = f"""
            <div style='padding: 1.5rem; border: 1px solid #e0e0e0; border-radius: 8px; background: #f9f9f9;'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;'>
                    <h3 style='margin: 0; color: #333;'>✅ Evaluation Results</h3>
                    <span style='background: #4CAF50; color: white; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem;'>
                        {response.serving_type.upper()}
                    </span>
                </div>
                
                <div style='margin-bottom: 1rem;'>
                    <strong>Model:</strong> {model.name} ({model.get_source_display_name()})<br>
                    <strong>Category:</strong> {model.get_category_display_name()}<br>
                    <strong>Latency:</strong> {response.latency_ms:.2f} ms
                </div>
                
                <div style='margin-bottom: 1rem;'>
                    <strong>Input:</strong><br>
                    <div style='background: white; padding: 0.75rem; border-radius: 4px; border: 1px solid #ddd; margin-top: 0.5rem;'>
                        {input_text}
                    </div>
                </div>
                
                <div>
                    <strong>Output:</strong><br>
                    <div style='background: white; padding: 0.75rem; border-radius: 4px; border: 1px solid #ddd; margin-top: 0.5rem; white-space: pre-wrap;'>
                        {output_text}
                    </div>
                </div>
            </div>
            """
        else:
            # Create error HTML
            results_html = f"""
            <div style='padding: 1.5rem; border: 1px solid #F44336; border-radius: 8px; background: #ffebee;'>
                <h3 style='margin: 0 0 1rem 0; color: #F44336;'>❌ Evaluation Failed</h3>
                
                <div style='margin-bottom: 1rem;'>
                    <strong>Model:</strong> {model.name} ({model.get_source_display_name()})<br>
                    <strong>Error:</strong> {response.error or 'Unknown error'}
                </div>
                
                <div style='margin-bottom: 1rem;'>
                    <strong>Input:</strong><br>
                    <div style='background: white; padding: 0.75rem; border-radius: 4px; border: 1px solid #ddd; margin-top: 0.5rem;'>
                        {input_text}
                    </div>
                </div>
            </div>
            """
        
        return results_html
    
    def _create_performance_chart(self) -> Any:
        """Create performance metrics chart."""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            
            if not self.evaluation_history:
                # Create empty chart
                fig = go.Figure()
                fig.add_annotation(
                    text="No evaluation data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                fig.update_layout(
                    title="Performance Metrics",
                    xaxis_title="Evaluation",
                    yaxis_title="Latency (ms)"
                )
                return fig
            
            # Extract data for chart
            latencies = [eval_record["latency_ms"] for eval_record in self.evaluation_history if eval_record["success"]]
            timestamps = [eval_record["timestamp"] for eval_record in self.evaluation_history if eval_record["success"]]
            
            if not latencies:
                # No successful evaluations
                fig = go.Figure()
                fig.add_annotation(
                    text="No successful evaluations",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                fig.update_layout(
                    title="Performance Metrics",
                    xaxis_title="Evaluation",
                    yaxis_title="Latency (ms)"
                )
                return fig
            
            # Create latency trend chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(latencies))),
                y=latencies,
                mode='lines+markers',
                name='Latency (ms)',
                line=dict(color='#2196F3', width=2),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title="Model Performance - Latency Trend",
                xaxis_title="Evaluation Number",
                yaxis_title="Latency (ms)",
                hovermode='x unified',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating performance chart: {e}")
            return None
    
    async def clear_evaluation(self) -> Tuple[str, Any]:
        """Clear evaluation results."""
        try:
            self.evaluation_history.clear()
            
            clear_html = "<div style='text-align: center; padding: 2rem; color: #666;'>Evaluation history cleared</div>"
            
            # Create empty chart
            try:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_annotation(
                    text="No evaluation data",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                fig.update_layout(
                    title="Performance Metrics",
                    xaxis_title="Evaluation",
                    yaxis_title="Latency (ms)"
                )
                return clear_html, fig
            except:
                return clear_html, None
                
        except Exception as e:
            logger.error(f"Error clearing evaluation: {e}")
            return f"<div style='color: red; padding: 1rem;'>Error clearing evaluation: {str(e)}</div>", None
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        if not self.evaluation_history:
            return {
                "total_evaluations": 0,
                "successful_evaluations": 0,
                "failed_evaluations": 0,
                "average_latency_ms": 0.0,
                "models_evaluated": set()
            }
        
        successful_evaluations = [e for e in self.evaluation_history if e["success"]]
        failed_evaluations = [e for e in self.evaluation_history if not e["success"]]
        
        avg_latency = 0.0
        if successful_evaluations:
            avg_latency = sum(e["latency_ms"] for e in successful_evaluations) / len(successful_evaluations)
        
        models_evaluated = set(e["model_id"] for e in self.evaluation_history)
        
        return {
            "total_evaluations": len(self.evaluation_history),
            "successful_evaluations": len(successful_evaluations),
            "failed_evaluations": len(failed_evaluations),
            "average_latency_ms": avg_latency,
            "models_evaluated": list(models_evaluated),
            "success_rate": len(successful_evaluations) / len(self.evaluation_history) if self.evaluation_history else 0.0
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.serving_interface:
            await self.serving_interface.__aexit__(None, None, None)
            self.serving_interface = None
