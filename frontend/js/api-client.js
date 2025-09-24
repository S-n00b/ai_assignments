/**
 * AI Platform API Client
 * Provides a unified interface for all platform services
 */

class AIServiceClient {
    constructor() {
        this.config = window.ServiceConfig;
        this.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        };
    }

    /**
     * Make a request to a service
     */
    async request(service, endpoint, options = {}) {
        const url = this.config.getServiceUrl(service, endpoint);
        
        const defaultOptions = {
            mode: 'cors',
            credentials: 'omit',
            headers: this.headers
        };
        
        const requestOptions = { ...defaultOptions, ...options };
        
        try {
            const response = await fetch(url, requestOptions);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            }
            
            return await response.text();
        } catch (error) {
            console.error(`Request failed: ${service}${endpoint}`, error);
            throw error;
        }
    }

    // ===== FastAPI Platform Methods =====
    
    /**
     * Check platform health
     */
    async checkHealth() {
        return this.request('api', '/health');
    }

    /**
     * Get platform information
     */
    async getPlatformInfo() {
        return this.request('api', '/info');
    }

    /**
     * List available models
     */
    async listModels() {
        return this.request('api', '/models');
    }

    /**
     * Generate text with a model
     */
    async generateText(modelName, prompt, parameters = {}) {
        return this.request('api', `/models/${modelName}/generate`, {
            method: 'POST',
            body: JSON.stringify({ prompt, parameters })
        });
    }

    // ===== Model Customization Methods =====
    
    /**
     * Start fine-tuning process
     */
    async startFineTuning(config) {
        return this.request('api', '/fine-tuning/start', {
            method: 'POST',
            body: JSON.stringify(config)
        });
    }

    /**
     * Get fine-tuning status
     */
    async getFineTuningStatus(jobId) {
        return this.request('api', `/fine-tuning/status/${jobId}`);
    }

    /**
     * Create QLoRA adapters
     */
    async createQLoRAAdapters(config) {
        return this.request('api', '/qlora/create', {
            method: 'POST',
            body: JSON.stringify(config)
        });
    }

    /**
     * Train custom embeddings
     */
    async trainEmbeddings(config) {
        return this.request('api', '/embeddings/train', {
            method: 'POST',
            body: JSON.stringify(config)
        });
    }

    // ===== Model Evaluation Methods =====
    
    /**
     * Test raw models
     */
    async testRawModels(models, testConfig) {
        return this.request('api', '/evaluation/raw-models', {
            method: 'POST',
            body: JSON.stringify({ models, config: testConfig })
        });
    }

    /**
     * Test custom models
     */
    async testCustomModels(models, testConfig) {
        return this.request('api', '/evaluation/custom-models', {
            method: 'POST',
            body: JSON.stringify({ models, config: testConfig })
        });
    }

    /**
     * Test agentic workflows
     */
    async testAgenticWorkflows(workflows, testConfig) {
        return this.request('api', '/evaluation/agentic-workflows', {
            method: 'POST',
            body: JSON.stringify({ workflows, config: testConfig })
        });
    }

    /**
     * Get evaluation results
     */
    async getEvaluationResults(evaluationId) {
        return this.request('api', `/evaluation/results/${evaluationId}`);
    }

    // ===== Factory Roster Methods =====
    
    /**
     * Get factory roster
     */
    async getFactoryRoster() {
        return this.request('api', '/factory-roster');
    }

    /**
     * Deploy model to production
     */
    async deployModel(modelName, deploymentConfig) {
        return this.request('api', '/factory-roster/deploy', {
            method: 'POST',
            body: JSON.stringify({ model: modelName, config: deploymentConfig })
        });
    }

    /**
     * Get model deployment status
     */
    async getDeploymentStatus(deploymentId) {
        return this.request('api', `/factory-roster/deployment/${deploymentId}`);
    }

    // ===== MLflow Methods =====
    
    /**
     * Get MLflow experiments
     */
    async getExperiments() {
        return this.request('mlflow', '/api/2.0/mlflow/experiments/list');
    }

    /**
     * Get MLflow runs
     */
    async getRuns(experimentId) {
        return this.request('mlflow', `/api/2.0/mlflow/runs/search`, {
            method: 'POST',
            body: JSON.stringify({ experiment_ids: [experimentId] })
        });
    }

    // ===== ChromaDB Methods =====
    
    /**
     * List ChromaDB collections
     */
    async listCollections() {
        return this.request('chromadb', '/api/v1/collections');
    }

    /**
     * Query ChromaDB collection
     */
    async queryCollection(collectionName, query, nResults = 5) {
        return this.request('chromadb', `/api/v1/collections/${collectionName}/query`, {
            method: 'POST',
            body: JSON.stringify({ query_texts: [query], n_results: nResults })
        });
    }

    // ===== Monitoring Methods =====
    
    /**
     * Get service metrics
     */
    async getMetrics() {
        return this.request('api', '/metrics');
    }

    /**
     * Get service logs
     */
    async getLogs(service, lines = 100) {
        return this.request('api', `/logs/${service}?lines=${lines}`);
    }

    // ===== WebSocket Methods =====
    
    /**
     * Connect to WebSocket for real-time updates
     */
    connectWebSocket(onMessage, onError) {
        const wsUrl = this.config.baseUrl.replace('https://', 'wss://').replace('http://', 'ws://');
        const ws = new WebSocket(`${wsUrl}/ws`);
        
        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                onMessage(data);
            } catch (error) {
                console.error('WebSocket message parse error:', error);
            }
        };
        
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            if (onError) onError(error);
        };
        
        ws.onclose = () => {
            console.log('WebSocket connection closed');
        };
        
        return ws;
    }
}

// Export as global variable
window.AIServiceClient = new AIServiceClient();

// Test the connection on load
window.addEventListener('DOMContentLoaded', async () => {
    try {
        const health = await window.AIServiceClient.checkHealth();
        console.log('API connection successful:', health);
    } catch (error) {
        console.warn('API connection failed:', error);
        console.log('Make sure your Codespace is running and services are started');
    }
});