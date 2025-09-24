/**
 * Service Configuration for AI Platform
 * Handles dynamic service discovery for GitHub Pages and Codespaces
 */

class ServiceConfig {
    constructor() {
        this.baseUrl = this.detectBaseUrl();
        this.services = {
            api: '/api',
            gradio: '/gradio',
            mlflow: '/mlflow',
            chromadb: '/chromadb',
            neo4j: '/neo4j'
        };
    }

    /**
     * Detect the base URL based on the current environment
     */
    detectBaseUrl() {
        const hostname = window.location.hostname;
        
        // GitHub Pages
        if (hostname === 's-n00b.github.io' || hostname.includes('github.io')) {
            // Check if we have Codespace configuration
            if (window.CODESPACE_NAME && window.GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN) {
                return `https://${window.CODESPACE_NAME}-8000.${window.GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}`;
            }
            // Fallback to a demo API or show connection instructions
            console.warn('No Codespace configuration found. Services may not be available.');
            return 'https://demo-api.example.com'; // Replace with actual demo API if available
        }
        
        // GitHub Codespaces preview
        if (hostname.includes('github.dev') || hostname.includes('app.github.dev')) {
            // Extract the codespace name from the URL
            const codespaceName = hostname.split('-')[0];
            const domain = hostname.split('.').slice(-3).join('.');
            return `https://${codespaceName}-8000.${domain}`;
        }
        
        // Local development
        if (hostname === 'localhost' || hostname === '127.0.0.1') {
            return 'http://localhost:8000';
        }
        
        // Codespaces direct access (when accessing the forwarded port directly)
        if (hostname.includes('githubpreview.dev')) {
            // Replace the current port with the gateway port
            const url = new URL(window.location.href);
            url.port = '8000';
            return url.origin;
        }
        
        // Default fallback
        console.warn('Unknown environment, defaulting to localhost');
        return 'http://localhost:8000';
    }

    /**
     * Get the full URL for a service endpoint
     */
    getServiceUrl(service, endpoint = '') {
        const serviceBase = this.services[service] || '';
        return `${this.baseUrl}${serviceBase}${endpoint}`;
    }

    /**
     * Check if services are available
     */
    async checkHealth() {
        try {
            const response = await fetch(`${this.baseUrl}/health`, {
                method: 'GET',
                mode: 'cors',
                credentials: 'omit'
            });
            return response.ok;
        } catch (error) {
            console.error('Health check failed:', error);
            return false;
        }
    }

    /**
     * Get service status for all services
     */
    async getServiceStatus() {
        const status = {};
        
        // Check main gateway
        status.gateway = await this.checkHealth();
        
        // Check individual services
        for (const [service, path] of Object.entries(this.services)) {
            try {
                const response = await fetch(`${this.baseUrl}${path}/health`, {
                    method: 'GET',
                    mode: 'cors',
                    credentials: 'omit'
                });
                status[service] = response.ok;
            } catch (error) {
                status[service] = false;
            }
        }
        
        return status;
    }

    /**
     * Display connection instructions if services are not available
     */
    showConnectionInstructions() {
        const instructions = `
            <div class="connection-instructions">
                <h3>🔌 Connect to Services</h3>
                <p>To connect this GitHub Pages site to your services:</p>
                <ol>
                    <li>Open your repository in GitHub Codespaces</li>
                    <li>Wait for all services to start (check the terminal)</li>
                    <li>Note your Codespace name and domain</li>
                    <li>Configure the connection:
                        <pre>
window.CODESPACE_NAME = 'your-codespace-name';
window.GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN = 'app.github.dev';
                        </pre>
                    </li>
                    <li>Refresh this page</li>
                </ol>
                <p>Current detected base URL: <code>${this.baseUrl}</code></p>
            </div>
        `;
        return instructions;
    }
}

// Export as global variable for use in other scripts
window.ServiceConfig = new ServiceConfig();

// Log the configuration on load
console.log('Service Configuration:', {
    baseUrl: window.ServiceConfig.baseUrl,
    environment: window.location.hostname,
    services: window.ServiceConfig.services
});