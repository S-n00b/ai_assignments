# Service Integration Diagrams

## 🎯 Overview

This section contains detailed service integration diagrams showing how all components work together in the Lenovo AAITC platform.

## 🔗 Service Integration Architecture

### Enhanced Service Integration Map with MCP Protocols

```mermaid
graph TB
    subgraph "Lenovo AAITC Enhanced Service Integration"
        subgraph "Core Platform Services"
            A[FastAPI Enterprise<br/>:8080] --> B[Unified Dashboard]
            C[Gradio App<br/>:7860] --> D[Model Evaluation]
            E[Open WebUI<br/>:8089] --> F[Model Playground]
            G[MkDocs<br/>:8082] --> H[Documentation]
        end

        subgraph "AI/ML Services"
            I[Ollama LLM<br/>:11434] --> J[Local Models]
            K[MLflow<br/>:5000] --> L[Experiment Tracking]
            M[ChromaDB<br/>:8081] --> N[Vector Storage]
            O[Neo4j<br/>:7687] --> P[Graph Database]
            Q[DuckDB<br/>Embedded] --> R[User Data Analytics]
        end

        subgraph "MCP Protocol Services"
            S[MemoryOS MCP<br/>:8084] --> T[Remote Memory]
            U[Context Engine<br/>:8085] --> V[Context Processing]
            W[RAG Orchestrator<br/>:8086] --> X[Multiple RAG Types]
            Y[NVIDIA Build API<br/>:8087] --> Z[Large Model Serving]
            AA[NeMo Agent Toolkit<br/>:8088] --> BB[Agent Orchestration]
            CC[FastMCP<br/>:8090] --> DD[Function Calls]
        end

        subgraph "Advanced Visualization"
            EE[LangGraph Studio<br/>:8083] --> FF[Agent Visualization]
            GG[QLoRA Fine-Tuning] --> HH[Model Customization]
            II[Faker Data Gen] --> JJ[Realistic Demos]
        end
    end

    A --> G
    A --> I
    A --> K
    A --> M
    A --> O
    A --> Q
    A --> S
    A --> U
    A --> W
    A --> Y
    A --> AA
    A --> CC

    C --> A
    E --> A

    style A fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    style C fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style E fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    style G fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style I fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    style K fill:#e0f2f1,stroke:#004d40,stroke-width:2px
    style M fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    style O fill:#fff8e1,stroke:#ff6f00,stroke-width:2px
    style Q fill:#e3f2fd,stroke:#0d47a1,stroke-width:2px
    style S fill:#f9fbe7,stroke:#827717,stroke-width:2px
    style U fill:#fce4ec,stroke:#ad1457,stroke-width:2px
    style W fill:#ffebee,stroke:#c62828,stroke-width:2px
    style Y fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px
    style AA fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
    style CC fill:#e0f2f1,stroke:#009688,stroke-width:2px
```

### Service Communication Flow

```mermaid
sequenceDiagram
    participant U as User
    participant F as FastAPI Platform
    participant G as Gradio App
    participant O as Ollama
    participant M as MLflow
    participant C as ChromaDB
    participant N as Neo4j
    participant P as Prometheus

    U->>F: Access Platform
    F->>G: Load Model Evaluation
    F->>O: Get Available Models
    O-->>F: Model List
    F-->>G: Model Options

    U->>G: Start Evaluation
    G->>F: Submit Evaluation Request
    F->>O: Run Model Inference
    O-->>F: Model Output
    F->>M: Log Experiment
    F->>C: Store Results
    F->>N: Update Knowledge Graph
    F->>P: Record Metrics

    F-->>G: Evaluation Results
    G-->>U: Display Results
```

### Port Configuration & Service Mapping

```mermaid
graph LR
    subgraph "Service Port Configuration"
        A[FastAPI Enterprise<br/>:8080] --> A1[Main Platform]
        B[Gradio App<br/>:7860] --> B1[Model Evaluation]
        C[MkDocs<br/>:8082] --> C1[Documentation]
        D[ChromaDB<br/>:8081] --> D1[Vector Database]
        E[MLflow<br/>:5000] --> E1[Experiment Tracking]
        F[Ollama<br/>:11434] --> F1[LLM Server]
        G[Grafana<br/>:3000] --> G1[Monitoring]
        H[LangFuse<br/>:3000] --> H1[LLM Observability]
        I[Prometheus<br/>:9090] --> I1[Metrics]
        J[Neo4j<br/>:7474] --> J1[Graph Database]
        K[Redis<br/>:6379] --> K1[Cache]
        L[PostgreSQL] --> L1[Metadata]
    end

    style A fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    style B fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style C fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    style D fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style E fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    style F fill:#e0f2f1,stroke:#004d40,stroke-width:2px
    style G fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    style H fill:#fff8e1,stroke:#ff6f00,stroke-width:2px
    style I fill:#e3f2fd,stroke:#0d47a1,stroke-width:2px
    style J fill:#f9fbe7,stroke:#827717,stroke-width:2px
    style K fill:#fce4ec,stroke:#ad1457,stroke-width:2px
    style L fill:#ffebee,stroke:#c62828,stroke-width:2px
```

## 🔄 Data Flow Integration

### Model Evaluation Data Flow

```mermaid
flowchart TD
    A[User Request] --> B[FastAPI Platform]
    B --> C[Gradio App]
    C --> D[Model Selection]
    D --> E[Ollama Inference]
    E --> F[Result Processing]
    F --> G[MLflow Logging]
    G --> H[ChromaDB Storage]
    H --> I[Neo4j Graph Update]
    I --> J[Prometheus Metrics]
    J --> K[Response to User]

    style A fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    style B fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style C fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    style D fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style E fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    style F fill:#e0f2f1,stroke:#004d40,stroke-width:2px
    style G fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    style H fill:#fff8e1,stroke:#ff6f00,stroke-width:2px
    style I fill:#e3f2fd,stroke:#0d47a1,stroke-width:2px
    style J fill:#f9fbe7,stroke:#827717,stroke-width:2px
    style K fill:#fce4ec,stroke:#ad1457,stroke-width:2px
```

### iframe Service Integration Flow

```mermaid
graph TB
    subgraph "iframe Service Integration"
        A[FastAPI Platform] --> B[iframe Manager]

        B --> C[Lenovo Pitch Page]
        B --> D[MLflow UI]
        B --> E[Gradio App]
        B --> F[ChromaDB UI]
        B --> G[MkDocs]
        B --> H[LangGraph Studio]
        B --> I[QLoRA Dashboard]
        B --> J[Neo4j Faker]

        C --> K[Unified Interface]
        D --> K
        E --> K
        F --> K
        G --> K
        H --> K
        I --> K
        J --> K

        K --> L[Service Communication]
        L --> M[Data Synchronization]
        M --> N[Unified Authentication]
    end

    style A fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    style B fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style K fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    style L fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style M fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    style N fill:#e0f2f1,stroke:#004d40,stroke-width:2px
```

## 🚀 Deployment Integration

### Local Development Setup

```mermaid
graph TB
    subgraph "Local Development Environment"
        A[PowerShell Terminal 1] --> B[ChromaDB :8081]
        C[PowerShell Terminal 2] --> D[MLflow :5000]
        E[PowerShell Terminal 3] --> F[FastAPI :8080]
        G[PowerShell Terminal 4] --> H[Gradio :7860]
        I[PowerShell Terminal 5] --> J[MkDocs :8082]

        K[Optional Services] --> L[Ollama :11434]
        K --> M[Grafana :3000]
        K --> N[Prometheus :9090]
        K --> O[Neo4j :7474]
        K --> P[Redis :6379]
    end

    F --> B
    F --> D
    F --> H
    F --> J
    F --> L
    F --> M
    F --> N
    F --> O
    F --> P

    style A fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    style C fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style E fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    style G fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style I fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    style F fill:#e0f2f1,stroke:#004d40,stroke-width:3px
```

### Production Deployment Architecture

```mermaid
graph TB
    subgraph "Production Kubernetes Cluster"
        A[Load Balancer] --> B[Ingress Controller]
        B --> C[FastAPI Service]
        B --> D[Gradio Service]
        B --> E[MLflow Service]
        B --> F[ChromaDB Service]

        C --> G[FastAPI Pods]
        D --> H[Gradio Pods]
        E --> I[MLflow Pods]
        F --> J[ChromaDB Pods]

        K[Monitoring Stack] --> L[Prometheus]
        K --> M[Grafana]
        K --> N[LangFuse]

        O[Data Layer] --> P[PostgreSQL]
        O --> Q[Redis]
        O --> R[Neo4j]

        G --> P
        G --> Q
        G --> R
        G --> L
    end

    style A fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    style B fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style C fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    style D fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style E fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    style F fill:#e0f2f1,stroke:#004d40,stroke-width:2px
    style K fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    style O fill:#fff8e1,stroke:#ff6f00,stroke-width:2px
```

## 🔧 Service Health & Monitoring

### Health Check Flow

```mermaid
graph LR
    A[Health Check Request] --> B[FastAPI Platform]
    B --> C[Service Discovery]
    C --> D[Health Endpoints]

    D --> E[ChromaDB Health]
    D --> F[MLflow Health]
    D --> G[Ollama Health]
    D --> H[Neo4j Health]
    D --> I[Redis Health]

    E --> J[Health Status]
    F --> J
    G --> J
    H --> J
    I --> J

    J --> K[Status Dashboard]
    K --> L[Alert System]

    style A fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    style B fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style C fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    style D fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style J fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    style K fill:#e0f2f1,stroke:#004d40,stroke-width:2px
    style L fill:#f1f8e9,stroke:#33691e,stroke-width:2px
```

---

**Last Updated**: January 19, 2025  
**Version**: 2.1.0  
**Status**: Production Ready  
**Integration**: Complete Service Integration
