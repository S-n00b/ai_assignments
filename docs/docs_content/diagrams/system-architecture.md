# System Architecture Diagrams

## 🎯 Overview

This section contains comprehensive system architecture diagrams for the Lenovo AAITC AI Assignments platform, showcasing the enterprise-grade architecture and service integration.

## 🏗️ Enterprise AI Architecture

### Overall System Architecture

```mermaid
graph TB
    subgraph "Lenovo AAITC Enterprise Platform"
        subgraph "Frontend Layer"
            A[FastAPI Enterprise Platform<br/>Port 8080] --> B[Gradio Model Evaluation<br/>Port 7860]
            A --> C[MkDocs Documentation<br/>Port 8082]
            A --> D[LangGraph Studio<br/>Agent Visualization]
        end
        
        subgraph "AI/ML Services"
            E[Ollama LLM Server<br/>Port 11434] --> F[Model Registry]
            G[MLflow Tracking<br/>Port 5000] --> H[Experiment Store]
            I[ChromaDB Vector DB<br/>Port 8081] --> J[Embeddings Store]
        end
        
        subgraph "Monitoring & Observability"
            K[Prometheus<br/>Port 9090] --> L[Metrics Collection]
            M[Grafana<br/>Port 3000] --> N[Dashboards]
            O[LangFuse<br/>Port 3000] --> P[LLM Observability]
        end
        
        subgraph "Data & Storage"
            Q[Neo4j Graph DB<br/>Port 7474] --> R[Knowledge Graph]
            S[Redis Cache<br/>Port 6379] --> T[Session Store]
            U[PostgreSQL] --> V[Metadata Store]
        end
    end
    
    A --> E
    A --> G
    A --> I
    A --> K
    A --> M
    A --> O
    A --> Q
    A --> S
    
    style A fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    style B fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style C fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    style D fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style E fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    style G fill:#e0f2f1,stroke:#004d40,stroke-width:2px
    style I fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    style K fill:#fff8e1,stroke:#ff6f00,stroke-width:2px
    style M fill:#e3f2fd,stroke:#0d47a1,stroke-width:2px
    style O fill:#f9fbe7,stroke:#827717,stroke-width:2px
    style Q fill:#fce4ec,stroke:#ad1457,stroke-width:2px
    style S fill:#ffebee,stroke:#c62828,stroke-width:2px
```

### Assignment 1: Model Evaluation Engineer Architecture

```mermaid
graph LR
    subgraph "Model Evaluation Engineer (Assignment 1)"
        A[Gradio Interface<br/>6 Tabs] --> B[Evaluation Pipeline]
        A --> C[Model Profiling]
        A --> D[Model Factory]
        A --> E[Practical Exercise]
        A --> F[Dashboard]
        A --> G[Reports]
        
        B --> H[ModelProfiler Class]
        C --> I[Performance Metrics]
        D --> J[ModelFactory Class]
        E --> K[Lenovo Documentation]
        F --> L[Real-time Visualization]
        G --> M[Export & Reporting]
    end
    
    subgraph "Enterprise Integration"
        N[FastAPI Platform] --> A
        O[MLflow Tracking] --> B
        P[Model Registry] --> D
        Q[Vector Database] --> E
    end
    
    style A fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    style B fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style C fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    style D fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style E fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    style F fill:#e0f2f1,stroke:#004d40,stroke-width:2px
    style G fill:#f1f8e9,stroke:#33691e,stroke-width:2px
```

### Assignment 2: AI Architect Enterprise Platform

```mermaid
graph TB
    subgraph "AI Architect Enterprise Platform (Assignment 2)"
        subgraph "Core Services"
            A[FastAPI Enterprise App<br/>Port 8080] --> B[Model Management]
            A --> C[Experiment Tracking]
            A --> D[Vector Search]
            A --> E[Agent Orchestration]
        end
        
        subgraph "Advanced Features"
            F[QLoRA Fine-Tuning] --> G[Adapter Management]
            H[LangGraph Studio] --> I[Agent Visualization]
            J[Neo4j GraphRAG] --> K[Knowledge Graph]
            L[Faker Data Gen] --> M[Realistic Demos]
        end
        
        subgraph "Infrastructure"
            N[Kubernetes] --> O[Container Orchestration]
            P[Docker] --> Q[Service Containers]
            R[Terraform] --> S[Infrastructure as Code]
        end
    end
    
    A --> F
    A --> H
    A --> J
    A --> L
    A --> N
    A --> P
    A --> R
    
    style A fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    style F fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style H fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    style J fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style L fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    style N fill:#e0f2f1,stroke:#004d40,stroke-width:2px
    style P fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    style R fill:#fff8e1,stroke:#ff6f00,stroke-width:2px
```

## 🔧 Service Integration Architecture

### Port Configuration & Service Mapping

```mermaid
graph LR
    subgraph "Service Port Configuration"
        A[FastAPI Enterprise<br/>:8080] --> B[Main Platform]
        C[Gradio App<br/>:7860] --> D[Model Evaluation]
        E[MkDocs<br/>:8082] --> F[Documentation]
        G[ChromaDB<br/>:8081] --> H[Vector Database]
        I[MLflow<br/>:5000] --> J[Experiment Tracking]
        K[Ollama<br/>:11434] --> L[LLM Server]
        M[Grafana<br/>:3000] --> N[Monitoring]
        O[LangFuse<br/>:3000] --> P[LLM Observability]
        Q[Prometheus<br/>:9090] --> R[Metrics]
        S[Neo4j<br/>:7474] --> T[Graph Database]
        U[Redis<br/>:6379] --> V[Cache]
    end
    
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
```

## 📊 Data Flow Architecture

### Model Evaluation Data Flow

```mermaid
sequenceDiagram
    participant U as User
    participant G as Gradio App
    participant F as FastAPI Platform
    participant M as MLflow
    participant V as Vector DB
    participant O as Ollama
    
    U->>G: Start Evaluation
    G->>F: Request Model List
    F->>O: Get Available Models
    O-->>F: Model Registry
    F-->>G: Model Options
    G->>F: Submit Evaluation
    F->>M: Log Experiment
    F->>V: Store Results
    F->>O: Run Model Inference
    O-->>F: Model Output
    F-->>G: Evaluation Results
    G-->>U: Display Results
```

### Enterprise Workflow Integration

```mermaid
graph TB
    subgraph "Enterprise Workflow"
        A[AI Architect] --> B[Custom Model Creation]
        B --> C[QLoRA Fine-Tuning]
        C --> D[Model Registry]
        
        E[Model Evaluation Engineer] --> F[Model Testing]
        F --> G[Performance Profiling]
        G --> H[Factory Roster]
        
        D --> F
        H --> I[Production Deployment]
    end
    
    subgraph "Supporting Services"
        J[MLflow] --> K[Experiment Tracking]
        L[ChromaDB] --> M[Vector Search]
        N[Neo4j] --> O[Knowledge Graph]
        P[Monitoring] --> Q[Observability]
    end
    
    B --> J
    F --> L
    I --> N
    I --> P
    
    style A fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    style E fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style I fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
```

---

**Last Updated**: January 19, 2025  
**Version**: 2.1.0  
**Status**: Production Ready  
**Integration**: Full Enterprise Architecture
