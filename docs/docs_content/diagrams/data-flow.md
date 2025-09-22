# Data Flow Diagrams

## 🎯 Overview

This section contains comprehensive data flow diagrams showing how data moves through the Lenovo AAITC platform, from user input to model output and storage.

## 🔄 Complete Data Flow Architecture

### End-to-End Data Flow

```mermaid
flowchart TD
    A[User Input] --> B[FastAPI Platform]
    B --> C[Request Processing]
    C --> D[Authentication & Authorization]
    D --> E[Service Routing]
    
    E --> F[Model Evaluation Request]
    E --> G[Knowledge Graph Query]
    E --> H[Experiment Tracking]
    E --> I[Vector Search]
    
    F --> J[Gradio App]
    J --> K[Model Selection]
    K --> L[Ollama Inference]
    L --> M[Result Processing]
    
    G --> N[Neo4j Query]
    N --> O[Graph Traversal]
    O --> P[Context Retrieval]
    
    H --> Q[MLflow Logging]
    Q --> R[Experiment Metadata]
    R --> S[Model Registry]
    
    I --> T[ChromaDB Search]
    T --> U[Vector Similarity]
    U --> V[Document Retrieval]
    
    M --> W[Response Aggregation]
    P --> W
    S --> W
    V --> W
    
    W --> X[Response Formatting]
    X --> Y[User Output]
    
    Y --> Z[Monitoring & Metrics]
    Z --> AA[Prometheus]
    Z --> BB[Grafana]
    Z --> CC[LangFuse]
    
    style A fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    style B fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style J fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    style L fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style M fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    style N fill:#e0f2f1,stroke:#004d40,stroke-width:2px
    style Q fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    style T fill:#fff8e1,stroke:#ff6f00,stroke-width:2px
    style W fill:#e3f2fd,stroke:#0d47a1,stroke-width:2px
    style Y fill:#f9fbe7,stroke:#827717,stroke-width:2px
    style Z fill:#fce4ec,stroke:#ad1457,stroke-width:2px
```

### Model Evaluation Data Flow

```mermaid
sequenceDiagram
    participant U as User
    participant G as Gradio App
    participant F as FastAPI Platform
    participant O as Ollama
    participant M as MLflow
    participant C as ChromaDB
    participant N as Neo4j
    participant P as Prometheus
    
    U->>G: Submit Evaluation Request
    G->>F: Process Request
    F->>O: Get Model List
    O-->>F: Available Models
    F-->>G: Model Options
    
    U->>G: Select Model & Task
    G->>F: Submit Evaluation
    F->>O: Run Model Inference
    O-->>F: Model Output
    
    F->>M: Log Experiment Start
    F->>C: Store Input Data
    F->>N: Update Knowledge Graph
    
    F->>O: Process Results
    O-->>F: Processed Output
    
    F->>M: Log Experiment Results
    F->>C: Store Output Data
    F->>N: Update Graph with Results
    F->>P: Record Performance Metrics
    
    F-->>G: Evaluation Results
    G-->>U: Display Results
```

### Knowledge Graph Data Flow

```mermaid
graph TB
    subgraph "Neo4j Knowledge Graph Data Flow"
        A[Document Input] --> B[Text Processing]
        B --> C[Entity Recognition]
        C --> D[Relationship Extraction]
        D --> E[Graph Construction]
        
        E --> F[Knowledge Graph]
        F --> G[Graph Queries]
        G --> H[Context Retrieval]
        H --> I[RAG Enhancement]
        
        J[Faker Data Generation] --> K[Realistic Entities]
        K --> L[Relationship Patterns]
        L --> M[Graph Population]
        M --> F
        
        N[User Queries] --> O[Natural Language Processing]
        O --> P[Query Translation]
        P --> Q[Graph Traversal]
        Q --> R[Result Aggregation]
        R --> S[Response Generation]
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
    style M fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px
    style N fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
    style O fill:#e0f2f1,stroke:#009688,stroke-width:2px
    style P fill:#fff3e0,stroke:#ff9800,stroke-width:2px
    style Q fill:#fce4ec,stroke:#e91e63,stroke-width:2px
    style R fill:#e1f5fe,stroke:#00bcd4,stroke-width:2px
    style S fill:#f1f8e9,stroke:#8bc34a,stroke-width:2px
```

### Vector Database Data Flow

```mermaid
graph LR
    subgraph "ChromaDB Vector Data Flow"
        A[Document Input] --> B[Text Chunking]
        B --> C[Embedding Generation]
        C --> D[Vector Storage]
        
        E[Query Input] --> F[Query Embedding]
        F --> G[Vector Similarity Search]
        G --> H[Result Ranking]
        H --> I[Context Retrieval]
        
        J[Model Evaluation Results] --> K[Result Embedding]
        K --> L[Vector Indexing]
        L --> M[Searchable Knowledge]
        
        D --> N[Vector Database]
        M --> N
        N --> G
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
    style M fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px
    style N fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
```

## 🔄 Experiment Tracking Data Flow

### MLflow Experiment Data Flow

```mermaid
graph TB
    subgraph "MLflow Experiment Tracking"
        A[Model Evaluation Start] --> B[Experiment Creation]
        B --> C[Run Initialization]
        C --> D[Parameter Logging]
        D --> E[Metric Collection]
        
        E --> F[Model Training/Inference]
        F --> G[Performance Metrics]
        G --> H[Artifact Storage]
        H --> I[Model Registry]
        
        J[Experiment Comparison] --> K[Metric Analysis]
        K --> L[Model Selection]
        L --> M[Production Deployment]
        
        N[Model Versioning] --> O[Artifact Management]
        O --> P[Deployment Tracking]
        P --> Q[Performance Monitoring]
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
    style M fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px
    style N fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
    style O fill:#e0f2f1,stroke:#009688,stroke-width:2px
    style P fill:#fff3e0,stroke:#ff9800,stroke-width:2px
    style Q fill:#fce4ec,stroke:#e91e63,stroke-width:2px
```

### QLoRA Fine-Tuning Data Flow

```mermaid
graph TB
    subgraph "QLoRA Fine-Tuning Data Flow"
        A[Base Model] --> B[Dataset Preparation]
        B --> C[Training Configuration]
        C --> D[QLoRA Adapter Creation]
        
        D --> E[Fine-Tuning Process]
        E --> F[Gradient Updates]
        F --> G[Adapter Weights]
        G --> H[Performance Evaluation]
        
        H --> I{Meets Requirements?}
        I -->|Yes| J[Adapter Registry]
        I -->|No| K[Hyperparameter Tuning]
        
        K --> E
        J --> L[Model Composition]
        L --> M[Custom MoE Architecture]
        
        M --> N[Production Deployment]
        N --> O[Performance Monitoring]
        O --> P[Continuous Improvement]
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
    style M fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px
    style N fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
    style O fill:#e0f2f1,stroke:#009688,stroke-width:2px
    style P fill:#fff3e0,stroke:#ff9800,stroke-width:2px
```

## 📊 Monitoring & Observability Data Flow

### Metrics Collection Flow

```mermaid
graph LR
    subgraph "Monitoring Data Flow"
        A[Application Metrics] --> B[Prometheus Collection]
        B --> C[Metric Storage]
        C --> D[Grafana Visualization]
        
        E[LLM Interactions] --> F[LangFuse Tracking]
        F --> G[Trace Storage]
        G --> H[Performance Analysis]
        
        I[System Metrics] --> J[Infrastructure Monitoring]
        J --> K[Alert Generation]
        K --> L[Notification System]
        
        M[User Interactions] --> N[Usage Analytics]
        N --> O[Business Metrics]
        O --> P[ROI Analysis]
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
    style M fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px
    style N fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
    style O fill:#e0f2f1,stroke:#009688,stroke-width:2px
    style P fill:#fff3e0,stroke:#ff9800,stroke-width:2px
```

### Real-time Data Processing

```mermaid
graph TB
    subgraph "Real-time Data Processing"
        A[User Input] --> B[Stream Processing]
        B --> C[Data Validation]
        C --> D[Feature Extraction]
        D --> E[Model Inference]
        
        E --> F[Result Processing]
        F --> G[Response Generation]
        G --> H[User Output]
        
        I[Background Processing] --> J[Data Aggregation]
        J --> K[Analytics Computation]
        K --> L[Insight Generation]
        
        M[Event Streaming] --> N[Real-time Updates]
        N --> O[Dashboard Refresh]
        O --> P[Live Monitoring]
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
    style M fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px
    style N fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
    style O fill:#e0f2f1,stroke:#009688,stroke-width:2px
    style P fill:#fff3e0,stroke:#ff9800,stroke-width:2px
```

---

**Last Updated**: January 19, 2025  
**Version**: 2.1.0  
**Status**: Production Ready  
**Integration**: Complete Data Flow Architecture
