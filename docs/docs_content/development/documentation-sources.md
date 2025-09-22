# Documentation Sources & Architecture

## 🎯 Overview

This document outlines the architecture and sources of all documentation in the Lenovo AAITC Solutions platform. Our documentation follows a unified approach with clear source attribution and embedded API documentation.

## 📚 Documentation Architecture

### Master Documentation Site (MkDocs)

- **Port**: 8082
- **URL**: http://localhost:8082
- **Source**: `docs/docs_content/` directory
- **Build Tool**: MkDocs with Material theme
- **Purpose**: Unified documentation hub for all platform components

### Documentation Sources

#### 1. **FastAPI Enterprise Platform**

- **Source**: `src/enterprise_llmops/` Python modules
- **API Docs**: http://localhost:8080/docs (Swagger UI)
- **Embedded In**: MkDocs via iframe and API references
- **Documentation Files**:
  - `docs/docs_content/api/fastapi-enterprise.md`
  - `docs/docs_content/api/chromadb-integration.md`

#### 2. **Gradio Model Evaluation App**

- **Source**: `src/gradio_app/` Python modules
- **Live App**: http://localhost:7860
- **Embedded In**: MkDocs via live app integration
- **Documentation Files**:
  - `docs/docs_content/api/gradio-model-evaluation.md`
  - `docs/docs_content/assignments/assignment1/`

#### 3. **Native Documentation**

- **Source**: `docs/docs_content/` directory
- **Purpose**: Platform overview, architecture, and user guides
- **Categories**:
  - `category1/` - Model Enablement & UX Evaluation
  - `category2/` - AI System Architecture & MLOps
  - `development/` - Setup, testing, deployment
  - `resources/` - Architecture diagrams, troubleshooting

#### 4. **Generated Documentation**

- **Source**: Auto-generated from code comments and OpenAPI specs
- **Integration**: Embedded in MkDocs via plugins
- **Examples**: API schemas, model configurations

## 🔄 Integration Strategy

### FastAPI Docs Integration

The FastAPI documentation is embedded into MkDocs through:

1. **Iframe Integration**: Direct embedding of Swagger UI
2. **API Reference Pages**: Manually maintained API documentation
3. **Cross-References**: Links between MkDocs and FastAPI docs

```markdown
<!-- In MkDocs pages -->
<div class="api-docs-container">
    <iframe 
        src="http://localhost:8080/docs" 
        width="100%" 
        height="800px"
        frameborder="0">
    </iframe>
</div>
```

### Gradio App Integration

The Gradio app is integrated through:

1. **Live App Embedding**: Direct iframe integration
2. **Demo Sections**: Interactive examples in documentation
3. **Screenshot Integration**: Static images for offline viewing

### Source Attribution

Each documentation section includes clear source attribution:

```markdown
<div class="source-attribution">
    <strong>Source:</strong> 
    <code>src/enterprise_llmops/frontend/fastapi_app.py</code>
    <br>
    <strong>API Endpoint:</strong> 
    <code>http://localhost:8080/docs</code>
    <br>
    <strong>Last Updated:</strong> January 19, 2025
</div>
```

## 📁 Directory Structure

```
docs/
├── docs_content/                    # Main documentation content
│   ├── category1/                   # Model Enablement & UX Evaluation
│   ├── category2/                   # AI System Architecture & MLOps
│   ├── api/                         # API documentation (manually maintained)
│   │   ├── fastapi-enterprise.md    # FastAPI platform docs
│   │   ├── chromadb-integration.md  # ChromaDB integration docs
│   │   └── gradio-model-evaluation.md # Gradio app docs
│   ├── assignments/                 # Assignment-specific documentation
│   ├── development/                 # Development guides
│   │   └── documentation-sources.md # This file
│   ├── resources/                   # Architecture diagrams, troubleshooting
│   └── live-applications/           # Live app demos
├── assets/                          # Images, CSS, JS files
├── mkdocs.yml                       # MkDocs configuration
└── site/                           # Generated site (auto-generated)

src/
├── enterprise_llmops/              # FastAPI source code
│   └── frontend/fastapi_app.py    # Auto-generates API docs at /docs
├── gradio_app/                     # Gradio source code
│   └── main.py                     # Auto-generates app at /7860
└── ai_architecture/                # Additional source code
```

## 🌐 Service Port Mapping

| Service                | Port | URL                   | Documentation Source                            |
| ---------------------- | ---- | --------------------- | ----------------------------------------------- |
| **FastAPI Enterprise** | 8080 | http://localhost:8080 | `src/enterprise_llmops/` + Manual docs          |
| **Gradio App**         | 7860 | http://localhost:7860 | `src/gradio_app/` + Manual docs                 |
| **MLflow**             | 5000 | http://localhost:5000 | External docs + Integration guides              |
| **ChromaDB**           | 8081 | http://localhost:8081 | `docs/docs_content/api/chromadb-integration.md` |
| **MkDocs**             | 8082 | http://localhost:8082 | `docs/docs_content/` (Master docs)              |

## 🔧 Maintenance Workflow

### Adding New Documentation

1. **For FastAPI endpoints**:

   - Add code documentation in Python files
   - Update `docs/docs_content/api/fastapi-enterprise.md`
   - Test API docs at http://localhost:8080/docs

2. **For Gradio features**:

   - Add code documentation in Python files
   - Update `docs/docs_content/api/gradio-model-evaluation.md`
   - Test app at http://localhost:7860

3. **For platform documentation**:
   - Add to appropriate category in `docs/docs_content/`
   - Update `docs/mkdocs.yml` navigation if needed
   - Test at http://localhost:8082

### Documentation Sync Process

1. **Code Changes**: Update source code with proper docstrings
2. **Manual Docs**: Update corresponding `.md` files in `docs/docs_content/`
3. **Build Test**: Test both individual services and unified MkDocs site
4. **Cross-Reference**: Ensure links between different documentation sources work

## 📋 Quality Assurance

### Documentation Standards

- **Source Attribution**: Every section must indicate its source
- **Cross-References**: Links between related documentation
- **Version Control**: All docs must include "Last Updated" dates
- **Consistency**: Uniform formatting and terminology across all sources

### Testing Checklist

- [ ] FastAPI docs accessible at http://localhost:8080/docs
- [ ] Gradio app accessible at http://localhost:7860
- [ ] MkDocs site accessible at http://localhost:8082
- [ ] All embedded iframes load correctly
- [ ] Cross-references work between documentation sources
- [ ] Source attribution is clear and accurate

## 🚀 Deployment

### Local Development

```bash
# Start all services
mkdocs serve --dev-addr 0.0.0.0:8082  # Master docs
python -m src.enterprise_llmops.main --port 8080  # FastAPI + API docs
python -m src.gradio_app.main --port 7860  # Gradio app
chroma run --port 8081 --path chroma_data  # ChromaDB
```

### Production Deployment

- **GitHub Pages**: MkDocs site deployed to https://s-n00b.github.io/ai_assignments
- **API Documentation**: Embedded via iframe or static exports
- **Live Apps**: Deployed separately with links from master docs

## 📞 Support

For documentation issues:

1. **FastAPI Docs**: Check http://localhost:8080/docs
2. **Gradio App**: Check http://localhost:7860
3. **Master Docs**: Check http://localhost:8082
4. **Source Code**: Review `src/` directory for code documentation
5. **Manual Docs**: Review `docs/docs_content/` for written documentation

---

**Last Updated**: January 19, 2025  
**Version**: 1.0.0  
**Status**: Production Ready
