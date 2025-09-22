"""
Neo4j API Endpoints for Enterprise LLMOps Platform

This module provides FastAPI endpoints for Neo4j operations,
including graph queries, GraphRAG functionality, and analytics.

Key Features:
- Graph query execution endpoints
- GraphRAG query endpoints
- Organizational structure endpoints
- B2B client data endpoints
- Custom query execution
- Health and monitoring endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging

from ..neo4j_service import (
    Neo4jService, 
    get_neo4j_service, 
    GraphQuery, 
    QueryType
)

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/neo4j", tags=["Neo4j"])


class QueryRequest(BaseModel):
    """Request model for custom queries"""
    query: str = Field(..., description="Cypher query to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    timeout: int = Field(default=30, description="Query timeout in seconds")


class GraphRAGRequest(BaseModel):
    """Request model for GraphRAG queries"""
    query: str = Field(..., description="Natural language query")
    limit: int = Field(default=10, description="Maximum number of results")


class QueryResponse(BaseModel):
    """Response model for query results"""
    data: List[Dict[str, Any]]
    summary: Dict[str, Any]
    execution_time: float
    query_type: str


class HealthResponse(BaseModel):
    """Response model for health status"""
    status: str
    uri: str
    database: str
    timestamp: str
    query_response: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None


class DatabaseInfoResponse(BaseModel):
    """Response model for database information"""
    node_count: int
    relationship_count: int
    labels: List[str]
    relationship_types: List[str]
    indexes: List[Dict[str, Any]]


@router.get("/health", response_model=HealthResponse)
async def get_neo4j_health(service: Neo4jService = Depends(get_neo4j_service)):
    """Get Neo4j service health status"""
    try:
        health_status = await service.get_health_status()
        return HealthResponse(**health_status)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/info", response_model=DatabaseInfoResponse)
async def get_database_info(service: Neo4jService = Depends(get_neo4j_service)):
    """Get Neo4j database information"""
    try:
        db_info = await service.get_database_info()
        return DatabaseInfoResponse(**db_info)
    except Exception as e:
        logger.error(f"Database info retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database info retrieval failed: {str(e)}")


@router.post("/query", response_model=QueryResponse)
async def execute_query(
    query_request: QueryRequest,
    service: Neo4jService = Depends(get_neo4j_service)
):
    """Execute custom Cypher query"""
    try:
        graph_query = GraphQuery(
            query=query_request.query,
            parameters=query_request.parameters,
            query_type=QueryType.READ,
            timeout=query_request.timeout
        )
        
        result = await service.execute_query(graph_query)
        
        return QueryResponse(
            data=result.data,
            summary=result.summary,
            execution_time=result.execution_time,
            query_type=result.query_type.value
        )
        
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")


@router.post("/graphrag", response_model=Dict[str, Any])
async def execute_graphrag_query(
    graphrag_request: GraphRAGRequest,
    service: Neo4jService = Depends(get_neo4j_service)
):
    """Execute GraphRAG query"""
    try:
        result = await service.execute_graphrag_query(
            query_text=graphrag_request.query,
            limit=graphrag_request.limit
        )
        return result
        
    except Exception as e:
        logger.error(f"GraphRAG query failed: {e}")
        raise HTTPException(status_code=500, detail=f"GraphRAG query failed: {str(e)}")


@router.get("/org-structure", response_model=Dict[str, Any])
async def get_organizational_structure(service: Neo4jService = Depends(get_neo4j_service)):
    """Get Lenovo organizational structure"""
    try:
        result = await service.get_organizational_structure()
        return result
        
    except Exception as e:
        logger.error(f"Organizational structure retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Organizational structure retrieval failed: {str(e)}")


@router.get("/b2b-clients", response_model=Dict[str, Any])
async def get_b2b_client_data(service: Neo4jService = Depends(get_neo4j_service)):
    """Get B2B client data and relationships"""
    try:
        result = await service.get_b2b_client_data()
        return result
        
    except Exception as e:
        logger.error(f"B2B client data retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"B2B client data retrieval failed: {str(e)}")


@router.get("/project-dependencies", response_model=Dict[str, Any])
async def get_project_dependencies(service: Neo4jService = Depends(get_neo4j_service)):
    """Get project dependency network"""
    try:
        result = await service.get_project_dependencies()
        return result
        
    except Exception as e:
        logger.error(f"Project dependencies retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Project dependencies retrieval failed: {str(e)}")


@router.get("/employees")
async def get_employees(
    limit: int = Query(default=50, description="Maximum number of employees to return"),
    department: Optional[str] = Query(default=None, description="Filter by department"),
    service: Neo4jService = Depends(get_neo4j_service)
):
    """Get employee information"""
    try:
        query = "MATCH (p:person)"
        parameters = {}
        
        if department:
            query += "-[:works_in]->(d:department {name: $department})"
            parameters["department"] = department
        
        query += " RETURN p LIMIT $limit"
        parameters["limit"] = limit
        
        graph_query = GraphQuery(
            query=query,
            parameters=parameters,
            query_type=QueryType.READ
        )
        
        result = await service.execute_query(graph_query)
        return {
            "employees": result.data,
            "total_returned": len(result.data),
            "execution_time": result.execution_time
        }
        
    except Exception as e:
        logger.error(f"Employee retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Employee retrieval failed: {str(e)}")


@router.get("/departments")
async def get_departments(service: Neo4jService = Depends(get_neo4j_service)):
    """Get department information with employee counts"""
    try:
        query = """
        MATCH (d:department)
        OPTIONAL MATCH (d)<-[:works_in]-(p:person)
        RETURN d, count(p) as employee_count
        ORDER BY employee_count DESC
        """
        
        graph_query = GraphQuery(
            query=query,
            parameters={},
            query_type=QueryType.READ
        )
        
        result = await service.execute_query(graph_query)
        return {
            "departments": result.data,
            "total_departments": len(result.data),
            "execution_time": result.execution_time
        }
        
    except Exception as e:
        logger.error(f"Department retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Department retrieval failed: {str(e)}")


@router.get("/projects")
async def get_projects(
    status: Optional[str] = Query(default=None, description="Filter by project status"),
    limit: int = Query(default=50, description="Maximum number of projects to return"),
    service: Neo4jService = Depends(get_neo4j_service)
):
    """Get project information"""
    try:
        query = "MATCH (p:project)"
        parameters = {}
        
        if status:
            query += " WHERE p.status = $status"
            parameters["status"] = status
        
        query += """
        OPTIONAL MATCH (p)<-[:participates_in]-(emp:person)
        RETURN p, collect(emp) as team_members
        ORDER BY p.priority DESC, p.start_date DESC
        LIMIT $limit
        """
        parameters["limit"] = limit
        
        graph_query = GraphQuery(
            query=query,
            parameters=parameters,
            query_type=QueryType.READ
        )
        
        result = await service.execute_query(graph_query)
        return {
            "projects": result.data,
            "total_returned": len(result.data),
            "execution_time": result.execution_time
        }
        
    except Exception as e:
        logger.error(f"Project retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Project retrieval failed: {str(e)}")


@router.get("/skills")
async def get_skills(
    category: Optional[str] = Query(default=None, description="Filter by skill category"),
    service: Neo4jService = Depends(get_neo4j_service)
):
    """Get skills and certifications information"""
    try:
        query = "MATCH (s:skill)"
        parameters = {}
        
        if category:
            query += " WHERE s.category = $category"
            parameters["category"] = category
        
        query += """
        OPTIONAL MATCH (s)<-[:has_skill]-(p:person)
        RETURN s, count(p) as employee_count
        ORDER BY employee_count DESC
        """
        
        graph_query = GraphQuery(
            query=query,
            parameters=parameters,
            query_type=QueryType.READ
        )
        
        result = await service.execute_query(graph_query)
        return {
            "skills": result.data,
            "total_skills": len(result.data),
            "execution_time": result.execution_time
        }
        
    except Exception as e:
        logger.error(f"Skills retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Skills retrieval failed: {str(e)}")


@router.get("/analytics/org-chart")
async def get_org_chart_analytics(service: Neo4jService = Depends(get_neo4j_service)):
    """Get organizational chart analytics"""
    try:
        query = """
        MATCH (p:person)-[:reports_to]->(m:person)
        WITH m, collect(p) as direct_reports
        RETURN m, size(direct_reports) as report_count, direct_reports
        ORDER BY report_count DESC
        LIMIT 20
        """
        
        graph_query = GraphQuery(
            query=query,
            parameters={},
            query_type=QueryType.READ
        )
        
        result = await service.execute_query(graph_query)
        return {
            "management_hierarchy": result.data,
            "execution_time": result.execution_time
        }
        
    except Exception as e:
        logger.error(f"Org chart analytics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Org chart analytics failed: {str(e)}")


@router.get("/analytics/project-metrics")
async def get_project_metrics(service: Neo4jService = Depends(get_neo4j_service)):
    """Get project metrics and analytics"""
    try:
        query = """
        MATCH (p:project)
        RETURN 
            p.status as status,
            count(p) as count,
            avg(p.budget) as avg_budget,
            collect(p.name)[0..5] as sample_projects
        ORDER BY count DESC
        """
        
        graph_query = GraphQuery(
            query=query,
            parameters={},
            query_type=QueryType.READ
        )
        
        result = await service.execute_query(graph_query)
        return {
            "project_metrics": result.data,
            "execution_time": result.execution_time
        }
        
    except Exception as e:
        logger.error(f"Project metrics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Project metrics failed: {str(e)}")


@router.get("/analytics/skill-gaps")
async def get_skill_gap_analysis(service: Neo4jService = Depends(get_neo4j_service)):
    """Get skill gap analysis"""
    try:
        query = """
        MATCH (s:skill)
        OPTIONAL MATCH (s)<-[:has_skill]-(p:person)
        WITH s, count(p) as employee_count
        WHERE employee_count < 5  // Skills with fewer than 5 people
        RETURN s, employee_count
        ORDER BY employee_count ASC
        LIMIT 20
        """
        
        graph_query = GraphQuery(
            query=query,
            parameters={},
            query_type=QueryType.READ
        )
        
        result = await service.execute_query(graph_query)
        return {
            "skill_gaps": result.data,
            "total_gaps": len(result.data),
            "execution_time": result.execution_time
        }
        
    except Exception as e:
        logger.error(f"Skill gap analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Skill gap analysis failed: {str(e)}")


# Export router for inclusion in main app
__all__ = ["router"]
