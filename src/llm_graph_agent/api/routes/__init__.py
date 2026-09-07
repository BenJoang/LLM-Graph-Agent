"""API 路由聚合。"""
from llm_graph_agent.api.routes.agent import router as agent_router
from llm_graph_agent.api.routes.health import router as health_router

__all__ = ["agent_router", "health_router"]
