"""FastAPI 应用组装。

create_app 接受 runner_factory（返回 AgentRunner 的可调用对象）；
默认用 api.deps.make_runner_factory 的真实运行时，测试可注入假图 factory。
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from llm_graph_agent.api.deps import AgentRunnerFactory, make_runner_factory
from llm_graph_agent.api.routes import agent_router, health_router


def create_app(
    runner_factory: AgentRunnerFactory | None = None,
) -> FastAPI:
    factory = runner_factory or make_runner_factory()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        runner = factory()
        if getattr(runner, "store", None) is not None:
            await runner.close()

    app = FastAPI(
        title="LLM Graph Agent",
        description="业务级 LangGraph Agent 运行时接口",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.runner_factory = factory

    app.include_router(health_router)
    app.include_router(agent_router)

    return app


app = create_app()
