# LLM Graph Agent

一 LangGraph Agent 运行时：从原有项目收敛迁移而来，只保留可复用的通用内核，并统一为全 async 通道。

架构对标 opencode：**纯逻辑与副作用分离、单一编排入口、分层配置发现**。

## 功能模块

```
llm_graph_agent/
├── config.py           分层配置发现（env → 用户目录 → 项目 config/）
├── paths.py            统一路径（项目根 / 配置 / 输出 / 数据）
├── llm/                模型连接 + 家族抽象 + profile/prompt 加载
├── context/            上下文：消息状态、系统提示词组装、重试、压缩
│   └── compression/    压缩引擎（segment/serialize/collapse/snip/overflow）
├── tools/              通用工具集 + 注册表（11 个工具，含递归子代理）
├── graph/              agent 循环：tool_agent（通用）+ sub_agent（子代理）
│   └── common.py       单一编排入口（压缩 + 重试 + 系统提示，对标 processor.ts）
├── runner.py           AgentRunner：图运行 + 对话持久化 + 流式/取消
├── api/                干净 JSON + SSE 接口（无 GUI）
├── rag/                知识库检索（chunking → embedding → 混合检索）
├── persistence/        对话日志存储（SQLite / Postgres 双后端）
└── observability/      Phoenix 追踪
```

## 快速开始

推荐用 [uv](https://docs.astral.sh/uv/) 管理项目内虚拟环境（`.venv`）。
Python 版本由 `.python-version` 固定，依赖由 `uv.lock` 锁定。

```bash
# 1) 创建 .venv 并装齐全部依赖（含所有 optional extras）
uv sync --all-extras

# 只装运行时 + API：
uv sync --extra api

# 不用 uv 的等价做法（需 Python >=3.12）：
# python -m venv .venv && .venv/Scripts/pip install -e ".[dev,api,rag,postgres,legacy-checkpoints,observability]"

# 2) 配置模型（复制 .env.example 到 .env 并填写 key）
cp .env.example .env

# 3) 启动 API（默认 127.0.0.1:8200）
uv run python -m llm_graph_agent.api
# 或命令行入口
uv run llm-graph-agent-api
```

> `legacy-checkpoints` extra 提供 `langgraph-checkpoint-sqlite` / `aiosqlite`，
> 持久化层依赖它——跑测试时缺它会在 collection 阶段报
> `No module named 'langgraph.checkpoint.sqlite'`。

启动后：

```bash
# 健康检查
curl http://127.0.0.1:8200/health

# 单轮问答（JSON）
curl -X POST http://127.0.0.1:8200/agent/tool \
  -H "Content-Type: application/json" \
  -d '{"question": "总结一下当前目录", "session_id": "sess-1"}'

# 流式执行（SSE）
curl -N -X POST http://127.0.0.1:8200/agent/turn \
  -H "Content-Type: application/json" \
  -d '{"question": "帮我查一下项目结构"}'
```

## 配置

配置采用分层发现（`config.find_config_file`），搜索顺序：
1. `LLM_GRAPH_USER_CONFIG_PATH` / `LLM_GRAPH_PROMPT_CONFIG_PATH` 指向的路径
2. `LLM_GRAPH_CONFIG_DIR` 指向的目录
3. 用户配置目录（`~/.config/llm-graph-agent/`）
4. 项目 `config/` 目录

- **模型 profile**：`config/user_config.json` 的 `profiles`（模型名、base_url/api_key env 变量、生成参数）
- **提示词**：`config/prompt_config.json` + `config/prompts/*.md`
- **持久化**：`LLM_GRAPH_DATABASE_URL`（`sqlite:///` 或 `postgresql://`）
- **RAG**：`RAG_POSTGRES_URL` + `RAG_EMBEDDING_*`（OpenAI 兼容 embedding 服务）

## 作为库使用

```python
import asyncio
from llm_graph_agent.graph.tool_agent import build_graph
from llm_graph_agent.runner import AgentRunner

async def main():
    runner = AgentRunner(graph_builder=build_graph, profile_name="qwen3.6")
    result = await runner.run(question="你好", session_id="sess-1")
    print(result["messages"][-1].content)

asyncio.run(main())
```

## 测试

```bash
uv run pytest tests -q
# 或直接用项目内解释器
.venv/Scripts/python -m pytest tests -q

# Postgres 契约测试需：设置 LLM_GRAPH_TEST_POSTGRES=1 并配置 postgres
```

