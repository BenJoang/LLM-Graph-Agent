"""全项目唯一的路径定义。替代散落在各文件的 Path(__file__).resolve().parents[N]。"""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]      # llm_graph_agent/ → 仓库根
CONFIG_DIR = Path(os.getenv("LLM_GRAPH_CONFIG_DIR", PROJECT_ROOT / "config"))
OUTPUT_DIR = Path(os.getenv("LLM_GRAPH_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
DATA_DIR   = Path(os.getenv("LLM_GRAPH_DATA_DIR", PROJECT_ROOT / "data"))
